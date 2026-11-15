//! Allocation-free damped modal oscillator bank, HRIR/RIR additional-delay convolution,
//! cubic Lagrange fractional delay, fluid impedance reflection/transmission, and
//! status-aware perceptual feature bridges.

use crate::{
    audio::{
        convolve_fir_into, spherical_ild_db, stft_frame_into, stft_one_sided_magnitudes_into,
    },
    bs1770::QualityStatus,
    filter::{exp_f64, ln_f64, sin_cos_f64, sqrt_f64},
    hoa::{decode_hoa_frame_to_direction_f64, HoaNormalization, HoaOrder},
    loudness::dct_ii_into,
    mel::{mel_apply_into, MelShape},
    pitch::{
        yin_absolute_threshold, yin_cmndf_in_place, yin_difference_into, yin_parabolic_min,
        yin_pitch_confidence,
    },
    SignalError,
};

pub const LN_1000: f64 = 6.907_755_278_982_137;
pub const MODAL_RESONANCE_RECORD_BYTES: usize = 32;

pub const OCTAHEDRAL_VIRTUAL_SPEAKERS_XYZ: [[f64; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModalResonanceRecord {
    pub frequency_hz: f32,
    pub amplitude_pa: f32,
    pub decay_rate_s1: f32,
    pub phase_rad: f32,
    pub mode_shape_index: u32,
    pub material_ref: u32,
    pub excitation_index: u32,
    pub reserved: u32,
}

impl ModalResonanceRecord {
    pub fn validate(&self) -> Result<(), SignalError> {
        if !self.frequency_hz.is_finite()
            || !self.amplitude_pa.is_finite()
            || !self.decay_rate_s1.is_finite()
            || !self.phase_rad.is_finite()
        {
            return Err(SignalError::NonFiniteInput);
        }
        if self.frequency_hz <= 0.0 || self.decay_rate_s1 < 0.0 || self.reserved != 0 {
            return Err(SignalError::InvalidParameter);
        }
        Ok(())
    }

    pub fn to_le_bytes(&self) -> Result<[u8; MODAL_RESONANCE_RECORD_BYTES], SignalError> {
        self.validate()?;
        let mut out = [0u8; MODAL_RESONANCE_RECORD_BYTES];
        out[0..4].copy_from_slice(&self.frequency_hz.to_le_bytes());
        out[4..8].copy_from_slice(&self.amplitude_pa.to_le_bytes());
        out[8..12].copy_from_slice(&self.decay_rate_s1.to_le_bytes());
        out[12..16].copy_from_slice(&self.phase_rad.to_le_bytes());
        out[16..20].copy_from_slice(&self.mode_shape_index.to_le_bytes());
        out[20..24].copy_from_slice(&self.material_ref.to_le_bytes());
        out[24..28].copy_from_slice(&self.excitation_index.to_le_bytes());
        out[28..32].copy_from_slice(&self.reserved.to_le_bytes());
        Ok(out)
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, SignalError> {
        if bytes.len() != MODAL_RESONANCE_RECORD_BYTES {
            return Err(SignalError::LengthMismatch);
        }
        let rec = Self {
            frequency_hz: f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            amplitude_pa: f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            decay_rate_s1: f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            phase_rad: f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            mode_shape_index: u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            material_ref: u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
            excitation_index: u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            reserved: u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
        };
        rec.validate()?;
        Ok(rec)
    }

    pub fn t60_seconds(&self) -> Result<(Option<f64>, QualityStatus), SignalError> {
        self.validate()?;
        modal_t60_seconds(f64::from(self.decay_rate_s1))
    }

    pub fn evaluate_at_time_f64(&self, time_seconds: f64) -> Result<f64, SignalError> {
        self.validate()?;
        if !time_seconds.is_finite() {
            return Err(SignalError::NonFiniteInput);
        }
        if time_seconds < 0.0 {
            return Err(SignalError::InvalidParameter);
        }
        let amp = f64::from(self.amplitude_pa);
        let decay = f64::from(self.decay_rate_s1);
        let freq = f64::from(self.frequency_hz);
        let phase0 = f64::from(self.phase_rad);

        let env = exp_f64(-decay * time_seconds);
        let cycles = freq * time_seconds;
        let whole_cycles = if cycles >= 0.0 {
            (cycles as i64) as f64
        } else {
            0.0
        };
        let frac_cycles = cycles - whole_cycles;
        let angle = core::f64::consts::TAU * frac_cycles + phase0;
        let (s, _) = sin_cos_f64(angle);
        Ok(amp * env * s)
    }
}

#[inline]
pub fn modal_t60_seconds(decay_rate_s1: f64) -> Result<(Option<f64>, QualityStatus), SignalError> {
    if !decay_rate_s1.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if decay_rate_s1 < 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    if decay_rate_s1 == 0.0 {
        return Ok((None, QualityStatus::ModelUnavailable));
    }
    let t60 = LN_1000 / decay_rate_s1;
    if !t60.is_finite() {
        return Err(SignalError::NumericRange);
    }
    Ok((Some(t60), QualityStatus::Valid))
}

pub fn evaluate_modal_bank_at_time_f64(
    modes: &[ModalResonanceRecord],
    mode_shape_weights: &[f32],
    excitation_weights: &[f32],
    time_seconds: f64,
) -> Result<f64, SignalError> {
    if modes.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if mode_shape_weights.iter().any(|w| !w.is_finite())
        || excitation_weights.iter().any(|w| !w.is_finite())
    {
        return Err(SignalError::NonFiniteInput);
    }
    let mut total = 0.0f64;
    for mode in modes {
        let shape_idx = mode.mode_shape_index as usize;
        let exc_idx = mode.excitation_index as usize;
        if shape_idx >= mode_shape_weights.len() || exc_idx >= excitation_weights.len() {
            return Err(SignalError::InvalidParameter);
        }
        let coupling =
            f64::from(mode_shape_weights[shape_idx]) * f64::from(excitation_weights[exc_idx]);
        total += coupling * mode.evaluate_at_time_f64(time_seconds)?;
    }
    Ok(total)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModalOscillatorState {
    sample_rate_hz: f64,
    sample_cursor: u64,
}

impl ModalOscillatorState {
    pub fn new(sample_rate_hz: f64) -> Result<Self, SignalError> {
        if !sample_rate_hz.is_finite() {
            return Err(SignalError::NonFiniteInput);
        }
        if sample_rate_hz <= 0.0 {
            return Err(SignalError::InvalidParameter);
        }
        Ok(Self {
            sample_rate_hz,
            sample_cursor: 0,
        })
    }

    #[inline]
    pub const fn sample_cursor(&self) -> u64 {
        self.sample_cursor
    }

    pub fn synthesize_block_into(
        &mut self,
        modes: &[ModalResonanceRecord],
        mode_shape_weights: &[f32],
        excitation_weights: &[f32],
        out_samples: &mut [f32],
    ) -> Result<usize, SignalError> {
        if modes.is_empty() {
            return Err(SignalError::EmptyInput);
        }
        for mode in modes {
            mode.validate()?;
            if (mode.mode_shape_index as usize) >= mode_shape_weights.len()
                || (mode.excitation_index as usize) >= excitation_weights.len()
            {
                return Err(SignalError::InvalidParameter);
            }
        }
        let inv_fs = 1.0 / self.sample_rate_hz;
        for (offset, dst) in out_samples.iter_mut().enumerate() {
            let n = self
                .sample_cursor
                .checked_add(offset as u64)
                .ok_or(SignalError::NumericRange)?;
            let t = (n as f64) * inv_fs;
            let val = evaluate_modal_bank_at_time_f64(
                modes,
                mode_shape_weights,
                excitation_weights,
                t,
            )?;
            let val_f32 = val as f32;
            if !val_f32.is_finite() {
                return Err(SignalError::NumericRange);
            }
            *dst = val_f32;
        }
        self.sample_cursor = self
            .sample_cursor
            .checked_add(out_samples.len() as u64)
            .ok_or(SignalError::NumericRange)?;
        Ok(out_samples.len())
    }
}

pub fn convolve_hrir_with_additional_delay_into(
    signal: &[f32],
    kernel: &[f32],
    additional_delay_taps: usize,
    output: &mut [f32],
) -> Result<usize, SignalError> {
    if signal.is_empty() || kernel.is_empty() {
        return Ok(0);
    }
    if signal.iter().any(|x| !x.is_finite()) || kernel.iter().any(|h| !h.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let conv_len = signal
        .len()
        .checked_add(kernel.len())
        .and_then(|v| v.checked_sub(1))
        .ok_or(SignalError::NumericRange)?;
    let total_len = conv_len
        .checked_add(additional_delay_taps)
        .ok_or(SignalError::NumericRange)?;
    if output.len() < total_len {
        return Err(SignalError::LengthMismatch);
    }

    output[..additional_delay_taps].fill(0.0);
    convolve_fir_into(
        signal,
        kernel,
        &mut output[additional_delay_taps..total_len],
    )?;
    Ok(total_len)
}

pub fn convolve_stereo_hrir_with_additional_delays_into(
    signal: &[f32],
    left_ir: &[f32],
    right_ir: &[f32],
    additional_delays_lr: [usize; 2],
    out_left: &mut [f32],
    out_right: &mut [f32],
) -> Result<(usize, usize), SignalError> {
    let n_l = convolve_hrir_with_additional_delay_into(
        signal,
        left_ir,
        additional_delays_lr[0],
        out_left,
    )?;
    let n_r = convolve_hrir_with_additional_delay_into(
        signal,
        right_ir,
        additional_delays_lr[1],
        out_right,
    )?;
    Ok((n_l, n_r))
}

pub fn place_cubic_lagrange_fractional_impulse(
    ir_buffer: &mut [f32],
    delay_samples: f64,
    gain: f64,
) -> Result<(), SignalError> {
    if ir_buffer.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !delay_samples.is_finite() || !gain.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if delay_samples < 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    let len = ir_buffer.len();
    let i_floor = delay_samples as usize;
    let frac = delay_samples - (i_floor as f64);

    if frac.abs() <= 1e-12 {
        if i_floor >= len {
            return Err(SignalError::InvalidParameter);
        }
        ir_buffer[i_floor] = (f64::from(ir_buffer[i_floor]) + gain) as f32;
        return Ok(());
    }
    if i_floor == 0 || i_floor + 2 >= len {
        if i_floor + 1 >= len {
            return Err(SignalError::InvalidParameter);
        }
        ir_buffer[i_floor] = (f64::from(ir_buffer[i_floor]) + gain * (1.0 - frac)) as f32;
        ir_buffer[i_floor + 1] = (f64::from(ir_buffer[i_floor + 1]) + gain * frac) as f32;
        return Ok(());
    }

    let d = frac + 1.0;
    let h0 = -((d - 1.0) * (d - 2.0) * (d - 3.0)) / 6.0;
    let h1 = (d * (d - 2.0) * (d - 3.0)) / 2.0;
    let h2 = -(d * (d - 1.0) * (d - 3.0)) / 2.0;
    let h3 = (d * (d - 1.0) * (d - 2.0)) / 6.0;

    let idx0 = i_floor - 1;
    ir_buffer[idx0] = (f64::from(ir_buffer[idx0]) + gain * h0) as f32;
    ir_buffer[idx0 + 1] = (f64::from(ir_buffer[idx0 + 1]) + gain * h1) as f32;
    ir_buffer[idx0 + 2] = (f64::from(ir_buffer[idx0 + 2]) + gain * h2) as f32;
    ir_buffer[idx0 + 3] = (f64::from(ir_buffer[idx0 + 3]) + gain * h3) as f32;
    Ok(())
}

pub fn impulse_first_moment_tap_f64(ir_buffer: &[f32]) -> Result<f64, SignalError> {
    if ir_buffer.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let mut sum_w = 0.0f64;
    let mut sum_nw = 0.0f64;
    for (n, &h) in ir_buffer.iter().enumerate() {
        let hd = f64::from(h);
        if !hd.is_finite() {
            return Err(SignalError::NonFiniteInput);
        }
        sum_w += hd;
        sum_nw += (n as f64) * hd;
    }
    if sum_w.abs() <= 1e-15 {
        return Err(SignalError::InvalidParameter);
    }
    Ok(sum_nw / sum_w)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidImpedanceInterface {
    pub pressure_reflection_r: f64,
    pub pressure_transmission_t: f64,
    pub energy_reflectance_big_r: f64,
    pub energy_transmittance_big_t: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdealAcousticBoundary {
    Rigid,
    PressureRelease,
}

impl IdealAcousticBoundary {
    #[inline]
    pub const fn pressure_reflection_r(self) -> f64 {
        match self {
            Self::Rigid => 1.0,
            Self::PressureRelease => -1.0,
        }
    }
}

pub fn normal_incidence_fluid_interface_f64(
    z1_rayl: f64,
    z2_rayl: f64,
) -> Result<FluidImpedanceInterface, SignalError> {
    if !z1_rayl.is_finite() || !z2_rayl.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if z1_rayl <= 0.0 || z2_rayl <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    let denom = z2_rayl + z1_rayl;
    let r = (z2_rayl - z1_rayl) / denom;
    let t = (2.0 * z2_rayl) / denom;
    let big_r = r * r;
    let big_t = (4.0 * z1_rayl * z2_rayl) / (denom * denom);
    Ok(FluidImpedanceInterface {
        pressure_reflection_r: r,
        pressure_transmission_t: t,
        energy_reflectance_big_r: big_r,
        energy_transmittance_big_t: big_t,
    })
}

#[inline]
pub fn ray_arrival_delay_taps_f64(
    distance_m: f64,
    speed_of_sound_m_s: f64,
    sample_rate_hz: f64,
) -> Result<(f64, f64), SignalError> {
    if !distance_m.is_finite() || !speed_of_sound_m_s.is_finite() || !sample_rate_hz.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if distance_m < 0.0 || speed_of_sound_m_s <= 0.0 || sample_rate_hz <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    let delay_s = distance_m / speed_of_sound_m_s;
    Ok((delay_s, delay_s * sample_rate_hz))
}

pub fn render_hoa_stream_binaural_into(
    order: HoaOrder,
    normalization: HoaNormalization,
    hoa_interleaved: &[f32],
    reference_freq_hz: f32,
    out_stereo: &mut [[f32; 2]],
) -> Result<usize, SignalError> {
    let ch = order.channel_count();
    if !hoa_interleaved.len().is_multiple_of(ch) {
        return Err(SignalError::LengthMismatch);
    }
    let num_frames = hoa_interleaved.len() / ch;
    if out_stereo.len() < num_frames {
        return Err(SignalError::LengthMismatch);
    }
    if !reference_freq_hz.is_finite() || reference_freq_hz <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    if hoa_interleaved.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }

    let mut spk_gains_l = [0.0f64; 6];
    let mut spk_gains_r = [0.0f64; 6];
    for (idx, &[x, y, _z]) in OCTAHEDRAL_VIRTUAL_SPEAKERS_XYZ.iter().enumerate() {
        let right_az = if y > 0.5 {
            -core::f32::consts::FRAC_PI_2
        } else if y < -0.5 {
            core::f32::consts::FRAC_PI_2
        } else {
            0.0
        };
        let _ild_db = f64::from(spherical_ild_db(right_az, reference_freq_hz)?);
        let half_angle = 0.25 * core::f64::consts::PI * (1.0 - y.clamp(-1.0, 1.0));
        let (sin_h, cos_h) = sin_cos_f64(half_angle);
        let front_boost = 1.0 + 0.05 * x;
        spk_gains_l[idx] = cos_h * front_boost;
        spk_gains_r[idx] = sin_h * front_boost;
    }

    let mut frame_f64 = [0.0f64; 16];
    for (frame_idx, dst) in out_stereo.iter_mut().enumerate().take(num_frames) {
        let base = frame_idx * ch;
        for c in 0..ch {
            frame_f64[c] = f64::from(hoa_interleaved[base + c]);
        }
        let mut l_acc = 0.0f64;
        let mut r_acc = 0.0f64;
        for (idx, &dir) in OCTAHEDRAL_VIRTUAL_SPEAKERS_XYZ.iter().enumerate() {
            let spk_sig = decode_hoa_frame_to_direction_f64(
                order,
                normalization,
                &frame_f64[..ch],
                dir,
                true,
            )?;
            l_acc += spk_sig * spk_gains_l[idx];
            r_acc += spk_sig * spk_gains_r[idx];
        }
        let norm = 1.0 / sqrt_f64(6.0);
        *dst = [(l_acc * norm) as f32, (r_acc * norm) as f32];
    }

    Ok(num_frames)
}

pub fn fir_dc_and_nyquist_gain_f64(kernel: &[f32]) -> Result<(f64, f64), SignalError> {
    if kernel.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if kernel.iter().any(|h| !h.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let mut dc = 0.0f64;
    let mut nyq = 0.0f64;
    for (m, &h) in kernel.iter().enumerate() {
        let hd = f64::from(h);
        dc += hd;
        if m % 2 == 0 {
            nyq += hd;
        } else {
            nyq -= hd;
        }
    }
    Ok((dc, nyq))
}

#[inline]
pub fn apply_linear_fir_filter_into(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
) -> Result<usize, SignalError> {
    convolve_fir_into(input, kernel, output)
}

#[allow(clippy::too_many_arguments)]
pub fn extract_mel_mfcc_frame_into(
    frame: &[f32],
    n_mel: usize,
    mel_filterbank: &[f32],
    stft_interleaved_scratch: &mut [f32],
    one_sided_power_scratch: &mut [f32],
    out_mel_energies: &mut [f32],
    out_mfcc: &mut [f32],
) -> Result<QualityStatus, SignalError> {
    let n = frame.len();
    if n == 0 || n_mel == 0 {
        return Err(SignalError::EmptyInput);
    }
    let bins = n / 2 + 1;
    if one_sided_power_scratch.len() < bins
        || out_mel_energies.len() < n_mel
        || out_mfcc.len() < n_mel
    {
        return Err(SignalError::LengthMismatch);
    }

    stft_frame_into(frame, stft_interleaved_scratch)?;
    stft_one_sided_magnitudes_into(
        &stft_interleaved_scratch[..2 * n],
        &mut one_sided_power_scratch[..bins],
    )?;
    for mag in &mut one_sided_power_scratch[..bins] {
        *mag = (*mag) * (*mag);
    }

    let shape = MelShape {
        n_frames: 1,
        n_bins: bins,
        n_mel,
    };
    mel_apply_into(
        &one_sided_power_scratch[..bins],
        mel_filterbank,
        shape,
        &mut out_mel_energies[..n_mel],
    )?;

    if out_mel_energies[..n_mel].iter().all(|&e| e <= 0.0) {
        out_mfcc[..n_mel].fill(0.0);
        return Ok(QualityStatus::Silence);
    }

    const MEL_FLOOR: f64 = 1e-12;
    let mut log_mel_scratch = [0.0f32; 128];
    if n_mel > log_mel_scratch.len() {
        return Err(SignalError::InvalidParameter);
    }
    for m in 0..n_mel {
        let e = f64::from(out_mel_energies[m]).max(MEL_FLOOR);
        log_mel_scratch[m] = ln_f64(e) as f32;
    }
    let written = dct_ii_into(&log_mel_scratch[..n_mel], &mut out_mfcc[..n_mel]);
    if written != n_mel {
        return Err(SignalError::LengthMismatch);
    }
    Ok(QualityStatus::Valid)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchReading {
    pub status: QualityStatus,
    pub f0_hz: Option<f32>,
    pub confidence: f32,
}

pub fn estimate_pitch_yin_with_status(
    frame: &[f32],
    sample_rate_hz: f32,
    aperiodicity_threshold: f32,
    diff_scratch: &mut [f32],
) -> Result<PitchReading, SignalError> {
    if frame.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !sample_rate_hz.is_finite()
        || !aperiodicity_threshold.is_finite()
        || frame.iter().any(|x| !x.is_finite())
    {
        return Err(SignalError::NonFiniteInput);
    }
    if sample_rate_hz <= 0.0 || !(0.01..=0.99).contains(&aperiodicity_threshold) || frame.len() < 8
    {
        return Err(SignalError::InvalidParameter);
    }
    let max_lag = (frame.len() / 2).min(diff_scratch.len().saturating_sub(1));
    let min_lag = 2usize;
    if max_lag < min_lag + 2 {
        return Err(SignalError::LengthMismatch);
    }

    if frame.iter().all(|&x| x == 0.0) {
        return Ok(PitchReading {
            status: QualityStatus::Silence,
            f0_hz: None,
            confidence: 0.0,
        });
    }

    yin_difference_into(frame, max_lag, &mut diff_scratch[..=max_lag]);
    yin_cmndf_in_place(&mut diff_scratch[..=max_lag]);
    let tau = yin_absolute_threshold(
        &diff_scratch[..=max_lag],
        min_lag,
        max_lag,
        aperiodicity_threshold,
    );
    let (better_tau, aperiodicity) =
        yin_parabolic_min(&diff_scratch[..=max_lag], tau, min_lag, max_lag);
    let confidence = yin_pitch_confidence(aperiodicity);

    if !(better_tau.is_finite() && better_tau > 0.0)
        || diff_scratch[tau] >= aperiodicity_threshold
        || aperiodicity > aperiodicity_threshold
    {
        return Ok(PitchReading {
            status: QualityStatus::Unvoiced,
            f0_hz: None,
            confidence,
        });
    }

    let f0 = sample_rate_hz / better_tau;
    if !f0.is_finite() || f0 <= 0.0 {
        return Ok(PitchReading {
            status: QualityStatus::Unvoiced,
            f0_hz: None,
            confidence: 0.0,
        });
    }
    Ok(PitchReading {
        status: QualityStatus::Valid,
        f0_hz: Some(f0),
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modal_and_hrir_delay_oracles() {
        let mode = ModalResonanceRecord {
            frequency_hz: 100.0,
            amplitude_pa: 2.0,
            decay_rate_s1: LN_1000 as f32,
            phase_rad: core::f32::consts::FRAC_PI_2,
            mode_shape_index: 0,
            material_ref: 1,
            excitation_index: 0,
            reserved: 0,
        };
        assert!((mode.evaluate_at_time_f64(0.0).unwrap() - 2.0).abs() <= 1e-6);
        assert!((mode.evaluate_at_time_f64(1.0).unwrap() - 0.002).abs() <= 1e-6);
        assert!((mode.t60_seconds().unwrap().0.unwrap() - 1.0).abs() <= 1e-6);

        let mut out_l = [0.0f32; 8];
        let mut out_r = [0.0f32; 8];
        let (nl, nr) = convolve_stereo_hrir_with_additional_delays_into(
            &[1.0f32],
            &[1.0f32, 0.0, 0.0],
            &[0.0f32, 0.5, 0.0],
            [2, 3],
            &mut out_l,
            &mut out_r,
        )
        .unwrap();
        assert_eq!(&out_l[..nl], &[0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(&out_r[..nr], &[0.0, 0.0, 0.0, 0.0, 0.5, 0.0]);
    }
}
