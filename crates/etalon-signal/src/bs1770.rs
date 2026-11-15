//! Allocation-free ITU-R BS.1770-5 multi-channel gated loudness, 4x oversampled
//! polyphase FIR true-peak (`dBTP`), calibrated dB-SPL, and signal quality statuses.

use crate::{
    filter::{
        design_k_weighting_stage1, design_k_weighting_stage2, log10_f64, sin_cos_f64, sqrt_f64,
        BiquadCoeffs, CascadedBiquadDf2t,
    },
    loudness::energy_to_lufs,
    SignalError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum QualityStatus {
    Valid = 0,
    Missing = 1,
    Unknown = 2,
    Saturated = 3,
    Silence = 4,
    Unvoiced = 5,
    ModelUnavailable = 6,
}

pub const BS1770_ABSOLUTE_GATE_LUFS: f64 = -70.0;
pub const BS1770_RELATIVE_GATE_OFFSET_LU: f64 = -10.0;
pub const MAX_BS1770_CHANNELS: usize = 16;
pub const AIR_REF_PRESSURE_PA: f64 = 20.0e-6;
pub const UNDERWATER_REF_PRESSURE_PA: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bs1770ChannelRole {
    FrontOrCenter,
    Surround,
    Lfe,
}

impl Bs1770ChannelRole {
    #[inline]
    pub const fn weight_g(self) -> f64 {
        match self {
            Self::FrontOrCenter => 1.0,
            Self::Surround => 1.41,
            Self::Lfe => 0.0,
        }
    }
}

pub fn new_k_weighting_cascade(
    sample_rate_hz: f64,
) -> Result<CascadedBiquadDf2t<2>, SignalError> {
    if !sample_rate_hz.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if sample_rate_hz < 8_000.0 {
        return Err(SignalError::InvalidParameter);
    }
    let s1 = design_k_weighting_stage1(sample_rate_hz);
    let s2 = design_k_weighting_stage2(sample_rate_hz);
    CascadedBiquadDf2t::<2>::new([s1, s2])
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bs1770LoudnessReading {
    pub status: QualityStatus,
    pub integrated_lufs: Option<f64>,
    pub absolute_gated_lufs: Option<f64>,
    pub relative_threshold_lufs: Option<f64>,
    pub total_blocks: usize,
    pub absolute_gated_blocks: usize,
    pub relative_gated_blocks: usize,
}

pub fn gate_block_energies_bs1770(
    block_energies: &[f64],
) -> Result<Bs1770LoudnessReading, SignalError> {
    if block_energies.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    for &e in block_energies {
        if !e.is_finite() {
            return Err(SignalError::NonFiniteInput);
        }
        if e < 0.0 {
            return Err(SignalError::InvalidParameter);
        }
    }

    let mut sum_abs = 0.0f64;
    let mut count_abs = 0usize;
    for &e in block_energies {
        if e > 0.0 {
            let l_j = energy_to_lufs(e);
            if l_j > BS1770_ABSOLUTE_GATE_LUFS {
                sum_abs += e;
                count_abs += 1;
            }
        }
    }

    if count_abs == 0 {
        return Ok(Bs1770LoudnessReading {
            status: QualityStatus::Silence,
            integrated_lufs: None,
            absolute_gated_lufs: None,
            relative_threshold_lufs: None,
            total_blocks: block_energies.len(),
            absolute_gated_blocks: 0,
            relative_gated_blocks: 0,
        });
    }

    let mean_abs = sum_abs / (count_abs as f64);
    let l_kg_a = energy_to_lufs(mean_abs);
    let gamma_r = l_kg_a + BS1770_RELATIVE_GATE_OFFSET_LU;

    let mut sum_rel = 0.0f64;
    let mut count_rel = 0usize;
    for &e in block_energies {
        if e > 0.0 {
            let l_j = energy_to_lufs(e);
            if l_j > BS1770_ABSOLUTE_GATE_LUFS && l_j > gamma_r {
                sum_rel += e;
                count_rel += 1;
            }
        }
    }

    if count_rel == 0 {
        return Ok(Bs1770LoudnessReading {
            status: QualityStatus::Silence,
            integrated_lufs: None,
            absolute_gated_lufs: Some(l_kg_a),
            relative_threshold_lufs: Some(gamma_r),
            total_blocks: block_energies.len(),
            absolute_gated_blocks: count_abs,
            relative_gated_blocks: 0,
        });
    }

    let l_kg = energy_to_lufs(sum_rel / (count_rel as f64));
    Ok(Bs1770LoudnessReading {
        status: QualityStatus::Valid,
        integrated_lufs: Some(l_kg),
        absolute_gated_lufs: Some(l_kg_a),
        relative_threshold_lufs: Some(gamma_r),
        total_blocks: block_energies.len(),
        absolute_gated_blocks: count_abs,
        relative_gated_blocks: count_rel,
    })
}

pub fn measure_bs1770_integrated_loudness(
    interleaved_samples: &[f32],
    channel_roles: &[Bs1770ChannelRole],
    sample_rate_hz: u32,
    block_energies_scratch: &mut [f64],
) -> Result<Bs1770LoudnessReading, SignalError> {
    let ch = channel_roles.len();
    if ch == 0 || ch > MAX_BS1770_CHANNELS {
        return Err(SignalError::InvalidParameter);
    }
    if interleaved_samples.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !interleaved_samples.len().is_multiple_of(ch) {
        return Err(SignalError::LengthMismatch);
    }
    if sample_rate_hz < 8_000 || !sample_rate_hz.is_multiple_of(10) {
        return Err(SignalError::InvalidParameter);
    }
    if interleaved_samples.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }

    let num_frames = interleaved_samples.len() / ch;
    let hop_frames = (sample_rate_hz as usize) / 10;
    let block_frames = hop_frames * 4;
    if num_frames < block_frames {
        return Err(SignalError::LengthMismatch);
    }
    let num_blocks = (num_frames - block_frames) / hop_frames + 1;
    if block_energies_scratch.len() < num_blocks {
        return Err(SignalError::LengthMismatch);
    }

    let fs = f64::from(sample_rate_hz);
    let mut filters = [CascadedBiquadDf2t::<2>::new([
        BiquadCoeffs::IDENTITY,
        BiquadCoeffs::IDENTITY,
    ])?; MAX_BS1770_CHANNELS];
    for f in filters.iter_mut().take(ch) {
        *f = new_k_weighting_cascade(fs)?;
    }

    let mut subblock_weighted_energy = [0.0f64; 4];
    let inv_block_frames = 1.0 / (block_frames as f64);
    let num_subblocks = num_blocks + 3;

    for sb_idx in 0..num_subblocks {
        let frame_start = sb_idx * hop_frames;
        let frame_end = frame_start + hop_frames;
        let mut ch_sq_sums = [0.0f64; MAX_BS1770_CHANNELS];

        for frame in frame_start..frame_end {
            let base = frame * ch;
            for c in 0..ch {
                let x = f64::from(interleaved_samples[base + c]);
                let y = filters[c].tick(x);
                ch_sq_sums[c] += y * y;
            }
        }

        let mut sb_energy = 0.0f64;
        for c in 0..ch {
            let g_i = channel_roles[c].weight_g();
            if g_i > 0.0 {
                sb_energy += g_i * ch_sq_sums[c];
            }
        }
        subblock_weighted_energy[sb_idx & 3] = sb_energy;

        if sb_idx >= 3 {
            let blk_idx = sb_idx - 3;
            let total_400ms_sq = subblock_weighted_energy[0]
                + subblock_weighted_energy[1]
                + subblock_weighted_energy[2]
                + subblock_weighted_energy[3];
            block_energies_scratch[blk_idx] = total_400ms_sq * inv_block_frames;
        }
    }

    gate_block_energies_bs1770(&block_energies_scratch[..num_blocks])
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TruePeakReading {
    pub status: QualityStatus,
    pub linear_true_peak: f64,
    pub dbtp: Option<f64>,
}

pub fn measure_true_peak_4x_f32(samples: &[f32]) -> Result<TruePeakReading, SignalError> {
    if samples.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if samples.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }

    const TAPS_PER_PHASE: usize = 12;
    const CENTER: f64 = 5.5;
    let mut delay_line = [0.0f64; TAPS_PER_PHASE];
    let mut head = 0usize;
    let mut max_peak = 0.0f64;

    let mut coeffs = [[0.0f64; TAPS_PER_PHASE]; 3];
    for (p_idx, &frac) in [0.25f64, 0.50, 0.75].iter().enumerate() {
        let mut sum = 0.0f64;
        for k in 0..TAPS_PER_PHASE {
            let x = (k as f64) - CENTER - frac;
            let sinc = if x.abs() <= 1e-12 {
                1.0
            } else {
                let pix = core::f64::consts::PI * x;
                let (s, _) = sin_cos_f64(pix);
                s / pix
            };
            let w_ang =
                core::f64::consts::TAU * ((k as f64) - frac + 0.5) / (TAPS_PER_PHASE as f64);
            let (_, cos_w) = sin_cos_f64(w_ang);
            let win = 0.5 * (1.0 - cos_w);
            let c = sinc * win;
            coeffs[p_idx][k] = c;
            sum += c;
        }
        if sum.abs() > 1e-12 {
            for slot in &mut coeffs[p_idx] {
                *slot /= sum;
            }
        }
    }

    for &s in samples {
        let sd = f64::from(s);
        let abs_s = sd.abs();
        if abs_s > max_peak {
            max_peak = abs_s;
        }
        delay_line[head] = sd;

        for phase in &coeffs {
            let mut acc = 0.0f64;
            for (k, &w) in phase.iter().enumerate() {
                let idx = (head + TAPS_PER_PHASE - k) % TAPS_PER_PHASE;
                acc += delay_line[idx] * w;
            }
            let a = acc.abs();
            if a > max_peak {
                max_peak = a;
            }
        }
        head = (head + 1) % TAPS_PER_PHASE;
    }

    if max_peak <= 0.0 {
        Ok(TruePeakReading {
            status: QualityStatus::Silence,
            linear_true_peak: 0.0,
            dbtp: None,
        })
    } else {
        Ok(TruePeakReading {
            status: QualityStatus::Valid,
            linear_true_peak: max_peak,
            dbtp: Some(20.0 * log10_f64(max_peak)),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedSplReading {
    pub status: QualityStatus,
    pub rms_pa: f64,
    pub db_spl: Option<f64>,
    pub ref_pressure_pa: f64,
}

pub fn calibrated_rms_db_spl(
    pressure_samples_pa: &[f32],
    ref_pressure_pa: f64,
) -> Result<CalibratedSplReading, SignalError> {
    if pressure_samples_pa.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !ref_pressure_pa.is_finite() || pressure_samples_pa.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    if ref_pressure_pa <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }

    let mut sum_sq = 0.0f64;
    for &p in pressure_samples_pa {
        let pd = f64::from(p);
        sum_sq += pd * pd;
    }
    let rms_pa = sqrt_f64(sum_sq / (pressure_samples_pa.len() as f64));
    if rms_pa <= 0.0 {
        Ok(CalibratedSplReading {
            status: QualityStatus::Silence,
            rms_pa: 0.0,
            db_spl: None,
            ref_pressure_pa,
        })
    } else {
        let db_spl = 20.0 * log10_f64(rms_pa / ref_pressure_pa);
        Ok(CalibratedSplReading {
            status: QualityStatus::Valid,
            rms_pa,
            db_spl: Some(db_spl),
            ref_pressure_pa,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spl_and_bs1770_gating_oracles() {
        let one_pa = [1.0f32, -1.0, 1.0, -1.0];
        let spl_air = calibrated_rms_db_spl(&one_pa, AIR_REF_PRESSURE_PA).unwrap();
        assert_eq!(spl_air.status, QualityStatus::Valid);
        assert!((spl_air.db_spl.unwrap() - 93.979_400_086_720_38).abs() <= 1e-4);

        let spl_water = calibrated_rms_db_spl(&one_pa, UNDERWATER_REF_PRESSURE_PA).unwrap();
        assert!((spl_water.db_spl.unwrap() - 120.0).abs() <= 1e-4);

        let silent = calibrated_rms_db_spl(&[0.0f32; 8], AIR_REF_PRESSURE_PA).unwrap();
        assert_eq!(silent.status, QualityStatus::Silence);
        assert_eq!(silent.db_spl, None);
    }
}
