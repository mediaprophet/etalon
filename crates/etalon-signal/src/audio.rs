//! Caller-selected PCM audio features, zero-heap STFT / COLA-verified WOLA ISTFT
//! transforms, and spherical-head Woodworth/Rayleigh ITD + pinna-notch ILD
//! binaural HRTF primitives.
//!
//! Digital values are not sound pressure levels without a separately qualified
//! sensor and calibration.
use crate::{
    dft::fft_radix2_in_place,
    fft::MAX_FFT_SAMPLES,
    filter::{cos_f64, floor_f64, round_symmetric_f64, sin_cos_f64, sin_f64, sqrt_f64},
    wave::Pcm16Wave,
    SignalError,
};

/// Standard anthropometric spherical-head radius in meters (8.75 cm).
pub const DEFAULT_HEAD_RADIUS_M: f32 = 0.0875;
/// Standard speed of sound in dry air at 20 °C in m/s.
pub const DEFAULT_SPEED_OF_SOUND_M_S: f32 = 343.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelPower {
    pub channel: u16,
    pub sample_rate_hz: u32,
    pub sample_count: usize,
    /// Mean square of signed PCM codes normalized by 32768.
    pub mean_square: f64,
}

/// Compute normalized mean-square sample power for one channel in a PCM16
/// recording. Channel numbering is zero-based. This is an electrical/digital
/// feature; it does not infer acoustic pressure, species or speech content.
pub fn pcm16_channel_power(wave: Pcm16Wave<'_>, channel: u16) -> Result<ChannelPower, SignalError> {
    if channel >= wave.channels() {
        return Err(SignalError::InvalidParameter);
    }
    if wave.frames() == 0 {
        return Err(SignalError::EmptyInput);
    }
    let channels = usize::from(wave.channels());
    let stride = channels * 2; // PCM parser checked u16 block alignment.
    let offset = usize::from(channel) * 2;
    let mut sum = 0.0f64;
    for frame in wave.samples().chunks_exact(stride) {
        let sample = i16::from_le_bytes([frame[offset], frame[offset + 1]]);
        let normalized = f64::from(sample) / 32768.0;
        sum += normalized * normalized;
    }
    let mean_square = sum / wave.frames() as f64;
    if !mean_square.is_finite() {
        return Err(SignalError::NumericRange);
    }
    Ok(ChannelPower {
        channel,
        sample_rate_hz: wave.sample_rate_hz(),
        sample_count: wave.frames(),
        mean_square,
    })
}

/// Evaluate a single symmetric Hann window sample:
/// $$w[i] = \frac{1}{2}\left(1 - \cos\!\left(\frac{2\pi i}{N - 1}\right)\right)$$
#[inline]
pub fn hann_window_sample(index: usize, window_len: usize) -> f32 {
    if window_len <= 1 {
        return 1.0;
    }
    if index >= window_len {
        return 0.0;
    }
    let angle = core::f64::consts::TAU * (index as f64) / ((window_len - 1) as f64);
    (0.5 * (1.0 - cos_f64(angle))) as f32
}

/// Fill `out` with a symmetric Hann window of length `out.len()`.
pub fn fill_hann_window_into(out: &mut [f32]) -> Result<usize, SignalError> {
    if out.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let n = out.len();
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = hann_window_sample(i, n);
    }
    Ok(n)
}

/// Verify whether `window` satisfies the Constant Overlap-Add (COLA) condition
/// at stride `hop` within `tolerance`.
///
/// Checks that across all hop-phase residues $m \in [0, \text{hop})$, the
/// overlapped window sum $S_m = \sum_{k} w[m + k \cdot \text{hop}]$ is strictly
/// positive and remains within `tolerance` of the mean envelope.
pub fn verify_cola_window(window: &[f32], hop: usize, tolerance: f32) -> Result<bool, SignalError> {
    if window.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if hop == 0 || hop > window.len() || !tolerance.is_finite() || tolerance < 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    if window.iter().any(|w| !w.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }

    let n = window.len();
    let mut total_sum = 0.0f64;
    for &w in window {
        total_sum += f64::from(w);
    }
    let mean_envelope = total_sum / (hop as f64);
    if mean_envelope <= 1e-6 {
        return Ok(false);
    }

    let tol = f64::from(tolerance);
    for phase in 0..hop {
        let mut sum_phase = 0.0f64;
        let mut idx = phase;
        while idx < n {
            sum_phase += f64::from(window[idx]);
            idx += hop;
        }
        if sum_phase <= 1e-6 || (sum_phase - mean_envelope).abs() > tol {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Window a single real time-domain `frame` of power-of-two length $N \in [2, 4096]$
/// with a Hann window and compute its $N$-bin interleaved complex forward FFT
/// (`[re_0, im_0, ..., re_{N-1}, im_{N-1}]`) into `out_spectrum[..2*N]`.
///
/// Returns the number of complex frequency bins written ($N$).
pub fn stft_frame_into(frame: &[f32], out_spectrum: &mut [f32]) -> Result<usize, SignalError> {
    if frame.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let n = frame.len();
    if !n.is_power_of_two() || !(2..=MAX_FFT_SAMPLES).contains(&n) {
        return Err(SignalError::InvalidParameter);
    }
    if out_spectrum.len() < 2 * n {
        return Err(SignalError::LengthMismatch);
    }
    if frame.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }

    for (i, &sample) in frame.iter().enumerate() {
        out_spectrum[2 * i] = sample * hann_window_sample(i, n);
        out_spectrum[2 * i + 1] = 0.0;
    }
    fft_radix2_in_place(&mut out_spectrum[..2 * n], false)
}

/// Compute the one-sided magnitude spectrum ($N/2 + 1$ bins, DC through Nyquist)
/// from an interleaved complex spectrum of length $2N$.
pub fn stft_one_sided_magnitudes_into(
    spectrum_interleaved: &[f32],
    out_magnitudes: &mut [f32],
) -> Result<usize, SignalError> {
    if spectrum_interleaved.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !spectrum_interleaved.len().is_multiple_of(2) {
        return Err(SignalError::LengthMismatch);
    }
    let n = spectrum_interleaved.len() / 2;
    let bins = n / 2 + 1;
    if out_magnitudes.len() < bins {
        return Err(SignalError::LengthMismatch);
    }
    if spectrum_interleaved.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }

    for k in 0..bins {
        let re = f64::from(spectrum_interleaved[2 * k]);
        let im = f64::from(spectrum_interleaved[2 * k + 1]);
        let mag = sqrt_f64(re * re + im * im) as f32;
        if !mag.is_finite() {
            return Err(SignalError::NumericRange);
        }
        out_magnitudes[k] = mag;
    }
    Ok(bins)
}

/// Reconstruct a time-domain signal from concatenated interleaved-complex STFT frames
/// (`num_frames * 2 * frame_size`) using in-place inverse radix-2 FFT and
/// weighted overlap-add (WOLA) normalized by the squared window envelope.
///
/// - `spectra`: `num_frames * 2 * frame_size` interleaved complex bins (clobbered in place).
/// - `frame_size`: power of two in `[2, 4096]`.
/// - `hop`: sample advance between frames (`> 0`).
/// - `window`: length-`frame_size` synthesis/analysis window.
/// - `norm_scratch`: caller-supplied scratch of length `>= out_len`.
/// - `output`: caller-supplied output slice of length `>= out_len`, where
///   `out_len = (num_frames - 1) * hop + frame_size`.
///
/// Returns the number of reconstructed samples written (`out_len`, or `0` if `spectra` is empty).
pub fn istft_overlap_add_into(
    spectra: &mut [f32],
    frame_size: usize,
    hop: usize,
    window: &[f32],
    norm_scratch: &mut [f32],
    output: &mut [f32],
) -> Result<usize, SignalError> {
    if frame_size == 0 || !frame_size.is_power_of_two() || frame_size > MAX_FFT_SAMPLES || hop == 0
    {
        return Err(SignalError::InvalidParameter);
    }
    if window.len() != frame_size {
        return Err(SignalError::LengthMismatch);
    }
    let two_n = 2 * frame_size;
    if !spectra.len().is_multiple_of(two_n) {
        return Err(SignalError::LengthMismatch);
    }
    if window.iter().any(|w| !w.is_finite()) || spectra.iter().any(|s| !s.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let num_frames = spectra.len() / two_n;
    if num_frames == 0 {
        return Ok(0);
    }

    let out_len = (num_frames - 1)
        .checked_mul(hop)
        .and_then(|v| v.checked_add(frame_size))
        .ok_or(SignalError::NumericRange)?;
    if norm_scratch.len() < out_len || output.len() < out_len {
        return Err(SignalError::LengthMismatch);
    }

    output[..out_len].fill(0.0);
    norm_scratch[..out_len].fill(0.0);

    for frame_idx in 0..num_frames {
        let frame_spec = &mut spectra[frame_idx * two_n..(frame_idx + 1) * two_n];
        fft_radix2_in_place(frame_spec, true)?;
        let base = frame_idx * hop;
        for j in 0..frame_size {
            let w = window[j];
            let x_re = frame_spec[2 * j];
            output[base + j] += w * x_re;
            norm_scratch[base + j] += w * w;
        }
    }

    const NOLA_EPS: f32 = 1e-8;
    for k in 0..out_len {
        let denom = norm_scratch[k];
        if denom.abs() > NOLA_EPS {
            output[k] /= denom;
        }
        if !output[k].is_finite() {
            return Err(SignalError::NumericRange);
        }
    }
    Ok(out_len)
}

/// Woodworth / Rayleigh spherical-head Interaural Time Difference (ITD) in seconds:
/// $$\text{ITD}(\theta) = \operatorname{sign}(\theta) \cdot \frac{a}{c}\left(\phi + \sin\phi\right)$$
/// where $\phi = \arcsin(\sin|\theta|) \in [0, \pi/2]$ is the lateral angle from the median plane,
/// $a$ is `head_radius_m` ($> 0$), and $c$ is `speed_of_sound_m_s` ($> 0$).
///
/// Negative ITD indicates source is to the left (left ear leads); positive ITD indicates
/// source is to the right (right ear leads).
pub fn woodworth_itd_seconds(
    azimuth_rad: f32,
    head_radius_m: f32,
    speed_of_sound_m_s: f32,
) -> Result<f32, SignalError> {
    if !azimuth_rad.is_finite() || !head_radius_m.is_finite() || !speed_of_sound_m_s.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if head_radius_m <= 0.0 || speed_of_sound_m_s <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }

    let az = f64::from(azimuth_rad);
    let k = round_symmetric_f64(az / core::f64::consts::TAU);
    let theta = az - k * core::f64::consts::TAU;
    let abs_theta = theta.abs();
    let phi = if abs_theta <= core::f64::consts::FRAC_PI_2 {
        abs_theta
    } else {
        (core::f64::consts::PI - abs_theta).max(0.0)
    };
    let sign = if theta > 0.0 {
        1.0
    } else if theta < 0.0 {
        -1.0
    } else {
        0.0
    };

    let itd =
        sign * (f64::from(head_radius_m) / f64::from(speed_of_sound_m_s)) * (phi + sin_f64(phi));
    let out = itd as f32;
    if !out.is_finite() {
        return Err(SignalError::NumericRange);
    }
    Ok(out)
}

/// Spherical-head Rayleigh / Brown-Duda diffraction Interaural Level Difference (ILD) in dB
/// as a function of horizontal `azimuth_rad` and `frequency_hz`.
///
/// Positive dB favors the right ear ($\theta > 0$); negative dB favors the left ear ($\theta < 0$).
pub fn spherical_ild_db(azimuth_rad: f32, frequency_hz: f32) -> Result<f32, SignalError> {
    if !azimuth_rad.is_finite() || !frequency_hz.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if frequency_hz <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }

    let ka = core::f64::consts::TAU * f64::from(frequency_hz) * f64::from(DEFAULT_HEAD_RADIUS_M)
        / f64::from(DEFAULT_SPEED_OF_SOUND_M_S);
    let max_ild_db = 1.5 + 18.5 * (ka / (1.0 + ka));
    let ild_db = sin_f64(f64::from(azimuth_rad)) * max_ild_db;
    let out = ild_db as f32;
    if !out.is_finite() {
        return Err(SignalError::NumericRange);
    }
    Ok(out)
}

/// Anthropometric pinna concha primary spectral notch center frequency in Hz
/// for vertical `elevation_rad` $\in [-\pi/2, \pi/2]$ (Hebrank & Wright / Lopez-Poveda model).
pub fn pinna_elevation_notch_hz(elevation_rad: f32) -> Result<f32, SignalError> {
    if !elevation_rad.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    let half_pi = core::f32::consts::FRAC_PI_2 + 1e-4;
    if elevation_rad < -half_pi || elevation_rad > half_pi {
        return Err(SignalError::InvalidParameter);
    }
    let clamped =
        f64::from(elevation_rad).clamp(-core::f64::consts::FRAC_PI_2, core::f64::consts::FRAC_PI_2);
    let notch_hz = 8000.0 + 3500.0 * sin_f64(clamped);
    Ok(notch_hz as f32)
}

/// Rotate a 3D world-space source position into the listener's head frame
/// (listener at origin, yaw rotation around +Y).
#[inline]
pub fn head_relative_position(source: [f32; 3], listener_yaw_rad: f32) -> [f32; 3] {
    let (s, c) = sin_cos_f64(f64::from(listener_yaw_rad));
    let x = f64::from(source[0]);
    let z = f64::from(source[2]);
    [(x * c - z * s) as f32, source[1], (x * s + z * c) as f32]
}

/// Caller-buffered direct linear FIR convolution:
/// $$y[n] = \sum_{m=0}^{M-1} x[n - m]\, h[m]$$
///
/// Writes `signal.len() + kernel.len() - 1` samples into `output` (or `0` if either input is empty).
pub fn convolve_fir_into(
    signal: &[f32],
    kernel: &[f32],
    output: &mut [f32],
) -> Result<usize, SignalError> {
    if signal.is_empty() || kernel.is_empty() {
        return Ok(0);
    }
    if signal.iter().any(|x| !x.is_finite()) || kernel.iter().any(|h| !h.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let out_len = signal
        .len()
        .checked_add(kernel.len())
        .and_then(|v| v.checked_sub(1))
        .ok_or(SignalError::NumericRange)?;
    if output.len() < out_len {
        return Err(SignalError::LengthMismatch);
    }

    output[..out_len].fill(0.0);
    for (j, &s) in signal.iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        for (m, &hm) in kernel.iter().enumerate() {
            output[j + m] += s * hm;
        }
    }
    if output[..out_len].iter().any(|y| !y.is_finite()) {
        return Err(SignalError::NumericRange);
    }
    Ok(out_len)
}

#[inline]
fn place_fractional_impulse(ir: &mut [f32], delay_samples: f32, gain: f32) {
    if ir.is_empty() {
        return;
    }
    let d = delay_samples.max(0.0);
    let i0 = floor_f64(f64::from(d)) as usize;
    let frac = d - (i0 as f32);
    if i0 < ir.len() {
        ir[i0] += gain * (1.0 - frac);
    }
    if i0 + 1 < ir.len() {
        ir[i0 + 1] += gain * frac;
    }
}

#[inline]
fn one_pole_lowpass_in_place(buf: &mut [f32], alpha: f32) {
    let a = alpha.clamp(0.0, 1.0);
    let mut y = 0.0_f32;
    for x in buf.iter_mut() {
        y += a * (*x - y);
        *x = y;
    }
}

/// Synthesize a stereo pair of head-related impulse responses (`left_ir`, `right_ir`)
/// into caller-supplied slices using linear-interpolated fractional ITD delay and
/// contralateral one-pole head-shadow low-pass filtering.
pub fn synthesize_hrir_into(
    gain_l: f32,
    gain_r: f32,
    itd_seconds: f32,
    sample_rate_hz: f32,
    shadow_alpha: f32,
    left_ir: &mut [f32],
    right_ir: &mut [f32],
) -> Result<usize, SignalError> {
    if left_ir.is_empty() || right_ir.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if left_ir.len() != right_ir.len() {
        return Err(SignalError::LengthMismatch);
    }
    if !gain_l.is_finite()
        || !gain_r.is_finite()
        || !itd_seconds.is_finite()
        || !sample_rate_hz.is_finite()
        || !shadow_alpha.is_finite()
    {
        return Err(SignalError::NonFiniteInput);
    }
    if sample_rate_hz <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }

    let taps = left_ir.len();
    left_ir.fill(0.0);
    right_ir.fill(0.0);

    let delay = (itd_seconds.abs() * sample_rate_hz.max(1.0)).max(0.0);
    if itd_seconds <= 0.0 {
        place_fractional_impulse(left_ir, 0.0, gain_l);
        place_fractional_impulse(right_ir, delay, gain_r);
        one_pole_lowpass_in_place(right_ir, shadow_alpha);
    } else {
        place_fractional_impulse(right_ir, 0.0, gain_r);
        place_fractional_impulse(left_ir, delay, gain_l);
        one_pole_lowpass_in_place(left_ir, shadow_alpha);
    }
    Ok(taps)
}

#[cfg(test)]
#[path = "audio_tests.rs"]
mod tests;

#[cfg(test)]
mod stft_hrtf_tests {
    use super::*;

    #[test]
    fn cola_window_verification_and_stft_istft_reconstruction_identity() {
        const N: usize = 64;
        const HOP: usize = N / 2;
        const NUM_FRAMES: usize = 7;
        const SIG_LEN: usize = (NUM_FRAMES - 1) * HOP + N;

        let mut window = [0.0f32; N];
        fill_hann_window_into(&mut window).unwrap();

        // Hann at 50% overlap (hop = N/2) and 75% overlap (hop = N/4) satisfies COLA;
        // hop = N (0% overlap) fails COLA at endpoints where w[0] = 0.
        let mut cola_window = [0.0f32; N];
        crate::window::hann_window(&mut cola_window).unwrap();
        assert!(verify_cola_window(&cola_window, HOP, 1e-5).unwrap());
        assert!(verify_cola_window(&cola_window, N / 4, 1e-5).unwrap());
        assert!(!verify_cola_window(&cola_window, N, 1e-5).unwrap());
        assert!(verify_cola_window(&window, HOP, 2e-2).unwrap());
        assert!(verify_cola_window(&window, N / 4, 1e-2).unwrap());
        assert!(!verify_cola_window(&window, N, 1e-2).unwrap());

        let mut signal = [0.0f32; SIG_LEN];
        for (i, slot) in signal.iter_mut().enumerate() {
            let t = i as f64 / 16_000.0;
            *slot = (0.6 * sin_f64(core::f64::consts::TAU * 440.0 * t)
                + 0.3 * cos_f64(core::f64::consts::TAU * 1250.0 * t)) as f32;
        }

        let mut spectra = [0.0f32; NUM_FRAMES * 2 * N];
        for f in 0..NUM_FRAMES {
            let start = f * HOP;
            let frame = &signal[start..start + N];
            let out_spec = &mut spectra[f * 2 * N..(f + 1) * 2 * N];
            assert_eq!(stft_frame_into(frame, out_spec), Ok(N));
        }

        let mut norm_scratch = [0.0f32; SIG_LEN];
        let mut reconstructed = [0.0f32; SIG_LEN];
        let written = istft_overlap_add_into(
            &mut spectra,
            N,
            HOP,
            &window,
            &mut norm_scratch,
            &mut reconstructed,
        )
        .unwrap();
        assert_eq!(written, SIG_LEN);

        // Steady-state interior [N, SIG_LEN - N) must reconstruct accurately.
        for i in N..(SIG_LEN - N) {
            let err = (reconstructed[i] - signal[i]).abs();
            assert!(
                err < 1e-4,
                "COLA WOLA interior mismatch at sample {i}: got {}, expected {} (err={err})",
                reconstructed[i],
                signal[i]
            );
        }
    }

    #[test]
    fn woodworth_itd_odd_symmetry_and_bounds() {
        let a = DEFAULT_HEAD_RADIUS_M;
        let c = DEFAULT_SPEED_OF_SOUND_M_S;

        assert_eq!(woodworth_itd_seconds(0.0, a, c).unwrap(), 0.0);
        assert!(
            woodworth_itd_seconds(core::f32::consts::PI, a, c)
                .unwrap()
                .abs()
                < 1e-7
        );

        for deg in [10.0_f32, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0] {
            let rad = deg * core::f32::consts::PI / 180.0;
            let right = woodworth_itd_seconds(rad, a, c).unwrap();
            let left = woodworth_itd_seconds(-rad, a, c).unwrap();
            assert!(right > 0.0);
            assert!(left < 0.0);
            assert!(
                (right + left).abs() < 1e-9,
                "Woodworth ITD must be odd-symmetric: right={right}, left={left}"
            );
        }

        let max_itd = woodworth_itd_seconds(core::f32::consts::FRAC_PI_2, a, c).unwrap();
        assert!(max_itd > 0.00060 && max_itd < 0.00070);
    }

    #[test]
    fn spherical_ild_and_pinna_elevation_notch_properties() {
        let ild_low = spherical_ild_db(core::f32::consts::FRAC_PI_2, 250.0).unwrap();
        let ild_high = spherical_ild_db(core::f32::consts::FRAC_PI_2, 4000.0).unwrap();
        let ild_left = spherical_ild_db(-core::f32::consts::FRAC_PI_2, 4000.0).unwrap();
        assert!(ild_high > ild_low);
        assert!((ild_high + ild_left).abs() < 1e-6);

        let n_down = pinna_elevation_notch_hz(-0.7).unwrap();
        let n_zero = pinna_elevation_notch_hz(0.0).unwrap();
        let n_up = pinna_elevation_notch_hz(0.7).unwrap();
        assert!(n_down < n_zero && n_zero < n_up);
        assert!((n_zero - 8000.0).abs() < 1e-3);
    }
}
