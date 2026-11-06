//! Zero-heap YIN fundamental-frequency estimation (de Cheveigné & Kawahara, 2002)
//! and equal-tempered MIDI / cents pitch conversions.

use crate::filter::{exp_f64, ln_f64, round_symmetric_f64};

/// Compute the YIN difference function $d(\tau) = \sum_{j=0}^{W-1} (x[j] - x[j+\tau])^2$
/// into `diff[0..=max_tau]` where $W = N - \text{max\_tau}$.
/// Returns the number of lag bins written.
pub fn yin_difference_into(frame: &[f32], max_tau: usize, diff: &mut [f32]) -> usize {
    if frame.len() <= max_tau || diff.is_empty() {
        return 0;
    }
    let hi = max_tau.min(diff.len() - 1);
    let w = frame.len() - hi;
    diff[0] = 0.0;
    for tau in 1..=hi {
        let mut acc = 0.0f64;
        for j in 0..w {
            let d = f64::from(frame[j] - frame[j + tau]);
            acc += d * d;
        }
        diff[tau] = acc as f32;
    }
    hi + 1
}

/// Convert raw difference function values `diff` in place into YIN's
/// Cumulative Mean Normalized Difference Function (CMNDF):
/// $d'(0) = 1$, $d'(\tau) = d(\tau) \cdot \tau / \sum_{k=1}^{\tau} d(k)$.
pub fn yin_cmndf_in_place(diff: &mut [f32]) {
    if diff.is_empty() {
        return;
    }
    diff[0] = 1.0;
    let mut running = 0.0f64;
    for (tau, d) in diff.iter_mut().enumerate().skip(1) {
        let raw = f64::from(*d);
        running += raw;
        *d = if running <= 0.0 {
            1.0
        } else {
            (raw * (tau as f64) / running) as f32
        };
    }
}

/// Absolute-threshold lag selection over `cmndf[min_lag..=max_lag]`.
pub fn yin_absolute_threshold(
    cmndf: &[f32],
    min_lag: usize,
    max_lag: usize,
    threshold: f32,
) -> usize {
    if cmndf.is_empty() {
        return 0;
    }
    let hi = max_lag.min(cmndf.len() - 1);
    let lo = min_lag.max(1).min(hi);
    let mut best_tau = lo;
    let mut best_val = cmndf[lo];
    let mut tau = lo;
    while tau <= hi {
        let v = cmndf[tau];
        if v < best_val {
            best_val = v;
            best_tau = tau;
        }
        if v < threshold {
            let mut t = tau;
            while t < hi && cmndf[t + 1] < cmndf[t] {
                t += 1;
            }
            return t;
        }
        tau += 1;
    }
    best_tau
}

/// Parabolic interpolation of the CMNDF minimum around integer lag `tau`,
/// returning `(refined_tau, interpolated_min_val)`.
#[inline]
pub fn yin_parabolic_min(
    cmndf: &[f32],
    tau: usize,
    min_lag: usize,
    max_lag: usize,
) -> (f32, f32) {
    if cmndf.is_empty() {
        return (0.0, 1.0);
    }
    let hi = max_lag.min(cmndf.len() - 1);
    if tau <= min_lag.max(1) || tau >= hi {
        return (tau as f32, cmndf[tau.min(cmndf.len() - 1)]);
    }
    let s0 = cmndf[tau - 1];
    let s1 = cmndf[tau];
    let s2 = cmndf[tau + 1];
    let denom = s0 - 2.0 * s1 + s2;
    if denom.abs() <= f32::EPSILON || !denom.is_finite() {
        return (tau as f32, s1);
    }
    let offset = (0.5 * (s0 - s2) / denom).clamp(-1.0, 1.0);
    let better_tau = (tau as f32) + offset;
    let min_val = s1 - 0.25 * (s0 - s2) * offset;
    (better_tau, min_val)
}

/// Parabolic interpolation of the CMNDF minimum around integer lag `tau`,
/// returning the sub-sample refined lag.
#[inline]
pub fn yin_parabolic_interp(cmndf: &[f32], tau: usize) -> f32 {
    if cmndf.len() < 3 {
        return tau as f32;
    }
    yin_parabolic_min(cmndf, tau, 1, cmndf.len() - 1).0
}

/// Map YIN aperiodicity to a voicing confidence in `[0.0, 1.0]`.
#[inline]
pub fn yin_pitch_confidence(aperiodicity: f32) -> f32 {
    if !aperiodicity.is_finite() {
        return 0.0;
    }
    let a = aperiodicity.clamp(0.0, 1.0);
    let base = 1.0 - a;
    (base * (2.0 - base)).clamp(0.0, 1.0)
}

/// Estimate the fundamental frequency $f_0$ in Hz of `frame` using YIN with caller-supplied
/// `diff_scratch`. Returns `None` if unvoiced or invalid parameters.
pub fn yin_estimate_f0_hz(
    frame: &[f32],
    sample_rate: f32,
    threshold: f32,
    diff_scratch: &mut [f32],
) -> Option<f32> {
    if !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !threshold.is_finite()
        || frame.len() < 8
        || diff_scratch.len() < 5
    {
        return None;
    }
    let max_lag = (frame.len() / 2).min(diff_scratch.len() - 1);
    let min_lag = 2usize;
    if max_lag < min_lag + 2 {
        return None;
    }
    yin_difference_into(frame, max_lag, &mut diff_scratch[..=max_lag]);
    yin_cmndf_in_place(&mut diff_scratch[..=max_lag]);
    let tau = yin_absolute_threshold(&diff_scratch[..=max_lag], min_lag, max_lag, threshold);
    let (better_tau, aperiodicity) =
        yin_parabolic_min(&diff_scratch[..=max_lag], tau, min_lag, max_lag);
    if !(better_tau.is_finite() && better_tau > 0.0) {
        return None;
    }
    if aperiodicity > 1.0 || (diff_scratch[tau] >= threshold && aperiodicity > 0.5) {
        return None;
    }
    Some(sample_rate / better_tau)
}

/// Convert frequency in Hz to fractional MIDI note number (concert A4 = 440 Hz = MIDI 69):
/// $\text{midi} = 69 + 12 \log_2(f / 440)$.
#[inline]
pub fn hz_to_midi(hz: f32) -> f32 {
    if !hz.is_finite() || hz <= 0.0 {
        return 0.0;
    }
    (69.0 + 12.0 * (ln_f64(f64::from(hz) / 440.0) * core::f64::consts::LOG2_E)) as f32
}

/// Convert fractional MIDI note number to frequency in Hz (concert A4 = 440 Hz = MIDI 69):
/// $f = 440 \cdot 2^{(\text{midi} - 69) / 12}$.
#[inline]
pub fn midi_to_hz(midi: f32) -> f32 {
    if !midi.is_finite() {
        return 0.0;
    }
    (440.0 * exp_f64(((f64::from(midi) - 69.0) / 12.0) * core::f64::consts::LN_2)) as f32
}

/// Signed pitch deviation in cents from `ref_hz` to `hz`:
/// $\text{cents} = 1200 \log_2(f / f_{\text{ref}})$.
#[inline]
pub fn cents_deviation(hz: f32, ref_hz: f32) -> f32 {
    if !hz.is_finite() || hz <= 0.0 || !ref_hz.is_finite() || ref_hz <= 0.0 {
        return 0.0;
    }
    (1200.0 * (ln_f64(f64::from(hz) / f64::from(ref_hz)) * core::f64::consts::LOG2_E)) as f32
}

/// Quantize `freq_hz` against reference tuning `ref_a4_hz` into `(midi_note_u8, cents_deviation)`.
#[inline]
pub fn hz_to_midi_quantized(freq_hz: f32, ref_a4_hz: f32) -> (u8, f32) {
    if !freq_hz.is_finite() || freq_hz <= 0.0 || !ref_a4_hz.is_finite() || ref_a4_hz <= 0.0 {
        return (0, 0.0);
    }
    let exact =
        (69.0 + 12.0 * (ln_f64(f64::from(freq_hz) / f64::from(ref_a4_hz)) * core::f64::consts::LOG2_E))
            as f32;
    let rounded = round_symmetric_f64(f64::from(exact)) as f32;
    let note = rounded.clamp(0.0, 127.0) as u8;
    let cents = (exact - (note as f32)) * 100.0;
    (note, cents)
}

/// Convert integer MIDI note `note` with reference tuning `ref_a4_hz` to frequency in Hz.
#[inline]
pub fn midi_note_to_hz(note: u8, ref_a4_hz: f32) -> f32 {
    if !ref_a4_hz.is_finite() || ref_a4_hz <= 0.0 {
        return 0.0;
    }
    (f64::from(ref_a4_hz)
        * exp_f64(((f64::from(note) - 69.0) / 12.0) * core::f64::consts::LN_2)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::sin_f64;

    #[test]
    fn yin_recovers_440_hz_and_midi_round_trips() {
        let sr = 44_100.0f32;
        let mut frame = [0.0f32; 1024];
        for (i, s) in frame.iter_mut().enumerate() {
            *s = sin_f64(core::f64::consts::TAU * 440.0 * (i as f64) / f64::from(sr)) as f32;
        }
        let mut scratch = [0.0f32; 513];
        let f0 = yin_estimate_f0_hz(&frame, sr, 0.1, &mut scratch).expect("voiced");
        assert!(cents_deviation(f0, 440.0).abs() < 10.0);
        assert!((hz_to_midi(440.0) - 69.0).abs() < 1e-4);
        assert!((midi_to_hz(69.0) - 440.0).abs() < 1e-3);
    }
}
