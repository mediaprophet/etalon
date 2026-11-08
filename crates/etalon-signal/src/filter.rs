//! Stateful one-pole and Direct-Form I / II Transposed (DF2T) biquad filters with
//! caller-owned output storage, RBJ Audio EQ Cookbook designers, and ITU-R BS.1770-4 K-weighting.

use crate::SignalError;

const PI_F64: f64 = core::f64::consts::PI;
const TAU_F64: f64 = core::f64::consts::TAU;
const LN2_HI: f64 = core::f64::consts::LN_2;
const LN2_LO: f64 = 2.319_046_813_846_299_6e-17;
const PIO2_HI: f64 = core::f64::consts::FRAC_PI_2;
const PIO2_LO: f64 = 6.123_233_995_736_766e-17;
const INV_PIO2: f64 = 2.0 / PI_F64;

#[inline]
pub(crate) fn floor_f64(x: f64) -> f64 {
    if !x.is_finite() || x.abs() >= 4_503_599_627_370_496.0 {
        return x;
    }
    let t = (x as i64) as f64;
    if x < t { t - 1.0 } else { t }
}

#[inline]
pub(crate) fn round_symmetric_f64(x: f64) -> f64 {
    if x >= 0.0 { floor_f64(x + 0.5) } else { -floor_f64(-x + 0.5) }
}

#[inline]
pub(crate) fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    let mut y = f64::from_bits((x.to_bits() >> 1) + (1023_u64 << 51));
    for _ in 0..6 {
        y = 0.5 * (y + x / y);
    }
    y
}

#[inline]
pub(crate) fn sin_cos_f64(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let k = round_symmetric_f64(x * INV_PIO2);
    let r = (x - k * PIO2_HI) - k * PIO2_LO;
    let r2 = r * r;
    let sin_r = r
        * (1.0
            + r2 * (-1.0 / 6.0
                + r2 * (1.0 / 120.0
                    + r2 * (-1.0 / 5040.0
                        + r2 * (1.0 / 362_880.0
                            + r2 * (-1.0 / 39_916_800.0
                                + r2 * (1.0 / 6_227_020_800.0
                                    + r2 * (-1.0 / 1_307_674_368_000.0
                                        + r2 * (1.0 / 355_687_428_096_000.0)))))))));
    let cos_r = 1.0
        + r2 * (-0.5
            + r2 * (1.0 / 24.0
                + r2 * (-1.0 / 720.0
                    + r2 * (1.0 / 40_320.0
                        + r2 * (-1.0 / 3_628_800.0
                            + r2 * (1.0 / 479_001_600.0
                                + r2 * (-1.0 / 87_178_291_200.0
                                    + r2 * (1.0 / 20_922_789_888_000.0))))))));
    match (k as i64) & 3 {
        0 => (sin_r, cos_r),
        1 => (cos_r, -sin_r),
        2 => (-sin_r, -cos_r),
        _ => (-cos_r, sin_r),
    }
}

#[inline]
pub(crate) fn sin_f64(x: f64) -> f64 { sin_cos_f64(x).0 }

#[inline]
pub(crate) fn cos_f64(x: f64) -> f64 { sin_cos_f64(x).1 }

#[inline]
pub(crate) fn tan_f64(x: f64) -> f64 {
    let (s, c) = sin_cos_f64(x);
    s / c
}

#[inline]
pub(crate) fn exp_f64(x: f64) -> f64 {
    if x.is_nan() { return f64::NAN; }
    if x >= 709.782_712_893_384 { return f64::INFINITY; }
    if x <= -708.396_418_532_264_1 { return 0.0; }
    let k_f = round_symmetric_f64(x * core::f64::consts::LOG2_E);
    let r = (x - k_f * LN2_HI) - k_f * LN2_LO;
    let exp_r = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0
                            + r * (1.0 / 720.0
                                + r * (1.0 / 5040.0
                                    + r * (1.0 / 40_320.0
                                        + r * (1.0 / 362_880.0
                                            + r * (1.0 / 3_628_800.0
                                                + r * (1.0 / 39_916_800.0
                                                    + r * (1.0 / 479_001_600.0))))))))))));
    exp_r * f64::from_bits(((1023 + (k_f as i64)) as u64) << 52)
}

#[inline]
pub(crate) fn ln_f64(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() { return f64::NAN; }
    if x == 0.0 { return f64::NEG_INFINITY; }
    if x.is_infinite() { return f64::INFINITY; }
    let bits = x.to_bits();
    let mut exp = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let ln_m = 2.0
        * u
        * (1.0
            + u2 * (1.0 / 3.0
                + u2 * (1.0 / 5.0
                    + u2 * (1.0 / 7.0
                        + u2 * (1.0 / 9.0 + u2 * (1.0 / 11.0 + u2 * (1.0 / 13.0 + u2 / 15.0)))))));
    (exp as f64) * core::f64::consts::LN_2 + ln_m
}

#[inline]
pub(crate) fn log10_f64(x: f64) -> f64 {
    ln_f64(x) * core::f64::consts::LOG10_E
}

#[inline]
pub(crate) fn pow10_f64(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { exp_f64(x * core::f64::consts::LN_10) }
}

#[inline]
pub(crate) fn pow_f64(base: f64, exp: f64) -> f64 {
    if exp == 0.0 || base == 1.0 { return 1.0; }
    if base == 0.0 { return if exp > 0.0 { 0.0 } else { f64::INFINITY }; }
    if base < 0.0 { return f64::NAN; }
    exp_f64(exp * ln_f64(base))
}

/// A one-pole low-pass filter. The caller provides the initial state explicitly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnePoleLowPass {
    alpha: f64,
    state: f64,
}

impl OnePoleLowPass {
    pub fn new(alpha: f32, initial_state: f32) -> Result<Self, SignalError> {
        if !alpha.is_finite() || !initial_state.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(SignalError::InvalidParameter);
        }
        Ok(Self { alpha: f64::from(alpha), state: f64::from(initial_state) })
    }

    pub fn state(&self) -> f64 { self.state }

    pub fn process_into(&mut self, samples: &[f32], out: &mut [f32]) -> Result<usize, SignalError> {
        if out.len() < samples.len() { return Err(SignalError::LengthMismatch); }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SignalError::NonFiniteInput);
        }
        let mut state = self.state;
        for (index, &sample) in samples.iter().enumerate() {
            state = self.alpha * f64::from(sample) + (1.0 - self.alpha) * state;
            let output = state as f32;
            if !state.is_finite() || !output.is_finite() {
                return Err(SignalError::NumericRange);
            }
            out[index] = output;
        }
        self.state = state;
        Ok(samples.len())
    }
}

/// Normalized biquad transfer-function coefficients ($a_0 = 1$).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Default for BiquadCoeffs {
    fn default() -> Self { Self::IDENTITY }
}

impl BiquadCoeffs {
    pub const IDENTITY: Self = Self { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };

    #[inline]
    pub const fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self { b0, b1, b2, a1, a2 }
    }

    #[inline]
    pub fn from_f32(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: f64::from(b0),
            b1: f64::from(b1),
            b2: f64::from(b2),
            a1: f64::from(a1),
            a2: f64::from(a2),
        }
    }

    /// Jury stability criterion: $|a_2| < 1$ and $|a_1| < 1 + a_2$.
    #[inline]
    pub fn is_stable(&self) -> bool {
        self.b0.is_finite()
            && self.b1.is_finite()
            && self.b2.is_finite()
            && self.a1.is_finite()
            && self.a2.is_finite()
            && self.a2.abs() < 1.0
            && self.a1.abs() < 1.0 + self.a2
    }

    /// Evaluate squared magnitude response $|H(e^{j\omega})|^2$ at digital angle $\omega \in [0, \pi]$.
    pub fn frequency_response_mag_sq(&self, omega_rad: f64) -> Result<f64, SignalError> {
        if !omega_rad.is_finite() { return Err(SignalError::NonFiniteInput); }
        if !self.is_stable() { return Err(SignalError::InvalidParameter); }
        let (sin_w, cos_w) = sin_cos_f64(omega_rad);
        let cos_2w = 2.0 * cos_w * cos_w - 1.0;
        let sin_2w = 2.0 * sin_w * cos_w;
        let num_re = self.b0 + self.b1 * cos_w + self.b2 * cos_2w;
        let num_im = -self.b1 * sin_w - self.b2 * sin_2w;
        let den_re = 1.0 + self.a1 * cos_w + self.a2 * cos_2w;
        let den_im = -self.a1 * sin_w - self.a2 * sin_2w;
        let num = num_re * num_re + num_im * num_im;
        let den = den_re * den_re + den_im * den_im;
        if den <= 0.0 || !num.is_finite() || !den.is_finite() {
            return Err(SignalError::NumericRange);
        }
        Ok(num / den)
    }
}

#[inline]
fn normalize_rbj(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Result<BiquadCoeffs, SignalError> {
    if !a0.is_finite() || a0.abs() <= f64::EPSILON {
        return Err(SignalError::NumericRange);
    }
    let inv = 1.0 / a0;
    let c = BiquadCoeffs::new(b0 * inv, b1 * inv, b2 * inv, a1 * inv, a2 * inv);
    if !c.is_stable() { return Err(SignalError::NumericRange); }
    Ok(c)
}

#[inline]
fn validate_rbj_inputs(sample_rate_hz: f64, freq_hz: f64, q: f64) -> Result<(f64, f64, f64), SignalError> {
    if !sample_rate_hz.is_finite() || !freq_hz.is_finite() || !q.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if sample_rate_hz <= 0.0 || freq_hz <= 0.0 || freq_hz >= 0.5 * sample_rate_hz || q <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    let (sin_w0, cos_w0) = sin_cos_f64(TAU_F64 * freq_hz / sample_rate_hz);
    Ok((sin_w0, cos_w0, sin_w0 / (2.0 * q)))
}

pub fn rbj_lowpass(sample_rate_hz: f64, cutoff_hz: f64, q: f64) -> Result<BiquadCoeffs, SignalError> {
    let (_s, cos_w0, alpha) = validate_rbj_inputs(sample_rate_hz, cutoff_hz, q)?;
    let b1 = 1.0 - cos_w0;
    let b0 = 0.5 * b1;
    normalize_rbj(b0, b1, b0, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

pub fn rbj_highpass(sample_rate_hz: f64, cutoff_hz: f64, q: f64) -> Result<BiquadCoeffs, SignalError> {
    let (_s, cos_w0, alpha) = validate_rbj_inputs(sample_rate_hz, cutoff_hz, q)?;
    let one_plus_cos = 1.0 + cos_w0;
    let b0 = 0.5 * one_plus_cos;
    normalize_rbj(b0, -one_plus_cos, b0, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

pub fn rbj_bandpass(sample_rate_hz: f64, center_hz: f64, q: f64) -> Result<BiquadCoeffs, SignalError> {
    let (_s, cos_w0, alpha) = validate_rbj_inputs(sample_rate_hz, center_hz, q)?;
    normalize_rbj(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

pub fn rbj_notch(sample_rate_hz: f64, center_hz: f64, q: f64) -> Result<BiquadCoeffs, SignalError> {
    let (_s, cos_w0, alpha) = validate_rbj_inputs(sample_rate_hz, center_hz, q)?;
    normalize_rbj(1.0, -2.0 * cos_w0, 1.0, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

pub fn rbj_peaking_eq(sample_rate_hz: f64, center_hz: f64, q: f64, gain_db: f64) -> Result<BiquadCoeffs, SignalError> {
    if !gain_db.is_finite() { return Err(SignalError::NonFiniteInput); }
    let (_s, cos_w0, alpha) = validate_rbj_inputs(sample_rate_hz, center_hz, q)?;
    let a = pow10_f64(gain_db / 40.0);
    if !a.is_finite() || a <= 0.0 { return Err(SignalError::NumericRange); }
    normalize_rbj(1.0 + alpha * a, -2.0 * cos_w0, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * cos_w0, 1.0 - alpha / a)
}

pub fn rbj_low_shelf(sample_rate_hz: f64, cutoff_hz: f64, q: f64, gain_db: f64) -> Result<BiquadCoeffs, SignalError> {
    if !gain_db.is_finite() { return Err(SignalError::NonFiniteInput); }
    let (_s, cos_w0, alpha) = validate_rbj_inputs(sample_rate_hz, cutoff_hz, q)?;
    let a = pow10_f64(gain_db / 40.0);
    if !a.is_finite() || a <= 0.0 { return Err(SignalError::NumericRange); }
    let t = 2.0 * sqrt_f64(a) * alpha;
    normalize_rbj(
        a * ((a + 1.0) - (a - 1.0) * cos_w0 + t),
        2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0),
        a * ((a + 1.0) - (a - 1.0) * cos_w0 - t),
        (a + 1.0) + (a - 1.0) * cos_w0 + t,
        -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0),
        (a + 1.0) + (a - 1.0) * cos_w0 - t,
    )
}

pub fn rbj_high_shelf(sample_rate_hz: f64, cutoff_hz: f64, q: f64, gain_db: f64) -> Result<BiquadCoeffs, SignalError> {
    if !gain_db.is_finite() { return Err(SignalError::NonFiniteInput); }
    let (_s, cos_w0, alpha) = validate_rbj_inputs(sample_rate_hz, cutoff_hz, q)?;
    let a = pow10_f64(gain_db / 40.0);
    if !a.is_finite() || a <= 0.0 { return Err(SignalError::NumericRange); }
    let t = 2.0 * sqrt_f64(a) * alpha;
    normalize_rbj(
        a * ((a + 1.0) + (a - 1.0) * cos_w0 + t),
        -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0),
        a * ((a + 1.0) + (a - 1.0) * cos_w0 - t),
        (a + 1.0) - (a - 1.0) * cos_w0 + t,
        2.0 * ((a - 1.0) - (a + 1.0) * cos_w0),
        (a + 1.0) - (a - 1.0) * cos_w0 - t,
    )
}

/// ITU-R BS.1770-4 / EBU R128 K-weighting Stage 1 (acoustic head-model high-shelf).
pub fn design_k_weighting_stage1(sample_rate_hz: f64) -> BiquadCoeffs {
    const SHELF_F0: f64 = 1_681.974_450_955_533;
    const SHELF_Q: f64 = 0.707_175_236_955_419_6;
    const SHELF_GAIN_DB: f64 = 3.999_843_853_973_347;
    const SHELF_VB_EXP: f64 = 0.499_666_774_154_541_6;

    let fs = if sample_rate_hz.is_finite() && sample_rate_hz > 1.0 { sample_rate_hz } else { 1.0 };
    let k = tan_f64(PI_F64 * SHELF_F0 / fs);
    let vh = pow10_f64(SHELF_GAIN_DB / 20.0);
    let vb = pow_f64(vh, SHELF_VB_EXP);
    let kk = k * k;
    let kq = k / SHELF_Q;
    let a0 = 1.0 + kq + kk;
    BiquadCoeffs::new(
        (vh + vb * kq + kk) / a0,
        (2.0 * (kk - vh)) / a0,
        (vh - vb * kq + kk) / a0,
        (2.0 * (kk - 1.0)) / a0,
        (1.0 - kq + kk) / a0,
    )
}

/// ITU-R BS.1770-4 / EBU R128 K-weighting Stage 2 (RLB 2nd-order high-pass filter).
pub fn design_k_weighting_stage2(sample_rate_hz: f64) -> BiquadCoeffs {
    const HP_F0: f64 = 38.135_470_876_024_44;
    const HP_Q: f64 = 0.500_327_037_323_877_3;

    let fs = if sample_rate_hz.is_finite() && sample_rate_hz > 1.0 { sample_rate_hz } else { 1.0 };
    let kb = tan_f64(PI_F64 * HP_F0 / fs);
    let kkb = kb * kb;
    let kqb = kb / HP_Q;
    let a0b = 1.0 + kqb + kkb;
    BiquadCoeffs::new(
        1.0,
        -2.0,
        1.0,
        (2.0 * (kkb - 1.0)) / a0b,
        (1.0 - kqb + kkb) / a0b,
    )
}

/// Single-sample Direct-Form I difference-equation kernel in `f32`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn df1_step_f32(
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: &mut f32,
    x2: &mut f32,
    y1: &mut f32,
    y2: &mut f32,
    x: f32,
) -> f32 {
    let y = b0 * x + b1 * *x1 + b2 * *x2 - a1 * *y1 - a2 * *y2;
    *x2 = *x1;
    *x1 = x;
    *y2 = *y1;
    *y1 = y;
    y
}

/// Single-sample Direct-Form II Transposed difference-equation kernel in `f32`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn df2t_step_f32(
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: &mut f32,
    s2: &mut f32,
    input: f32,
) -> f32 {
    let y = b0 * input + *s1;
    *s1 = b1 * input - a1 * y + *s2;
    *s2 = b2 * input - a2 * y;
    y
}

/// Zero-heap Direct-Form II Transposed (DF2T) second-order biquad filter.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadDf2t {
    coeffs: BiquadCoeffs,
    s1: f64,
    s2: f64,
}

impl Default for BiquadDf2t {
    fn default() -> Self {
        Self { coeffs: BiquadCoeffs::IDENTITY, s1: 0.0, s2: 0.0 }
    }
}

impl BiquadDf2t {
    pub fn new(coeffs: BiquadCoeffs) -> Result<Self, SignalError> {
        if !coeffs.is_stable() { return Err(SignalError::InvalidParameter); }
        Ok(Self { coeffs, s1: 0.0, s2: 0.0 })
    }

    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs) -> Result<(), SignalError> {
        if !coeffs.is_stable() { return Err(SignalError::InvalidParameter); }
        self.coeffs = coeffs;
        Ok(())
    }

    #[inline]
    pub const fn coeffs(&self) -> BiquadCoeffs { self.coeffs }

    #[inline]
    pub const fn state(&self) -> (f64, f64) { (self.s1, self.s2) }

    #[inline]
    pub fn is_stable(&self) -> bool { self.coeffs.is_stable() }

    #[inline]
    pub fn reset(&mut self) { self.s1 = 0.0; self.s2 = 0.0; }

    #[inline]
    pub fn tick(&mut self, x: f64) -> f64 {
        let y = self.coeffs.b0 * x + self.s1;
        self.s1 = self.coeffs.b1 * x - self.coeffs.a1 * y + self.s2;
        self.s2 = self.coeffs.b2 * x - self.coeffs.a2 * y;
        y
    }

    #[inline]
    pub fn tick_f32(&mut self, x: f32) -> f32 { self.tick(f64::from(x)) as f32 }

    pub fn process_block_into(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), SignalError> {
        if output.len() < input.len() { return Err(SignalError::LengthMismatch); }
        if input.iter().any(|sample| !sample.is_finite()) { return Err(SignalError::NonFiniteInput); }
        if !self.coeffs.is_stable() { return Err(SignalError::InvalidParameter); }
        let mut sim = *self;
        for &sample in input {
            let y = sim.tick(f64::from(sample));
            if !y.is_finite() || !(y as f32).is_finite() { return Err(SignalError::NumericRange); }
        }
        for (dst, &sample) in output.iter_mut().zip(input.iter()) {
            *dst = self.tick(f64::from(sample)) as f32;
        }
        Ok(())
    }
}

/// Fixed-stage cascaded Direct-Form II Transposed biquad filter bank.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadedBiquadDf2t<const STAGES: usize> {
    stages: [BiquadDf2t; STAGES],
}

impl<const STAGES: usize> CascadedBiquadDf2t<STAGES> {
    pub fn new(coeffs: [BiquadCoeffs; STAGES]) -> Result<Self, SignalError> {
        if STAGES == 0 { return Err(SignalError::InvalidParameter); }
        let mut stages = [BiquadDf2t::default(); STAGES];
        let mut i = 0;
        while i < STAGES {
            stages[i] = BiquadDf2t::new(coeffs[i])?;
            i += 1;
        }
        Ok(Self { stages })
    }

    #[inline]
    pub fn is_stable(&self) -> bool { STAGES > 0 && self.stages.iter().all(BiquadDf2t::is_stable) }

    #[inline]
    pub fn stage(&self, index: usize) -> Option<&BiquadDf2t> { self.stages.get(index) }

    pub fn set_stage_coeffs(&mut self, index: usize, coeffs: BiquadCoeffs) -> Result<(), SignalError> {
        self.stages.get_mut(index).ok_or(SignalError::InvalidParameter)?.set_coeffs(coeffs)
    }

    #[inline]
    pub fn reset(&mut self) {
        for stage in &mut self.stages { stage.reset(); }
    }

    #[inline]
    pub fn tick(&mut self, x: f64) -> f64 {
        let mut acc = x;
        for stage in &mut self.stages { acc = stage.tick(acc); }
        acc
    }

    #[inline]
    pub fn tick_f32(&mut self, x: f32) -> f32 { self.tick(f64::from(x)) as f32 }

    pub fn process_block_into(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), SignalError> {
        if output.len() < input.len() { return Err(SignalError::LengthMismatch); }
        if input.iter().any(|sample| !sample.is_finite()) { return Err(SignalError::NonFiniteInput); }
        if !self.is_stable() { return Err(SignalError::InvalidParameter); }
        let mut sim = *self;
        for &sample in input {
            let y = sim.tick(f64::from(sample));
            if !y.is_finite() || !(y as f32).is_finite() { return Err(SignalError::NumericRange); }
        }
        for (dst, &sample) in output.iter_mut().zip(input.iter()) {
            *dst = self.tick(f64::from(sample)) as f32;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "filter_tests.rs"]
mod tests;

#[cfg(test)]
mod biquad_tests {
    use super::*;

    #[test]
    fn jury_stability_and_rbj_frequency_response() {
        let sr = 48_000.0;
        let lp = rbj_lowpass(sr, 1_000.0, 0.7071).unwrap();
        let hp = rbj_highpass(sr, 1_000.0, 0.7071).unwrap();
        let notch = rbj_notch(sr, 1_000.0, 4.0).unwrap();
        let peak = rbj_peaking_eq(sr, 1_000.0, 2.0, 6.0).unwrap();
        assert!(lp.is_stable() && hp.is_stable() && notch.is_stable() && peak.is_stable());
        assert!(!BiquadCoeffs::new(1.0, 0.0, 0.0, 0.0, 1.0).is_stable());
        assert!(!BiquadCoeffs::new(1.0, 0.0, 0.0, -1.3, 0.2).is_stable());

        assert!((lp.frequency_response_mag_sq(0.0).unwrap() - 1.0).abs() < 1e-12);
        assert!(lp.frequency_response_mag_sq(PI_F64).unwrap() < 1e-12);
        assert!(hp.frequency_response_mag_sq(0.0).unwrap() < 1e-12);
        assert!((hp.frequency_response_mag_sq(PI_F64).unwrap() - 1.0).abs() < 1e-12);

        let k1 = design_k_weighting_stage1(48_000.0);
        let k2 = design_k_weighting_stage2(48_000.0);
        assert!(k1.is_stable() && k2.is_stable());
        assert!((k1.b0 - 1.535_124_9).abs() < 1e-4);
        assert!((k2.a1 - (-1.990_047_5)).abs() < 1e-4);
    }

    #[test]
    fn cascaded_biquad_df2t_block_processing_attenuates_stopband() {
        let lp = rbj_lowpass(48_000.0, 2_000.0, 0.7071).unwrap();
        let mut cascade = CascadedBiquadDf2t::<2>::new([lp, lp]).unwrap();
        let mut nyq_in = [0.0f32; 256];
        for (i, s) in nyq_in.iter_mut().enumerate() { *s = if i % 2 == 0 { 1.0 } else { -1.0 }; }
        let mut nyq_out = [0.0f32; 256];
        cascade.process_block_into(&nyq_in, &mut nyq_out).unwrap();
        let tail_max = nyq_out[128..].iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert!(tail_max < 1e-6);
    }
}
