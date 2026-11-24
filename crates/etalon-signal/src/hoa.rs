//! Allocation-free Higher-Order Ambisonics (HOA) ACN / SN3D / N3D encoding, decoding,
//! and Ivanic-Ruedenberg real spherical-harmonic Wigner-D rotation up to order 3.

#[path = "hoa_wigner.rs"]
mod hoa_wigner;
pub use hoa_wigner::*;

use crate::{
    filter::{sin_cos_f64, sqrt_f64},
    SignalError,
};

pub const MAX_HOA_ORDER: usize = 3;
pub const MAX_HOA_CHANNELS: usize = (MAX_HOA_ORDER + 1) * (MAX_HOA_ORDER + 1);
pub const QUATERNION_NORM_SQ_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HoaOrder {
    Order0 = 0,
    Order1 = 1,
    Order2 = 2,
    Order3 = 3,
}

impl HoaOrder {
    #[inline]
    pub const fn from_u8(order: u8) -> Result<Self, SignalError> {
        match order {
            0 => Ok(Self::Order0),
            1 => Ok(Self::Order1),
            2 => Ok(Self::Order2),
            3 => Ok(Self::Order3),
            _ => Err(SignalError::InvalidParameter),
        }
    }

    #[inline]
    pub const fn from_channel_count(channels: usize) -> Result<Self, SignalError> {
        match channels {
            1 => Ok(Self::Order0),
            4 => Ok(Self::Order1),
            9 => Ok(Self::Order2),
            16 => Ok(Self::Order3),
            _ => Err(SignalError::InvalidParameter),
        }
    }

    #[inline]
    pub const fn degree(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn channel_count(self) -> usize {
        let l = self as usize;
        (l + 1) * (l + 1)
    }

    #[inline]
    pub const fn record_stride_bytes(self) -> usize {
        self.channel_count() * 4
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoaNormalization {
    Sn3d,
    N3d,
}

#[inline]
pub const fn acn_index(l: usize, m: isize) -> Result<usize, SignalError> {
    if l > MAX_HOA_ORDER {
        return Err(SignalError::InvalidParameter);
    }
    let l_signed = l as isize;
    if m < -l_signed || m > l_signed {
        return Err(SignalError::InvalidParameter);
    }
    Ok(((l_signed * (l_signed + 1)) + m) as usize)
}

#[inline]
pub const fn acn_degree_order(acn: usize) -> Result<(usize, isize), SignalError> {
    if acn >= MAX_HOA_CHANNELS {
        return Err(SignalError::InvalidParameter);
    }
    let l = match acn {
        0 => 0,
        1..=3 => 1,
        4..=8 => 2,
        _ => 3,
    };
    let center = (l * (l + 1)) as isize;
    Ok((l, (acn as isize) - center))
}

#[inline]
pub const fn n3d_over_sn3d_factor_f64(l: usize) -> Result<f64, SignalError> {
    match l {
        0 => Ok(1.0),
        1 => Ok(1.732_050_807_568_877_2),
        2 => Ok(2.236_067_977_499_79),
        3 => Ok(2.645_751_311_064_590_7),
        _ => Err(SignalError::InvalidParameter),
    }
}

#[inline]
pub const fn sn3d_over_n3d_factor_f64(l: usize) -> Result<f64, SignalError> {
    match l {
        0 => Ok(1.0),
        1 => Ok(0.577_350_269_189_625_8),
        2 => Ok(0.447_213_595_499_957_9),
        3 => Ok(0.377_964_473_009_227_2),
        _ => Err(SignalError::InvalidParameter),
    }
}

#[inline]
pub fn spherical_to_unit_cartesian_f64(
    azimuth_rad: f64,
    elevation_rad: f64,
) -> Result<[f64; 3], SignalError> {
    if !azimuth_rad.is_finite() || !elevation_rad.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    let half_pi = core::f64::consts::FRAC_PI_2 + 1e-12;
    if elevation_rad < -half_pi || elevation_rad > half_pi {
        return Err(SignalError::InvalidParameter);
    }
    let (sin_az, cos_az) = sin_cos_f64(azimuth_rad);
    let (sin_el, cos_el) = sin_cos_f64(
        elevation_rad.clamp(-core::f64::consts::FRAC_PI_2, core::f64::consts::FRAC_PI_2),
    );
    Ok([cos_el * cos_az, cos_el * sin_az, sin_el])
}

pub fn evaluate_hoa_basis_f64(
    order: HoaOrder,
    normalization: HoaNormalization,
    direction_xyz: [f64; 3],
    out_coeffs: &mut [f64],
) -> Result<usize, SignalError> {
    let ch = order.channel_count();
    if out_coeffs.len() < ch {
        return Err(SignalError::LengthMismatch);
    }
    let [dx, dy, dz] = direction_xyz;
    if !dx.is_finite() || !dy.is_finite() || !dz.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    let r2 = ((dx * dx) + (dy * dy)) + (dz * dz);
    if !r2.is_finite() || r2 <= 1e-30 {
        return Err(SignalError::InvalidParameter);
    }
    let inv_r = 1.0 / sqrt_f64(r2);
    let x = dx * inv_r;
    let y = dy * inv_r;
    let z = dz * inv_r;

    out_coeffs[0] = 1.0;

    if order.degree() >= 1 {
        let s1 = match normalization {
            HoaNormalization::Sn3d => 1.0,
            HoaNormalization::N3d => 1.732_050_807_568_877_2,
        };
        out_coeffs[1] = s1 * y;
        out_coeffs[2] = s1 * z;
        out_coeffs[3] = s1 * x;
    }

    if order.degree() >= 2 {
        const SQRT_3: f64 = 1.732_050_807_568_877_2;
        const HALF_SQRT_3: f64 = 0.866_025_403_784_438_6;
        let s2 = match normalization {
            HoaNormalization::Sn3d => 1.0,
            HoaNormalization::N3d => 2.236_067_977_499_79,
        };
        out_coeffs[4] = s2 * (SQRT_3 * x * y);
        out_coeffs[5] = s2 * (SQRT_3 * y * z);
        out_coeffs[6] = s2 * (0.5 * (3.0 * z * z - 1.0));
        out_coeffs[7] = s2 * (SQRT_3 * x * z);
        out_coeffs[8] = s2 * (HALF_SQRT_3 * (x * x - y * y));
    }

    if order.degree() >= 3 {
        const SQRT_5_8: f64 = 0.790_569_415_042_094_9;
        const SQRT_15: f64 = 3.872_983_346_207_417;
        const HALF_SQRT_15: f64 = 1.936_491_673_103_708_5;
        const SQRT_3_8: f64 = 0.612_372_435_695_794_5;
        let s3 = match normalization {
            HoaNormalization::Sn3d => 1.0,
            HoaNormalization::N3d => 2.645_751_311_064_590_7,
        };
        let x2 = x * x;
        let y2 = y * y;
        let z2 = z * z;
        out_coeffs[9] = s3 * (SQRT_5_8 * y * (3.0 * x2 - y2));
        out_coeffs[10] = s3 * (SQRT_15 * x * y * z);
        out_coeffs[11] = s3 * (SQRT_3_8 * y * (5.0 * z2 - 1.0));
        out_coeffs[12] = s3 * (0.5 * z * (5.0 * z2 - 3.0));
        out_coeffs[13] = s3 * (SQRT_3_8 * x * (5.0 * z2 - 1.0));
        out_coeffs[14] = s3 * (HALF_SQRT_15 * z * (x2 - y2));
        out_coeffs[15] = s3 * (SQRT_5_8 * x * (x2 - 3.0 * y2));
    }

    Ok(ch)
}

pub fn evaluate_hoa_basis_f32(
    order: HoaOrder,
    normalization: HoaNormalization,
    direction_xyz: [f32; 3],
    out_coeffs: &mut [f32],
) -> Result<usize, SignalError> {
    let ch = order.channel_count();
    if out_coeffs.len() < ch {
        return Err(SignalError::LengthMismatch);
    }
    let mut tmp = [0.0f64; MAX_HOA_CHANNELS];
    evaluate_hoa_basis_f64(
        order,
        normalization,
        [
            f64::from(direction_xyz[0]),
            f64::from(direction_xyz[1]),
            f64::from(direction_xyz[2]),
        ],
        &mut tmp[..ch],
    )?;
    for i in 0..ch {
        out_coeffs[i] = tmp[i] as f32;
    }
    Ok(ch)
}

pub fn convert_hoa_normalization_in_place_f64(
    order: HoaOrder,
    from: HoaNormalization,
    to: HoaNormalization,
    frame: &mut [f64],
) -> Result<(), SignalError> {
    let ch = order.channel_count();
    if frame.len() < ch {
        return Err(SignalError::LengthMismatch);
    }
    if frame[..ch].iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    if from == to {
        return Ok(());
    }
    for (acn, slot) in frame[..ch].iter_mut().enumerate() {
        let (l, _) = acn_degree_order(acn)?;
        let scale = match (from, to) {
            (HoaNormalization::Sn3d, HoaNormalization::N3d) => n3d_over_sn3d_factor_f64(l)?,
            (HoaNormalization::N3d, HoaNormalization::Sn3d) => sn3d_over_n3d_factor_f64(l)?,
            _ => 1.0,
        };
        *slot *= scale;
    }
    Ok(())
}

pub fn convert_hoa_stream_normalization_in_place_f32(
    order: HoaOrder,
    from: HoaNormalization,
    to: HoaNormalization,
    interleaved_frames: &mut [f32],
) -> Result<usize, SignalError> {
    let ch = order.channel_count();
    if !interleaved_frames.len().is_multiple_of(ch) {
        return Err(SignalError::LengthMismatch);
    }
    if interleaved_frames.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let num_frames = interleaved_frames.len() / ch;
    if from == to {
        return Ok(num_frames);
    }
    let mut scales = [1.0f64; MAX_HOA_CHANNELS];
    for (acn, slot) in scales[..ch].iter_mut().enumerate() {
        let (l, _) = acn_degree_order(acn)?;
        *slot = match (from, to) {
            (HoaNormalization::Sn3d, HoaNormalization::N3d) => n3d_over_sn3d_factor_f64(l)?,
            (HoaNormalization::N3d, HoaNormalization::Sn3d) => sn3d_over_n3d_factor_f64(l)?,
            _ => 1.0,
        };
    }
    for frame in interleaved_frames.chunks_exact_mut(ch) {
        for acn in 0..ch {
            frame[acn] = (f64::from(frame[acn]) * scales[acn]) as f32;
        }
    }
    Ok(num_frames)
}

pub fn encode_mono_to_hoa_accum_into(
    mono_samples: &[f32],
    order: HoaOrder,
    normalization: HoaNormalization,
    direction_xyz: [f32; 3],
    gain: f32,
    out_hoa_interleaved: &mut [f32],
) -> Result<usize, SignalError> {
    if !gain.is_finite() || mono_samples.iter().any(|s| !s.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let ch = order.channel_count();
    let required = mono_samples
        .len()
        .checked_mul(ch)
        .ok_or(SignalError::NumericRange)?;
    if out_hoa_interleaved.len() < required {
        return Err(SignalError::LengthMismatch);
    }
    let mut basis = [0.0f64; MAX_HOA_CHANNELS];
    evaluate_hoa_basis_f64(
        order,
        normalization,
        [
            f64::from(direction_xyz[0]),
            f64::from(direction_xyz[1]),
            f64::from(direction_xyz[2]),
        ],
        &mut basis[..ch],
    )?;
    let g = f64::from(gain);
    for (frame_idx, &sample) in mono_samples.iter().enumerate() {
        let s = f64::from(sample) * g;
        let base = frame_idx * ch;
        for acn in 0..ch {
            let val = f64::from(out_hoa_interleaved[base + acn]) + s * basis[acn];
            let val_f32 = val as f32;
            if !val_f32.is_finite() {
                return Err(SignalError::NumericRange);
            }
            out_hoa_interleaved[base + acn] = val_f32;
        }
    }
    Ok(mono_samples.len())
}

#[inline]
pub fn max_re_degree_weight_f64(order: HoaOrder, l: usize) -> Result<f64, SignalError> {
    let max_l = order.degree();
    if l > max_l {
        return Err(SignalError::InvalidParameter);
    }
    if max_l == 0 || l == 0 {
        return Ok(1.0);
    }
    let theta_rad = (137.9 * core::f64::consts::PI / 180.0) / ((max_l as f64) + 1.51);
    let (_, mu) = sin_cos_f64(theta_rad);
    match l {
        0 => Ok(1.0),
        1 => Ok(mu),
        2 => Ok(0.5 * (3.0 * mu * mu - 1.0)),
        3 => Ok(0.5 * mu * (5.0 * mu * mu - 3.0)),
        _ => Err(SignalError::InvalidParameter),
    }
}

pub fn decode_hoa_frame_to_direction_f64(
    order: HoaOrder,
    normalization: HoaNormalization,
    frame_hoa: &[f64],
    look_direction_xyz: [f64; 3],
    apply_max_re: bool,
) -> Result<f64, SignalError> {
    let ch = order.channel_count();
    if frame_hoa.len() < ch {
        return Err(SignalError::LengthMismatch);
    }
    if frame_hoa[..ch].iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let mut y_sn3d = [0.0f64; MAX_HOA_CHANNELS];
    evaluate_hoa_basis_f64(
        order,
        HoaNormalization::Sn3d,
        look_direction_xyz,
        &mut y_sn3d[..ch],
    )?;

    let mut sum = 0.0f64;
    for acn in 0..ch {
        let (l, _) = acn_degree_order(acn)?;
        let b_sn3d = match normalization {
            HoaNormalization::Sn3d => frame_hoa[acn],
            HoaNormalization::N3d => frame_hoa[acn] * sn3d_over_n3d_factor_f64(l)?,
        };
        let deg_weight = (2 * l + 1) as f64;
        let re_w = if apply_max_re {
            max_re_degree_weight_f64(order, l)?
        } else {
            1.0
        };
        sum += deg_weight * re_w * y_sn3d[acn] * b_sn3d;
    }
    Ok(sum / (ch as f64))
}

pub fn decode_hoa_stream_to_speakers_into(
    order: HoaOrder,
    normalization: HoaNormalization,
    hoa_interleaved: &[f32],
    speaker_directions_xyz: &[[f32; 3]],
    apply_max_re: bool,
    out_speaker_interleaved: &mut [f32],
) -> Result<usize, SignalError> {
    if speaker_directions_xyz.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let ch = order.channel_count();
    if !hoa_interleaved.len().is_multiple_of(ch) {
        return Err(SignalError::LengthMismatch);
    }
    if hoa_interleaved.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let num_frames = hoa_interleaved.len() / ch;
    let n_spk = speaker_directions_xyz.len();
    let required = num_frames
        .checked_mul(n_spk)
        .ok_or(SignalError::NumericRange)?;
    if out_speaker_interleaved.len() < required {
        return Err(SignalError::LengthMismatch);
    }

    let inv_n_spk = 1.0 / (n_spk as f64);
    for (spk_idx, &dir) in speaker_directions_xyz.iter().enumerate() {
        let mut y_sn3d = [0.0f64; MAX_HOA_CHANNELS];
        evaluate_hoa_basis_f64(
            order,
            HoaNormalization::Sn3d,
            [f64::from(dir[0]), f64::from(dir[1]), f64::from(dir[2])],
            &mut y_sn3d[..ch],
        )?;
        let mut weights = [0.0f64; MAX_HOA_CHANNELS];
        for acn in 0..ch {
            let (l, _) = acn_degree_order(acn)?;
            let to_sn3d = match normalization {
                HoaNormalization::Sn3d => 1.0,
                HoaNormalization::N3d => sn3d_over_n3d_factor_f64(l)?,
            };
            let re_w = if apply_max_re {
                max_re_degree_weight_f64(order, l)?
            } else {
                1.0
            };
            weights[acn] = ((2 * l + 1) as f64) * re_w * y_sn3d[acn] * to_sn3d * inv_n_spk;
        }

        for frame_idx in 0..num_frames {
            let in_base = frame_idx * ch;
            let mut acc = 0.0f64;
            for acn in 0..ch {
                acc += f64::from(hoa_interleaved[in_base + acn]) * weights[acn];
            }
            let out_f32 = acc as f32;
            if !out_f32.is_finite() {
                return Err(SignalError::NumericRange);
            }
            out_speaker_interleaved[frame_idx * n_spk + spk_idx] = out_f32;
        }
    }
    Ok(num_frames)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoaDecodeShape {
    pub n_frames: usize,
    pub n_channels: usize,
    pub n_speakers: usize,
}

impl HoaDecodeShape {
    pub fn input_len(self) -> Result<usize, SignalError> {
        self.n_frames
            .checked_mul(self.n_channels)
            .ok_or(SignalError::NumericRange)
    }

    pub fn matrix_len(self) -> Result<usize, SignalError> {
        self.n_speakers
            .checked_mul(self.n_channels)
            .ok_or(SignalError::NumericRange)
    }

    pub fn output_len(self) -> Result<usize, SignalError> {
        self.n_frames
            .checked_mul(self.n_speakers)
            .ok_or(SignalError::NumericRange)
    }
}

pub fn hoa_decode_matrix_apply_into(
    hoa_frames: &[f32],
    decode_matrix: &[f32],
    shape: HoaDecodeShape,
    out_speakers: &mut [f32],
) -> Result<usize, SignalError> {
    let out_len = shape.output_len()?;
    if hoa_frames.len() != shape.input_len()?
        || decode_matrix.len() != shape.matrix_len()?
        || out_speakers.len() < out_len
    {
        return Err(SignalError::LengthMismatch);
    }
    if out_len == 0 {
        return Ok(0);
    }
    for frame in 0..shape.n_frames {
        let in_base = frame * shape.n_channels;
        for spk in 0..shape.n_speakers {
            let mat_base = spk * shape.n_channels;
            let mut acc = 0.0f32;
            for c in 0..shape.n_channels {
                acc += hoa_frames[in_base + c] * decode_matrix[mat_base + c];
            }
            out_speakers[frame * shape.n_speakers + spk] = acc;
        }
    }
    Ok(out_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hoa_basis_and_wigner_rotation_oracles() {
        let mut x_sn3d = [0.0f64; 16];
        let mut y_sn3d = [0.0f64; 16];
        evaluate_hoa_basis_f64(
            HoaOrder::Order3,
            HoaNormalization::Sn3d,
            [1.0, 0.0, 0.0],
            &mut x_sn3d,
        )
        .unwrap();
        evaluate_hoa_basis_f64(
            HoaOrder::Order3,
            HoaNormalization::Sn3d,
            [0.0, 1.0, 0.0],
            &mut y_sn3d,
        )
        .unwrap();
        assert_eq!(&x_sn3d[..4], &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(&y_sn3d[..4], &[1.0, 1.0, 0.0, 0.0]);

        let s = core::f64::consts::FRAC_1_SQRT_2;
        let rot = HoaWignerRotation::from_quaternion_xyzw([0.0, 0.0, s, s]).unwrap();
        let mut out = [0.0f64; 16];
        rot.apply_frame_f64(HoaOrder::Order3, &x_sn3d, &mut out)
            .unwrap();
        for c in 0..16 {
            assert!((out[c] - y_sn3d[c]).abs() <= 1e-14);
        }
    }
}
