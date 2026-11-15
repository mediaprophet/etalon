//! Real spherical-harmonic Wigner-D / Ivanic-Ruedenberg rotation blocks up to order 3
//! and active Hamilton quaternion / symmetric tensor transforms.

use crate::{
    filter::sqrt_f64,
    hoa::{HoaOrder, MAX_HOA_CHANNELS, QUATERNION_NORM_SQ_TOLERANCE},
    SignalError,
};

#[inline]
pub fn quaternion_ordered_norm_sq_f64(xyzw: [f64; 4]) -> f64 {
    let [x, y, z, w] = xyzw;
    let xy = (x * x) + (y * y);
    let xyz = xy + (z * z);
    xyz + (w * w)
}

#[inline]
pub fn admit_and_derive_unit_quaternion_f64(
    xyzw: [f64; 4],
) -> Result<([f64; 4], f64), SignalError> {
    let [x, y, z, w] = xyzw;
    if !x.is_finite() || !y.is_finite() || !z.is_finite() || !w.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    let n2 = quaternion_ordered_norm_sq_f64(xyzw);
    if !n2.is_finite() || (n2 - 1.0).abs() > QUATERNION_NORM_SQ_TOLERANCE {
        return Err(SignalError::InvalidParameter);
    }
    let n = sqrt_f64(n2);
    if !n.is_finite() || n <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    Ok(([x / n, y / n, z / n, w / n], n2))
}

#[inline]
pub fn hamilton_product_xyzw(q_parent: [f64; 4], q_local: [f64; 4]) -> [f64; 4] {
    let [x1, y1, z1, w1] = q_parent;
    let [x2, y2, z2, w2] = q_local;
    [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
}

#[inline]
pub const fn quaternion_conjugate_xyzw(xyzw: [f64; 4]) -> [f64; 4] {
    [-xyzw[0], -xyzw[1], -xyzw[2], xyzw[3]]
}

pub fn rotation_matrix_from_quaternion_f64(
    xyzw: [f64; 4],
) -> Result<[[f64; 3]; 3], SignalError> {
    let ([qx, qy, qz, qw], _) = admit_and_derive_unit_quaternion_f64(xyzw)?;
    let xx = qx * qx;
    let yy = qy * qy;
    let zz = qz * qz;
    let xy = qx * qy;
    let xz = qx * qz;
    let yz = qy * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    Ok([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])
}

#[inline]
pub fn rotate_vector_active_f64(
    xyzw: [f64; 4],
    v_local: [f64; 3],
) -> Result<[f64; 3], SignalError> {
    if v_local.iter().any(|c| !c.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let r = rotation_matrix_from_quaternion_f64(xyzw)?;
    Ok([
        r[0][0] * v_local[0] + r[0][1] * v_local[1] + r[0][2] * v_local[2],
        r[1][0] * v_local[0] + r[1][1] * v_local[1] + r[1][2] * v_local[2],
        r[2][0] * v_local[0] + r[2][1] * v_local[1] + r[2][2] * v_local[2],
    ])
}

pub fn rotate_symmetric_tensor_active_f64(
    xyzw: [f64; 4],
    tensor_xx_yy_zz_xy_yz_xz: [f64; 6],
) -> Result<[f64; 6], SignalError> {
    if tensor_xx_yy_zz_xy_yz_xz.iter().any(|c| !c.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let r = rotation_matrix_from_quaternion_f64(xyzw)?;
    let [txx, tyy, tzz, txy, tyz, txz] = tensor_xx_yy_zz_xy_yz_xz;
    let t = [[txx, txy, txz], [txy, tyy, tyz], [txz, tyz, tzz]];
    let mut rt = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rt[i][j] = r[i][0] * t[0][j] + r[i][1] * t[1][j] + r[i][2] * t[2][j];
        }
    }
    let mut rtrt = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rtrt[i][j] = rt[i][0] * r[j][0] + rt[i][1] * r[j][1] + rt[i][2] * r[j][2];
        }
    }
    Ok([
        rtrt[0][0],
        rtrt[1][1],
        rtrt[2][2],
        rtrt[0][1],
        rtrt[1][2],
        rtrt[0][2],
    ])
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoaWignerRotation {
    pub cartesian_r: [[f64; 3]; 3],
    pub d1: [[f64; 3]; 3],
    pub d2: [[f64; 5]; 5],
    pub d3: [[f64; 7]; 7],
}

#[inline]
fn d1_get(d1: &[[f64; 3]; 3], m: isize, mp: isize) -> f64 {
    d1[(m + 1) as usize][(mp + 1) as usize]
}

#[inline]
fn prev_get<const N: usize>(prev: &[[f64; N]; N], l_prev: isize, a: isize, b: isize) -> f64 {
    prev[(a + l_prev) as usize][(b + l_prev) as usize]
}

#[inline]
fn ivanic_p<const N_PREV: usize>(
    l: isize,
    i: isize,
    a: isize,
    mp: isize,
    d1: &[[f64; 3]; 3],
    d_prev: &[[f64; N_PREV]; N_PREV],
) -> f64 {
    let l_prev = l - 1;
    if a < -l_prev || a > l_prev {
        return 0.0;
    }
    if mp == l {
        d1_get(d1, i, 1) * prev_get(d_prev, l_prev, a, l_prev)
            - d1_get(d1, i, -1) * prev_get(d_prev, l_prev, a, -l_prev)
    } else if mp == -l {
        d1_get(d1, i, 1) * prev_get(d_prev, l_prev, a, -l_prev)
            + d1_get(d1, i, -1) * prev_get(d_prev, l_prev, a, l_prev)
    } else {
        d1_get(d1, i, 0) * prev_get(d_prev, l_prev, a, mp)
    }
}

fn ivanic_entry<const N_PREV: usize>(
    l: isize,
    m: isize,
    mp: isize,
    d1: &[[f64; 3]; 3],
    d_prev: &[[f64; N_PREV]; N_PREV],
) -> f64 {
    let abs_m = m.abs();
    let abs_mp = mp.abs();
    let denom = if abs_mp < l {
        ((l + mp) * (l - mp)) as f64
    } else {
        (2 * l * (2 * l - 1)) as f64
    };

    let delta_m0 = if m == 0 { 1.0 } else { 0.0 };
    let u_coeff = sqrt_f64((((l + m) * (l - m)) as f64) / denom);
    let v_coeff = 0.5
        * sqrt_f64(((1.0 + delta_m0) * (((l + abs_m - 1) * (l + abs_m)) as f64)) / denom)
        * (1.0 - 2.0 * delta_m0);
    let w_coeff = if abs_m > l - 2 {
        0.0
    } else {
        -0.5 * sqrt_f64((((l - abs_m - 1) * (l - abs_m)) as f64) / denom) * (1.0 - delta_m0)
    };

    let big_u = if abs_m < l {
        ivanic_p(l, 0, m, mp, d1, d_prev)
    } else {
        0.0
    };

    let big_v = if m == 0 {
        ivanic_p(l, 1, 1, mp, d1, d_prev) + ivanic_p(l, -1, -1, mp, d1, d_prev)
    } else if m > 0 {
        if m == 1 {
            core::f64::consts::SQRT_2 * ivanic_p(l, 1, 0, mp, d1, d_prev)
        } else {
            ivanic_p(l, 1, m - 1, mp, d1, d_prev) - ivanic_p(l, -1, -m + 1, mp, d1, d_prev)
        }
    } else if m == -1 {
        core::f64::consts::SQRT_2 * ivanic_p(l, -1, 0, mp, d1, d_prev)
    } else {
        ivanic_p(l, 1, m + 1, mp, d1, d_prev) + ivanic_p(l, -1, -m - 1, mp, d1, d_prev)
    };

    let big_w = if w_coeff == 0.0 || m == 0 {
        0.0
    } else if m > 0 {
        ivanic_p(l, 1, m + 1, mp, d1, d_prev) + ivanic_p(l, -1, -m - 1, mp, d1, d_prev)
    } else {
        ivanic_p(l, 1, m - 1, mp, d1, d_prev) - ivanic_p(l, -1, -m + 1, mp, d1, d_prev)
    };

    u_coeff * big_u + v_coeff * big_v + w_coeff * big_w
}

impl HoaWignerRotation {
    pub fn from_quaternion_xyzw(xyzw: [f64; 4]) -> Result<Self, SignalError> {
        let r = rotation_matrix_from_quaternion_f64(xyzw)?;
        Ok(Self::from_cartesian_rotation_unchecked(r))
    }

    pub fn from_cartesian_rotation_unchecked(r: [[f64; 3]; 3]) -> Self {
        let perm = [1usize, 2usize, 0usize];
        let mut d1 = [[0.0f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                d1[row][col] = r[perm[row]][perm[col]];
            }
        }

        let mut d2 = [[0.0f64; 5]; 5];
        for m in -2..=2 {
            for mp in -2..=2 {
                d2[(m + 2) as usize][(mp + 2) as usize] = ivanic_entry(2, m, mp, &d1, &d1);
            }
        }

        let mut d3 = [[0.0f64; 7]; 7];
        for m in -3..=3 {
            for mp in -3..=3 {
                d3[(m + 3) as usize][(mp + 3) as usize] = ivanic_entry(3, m, mp, &d1, &d2);
            }
        }

        Self {
            cartesian_r: r,
            d1,
            d2,
            d3,
        }
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        let mut r_t = [[0.0f64; 3]; 3];
        let mut d1_t = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r_t[i][j] = self.cartesian_r[j][i];
                d1_t[i][j] = self.d1[j][i];
            }
        }
        let mut d2_t = [[0.0f64; 5]; 5];
        for i in 0..5 {
            for j in 0..5 {
                d2_t[i][j] = self.d2[j][i];
            }
        }
        let mut d3_t = [[0.0f64; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                d3_t[i][j] = self.d3[j][i];
            }
        }
        Self {
            cartesian_r: r_t,
            d1: d1_t,
            d2: d2_t,
            d3: d3_t,
        }
    }

    pub fn apply_frame_f64(
        &self,
        order: HoaOrder,
        in_frame: &[f64],
        out_frame: &mut [f64],
    ) -> Result<(), SignalError> {
        let ch = order.channel_count();
        if in_frame.len() < ch || out_frame.len() < ch {
            return Err(SignalError::LengthMismatch);
        }
        if in_frame[..ch].iter().any(|v| !v.is_finite()) {
            return Err(SignalError::NonFiniteInput);
        }

        out_frame[0] = in_frame[0];

        if order.degree() >= 1 {
            for row in 0..3 {
                let mut acc = 0.0f64;
                for col in 0..3 {
                    acc += self.d1[row][col] * in_frame[1 + col];
                }
                out_frame[1 + row] = acc;
            }
        }

        if order.degree() >= 2 {
            for row in 0..5 {
                let mut acc = 0.0f64;
                for col in 0..5 {
                    acc += self.d2[row][col] * in_frame[4 + col];
                }
                out_frame[4 + row] = acc;
            }
        }

        if order.degree() >= 3 {
            for row in 0..7 {
                let mut acc = 0.0f64;
                for col in 0..7 {
                    acc += self.d3[row][col] * in_frame[9 + col];
                }
                out_frame[9 + row] = acc;
            }
        }

        Ok(())
    }

    pub fn apply_frame_in_place_f64(
        &self,
        order: HoaOrder,
        frame: &mut [f64],
    ) -> Result<(), SignalError> {
        let ch = order.channel_count();
        if frame.len() < ch {
            return Err(SignalError::LengthMismatch);
        }
        let mut tmp = [0.0f64; MAX_HOA_CHANNELS];
        tmp[..ch].copy_from_slice(&frame[..ch]);
        self.apply_frame_f64(order, &tmp[..ch], &mut frame[..ch])
    }

    pub fn apply_stream_into_f32(
        &self,
        order: HoaOrder,
        in_stream: &[f32],
        out_stream: &mut [f32],
    ) -> Result<usize, SignalError> {
        let ch = order.channel_count();
        if !in_stream.len().is_multiple_of(ch) || out_stream.len() < in_stream.len() {
            return Err(SignalError::LengthMismatch);
        }
        if in_stream.iter().any(|v| !v.is_finite()) {
            return Err(SignalError::NonFiniteInput);
        }
        let num_frames = in_stream.len() / ch;
        let mut in_f64 = [0.0f64; MAX_HOA_CHANNELS];
        let mut out_f64 = [0.0f64; MAX_HOA_CHANNELS];

        for frame_idx in 0..num_frames {
            let base = frame_idx * ch;
            for c in 0..ch {
                in_f64[c] = f64::from(in_stream[base + c]);
            }
            self.apply_frame_f64(order, &in_f64[..ch], &mut out_f64[..ch])?;
            for c in 0..ch {
                let v = out_f64[c] as f32;
                if !v.is_finite() {
                    return Err(SignalError::NumericRange);
                }
                out_stream[base + c] = v;
            }
        }
        Ok(num_frames)
    }

    pub fn max_orthogonality_residual(&self, order: HoaOrder) -> f64 {
        let mut max_err = 0.0f64;
        if order.degree() >= 1 {
            for i in 0..3 {
                for j in 0..3 {
                    let mut dot = 0.0f64;
                    for k in 0..3 {
                        dot += self.d1[k][i] * self.d1[k][j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    let err = (dot - target).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
        }
        if order.degree() >= 2 {
            for i in 0..5 {
                for j in 0..5 {
                    let mut dot = 0.0f64;
                    for k in 0..5 {
                        dot += self.d2[k][i] * self.d2[k][j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    let err = (dot - target).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
        }
        if order.degree() >= 3 {
            for i in 0..7 {
                for j in 0..7 {
                    let mut dot = 0.0f64;
                    for k in 0..7 {
                        dot += self.d3[k][i] * self.d3[k][j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    let err = (dot - target).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
        }
        max_err
    }
}
