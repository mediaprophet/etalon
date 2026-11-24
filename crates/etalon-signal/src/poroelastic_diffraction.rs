//! Biot (1956/1962) Poroelastic Wave Dispersion, 21-Constant Anisotropic Elastodynamic
//! Christoffel Solver, and Kouyoumjian-Pathak UTD Wedge Diffraction (`#![no_std]`).
//!
//! Closes R03 Gap #3 for advanced continuum/wave acoustic models in `etalon-signal`
//! using allocation-free caller-owned structs and `crate::filter` `no_std` transcendentals.

#![allow(clippy::needless_range_loop, clippy::should_implement_trait)]

use crate::filter::{cos_f64, round_symmetric_f64, sin_cos_f64, sin_f64, sqrt_f64};
use crate::SignalError;

const PI: f64 = core::f64::consts::PI;
const FRAC_PI_4: f64 = core::f64::consts::FRAC_PI_4;

/// Complex double-precision scalar $(z_{\text{re}} + i z_{\text{im}})$ for `no_std` wave algebra.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline]
    pub fn abs(self) -> f64 {
        sqrt_f64(self.re * self.re + self.im * self.im)
    }
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    #[inline]
    pub fn div(self, rhs: Self) -> Result<Self, SignalError> {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        if !denom.is_finite() || denom <= 0.0 {
            return Err(SignalError::NumericRange);
        }
        Ok(Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        ))
    }
    /// Principal square root with $\text{Re}(\sqrt{z}) \ge 0$.
    #[inline]
    pub fn sqrt_principal(self) -> Self {
        let r = self.abs();
        if r == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let u = sqrt_f64(0.5 * (r + self.re).max(0.0));
        let v_mag = sqrt_f64(0.5 * (r - self.re).max(0.0));
        Self::new(u, if self.im < 0.0 { -v_mag } else { v_mag })
    }
}

/// Biot (1956/1962) saturated porous medium parameters with Johnson-Champoux-Allard (JCA) viscous drag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiotPoroelasticParams {
    pub porosity: f64,
    pub rho_fluid_kg_m3: f64,
    pub rho_solid_kg_m3: f64,
    pub tortuosity_inf: f64,
    pub permeability_m2: f64,
    pub viscosity_pa_s: f64,
    pub biot_willis_alpha: f64,
    pub biot_modulus_m_pa: f64,
    pub frame_bulk_modulus_pa: f64,
    pub frame_shear_modulus_pa: f64,
}

impl BiotPoroelasticParams {
    pub fn validate(&self) -> Result<(), SignalError> {
        for v in [
            self.porosity,
            self.rho_fluid_kg_m3,
            self.rho_solid_kg_m3,
            self.tortuosity_inf,
            self.permeability_m2,
            self.viscosity_pa_s,
            self.biot_willis_alpha,
            self.biot_modulus_m_pa,
            self.frame_bulk_modulus_pa,
            self.frame_shear_modulus_pa,
        ] {
            if !v.is_finite() {
                return Err(SignalError::NonFiniteInput);
            }
        }
        if self.porosity <= 0.0
            || self.porosity >= 1.0
            || self.rho_fluid_kg_m3 <= 0.0
            || self.rho_solid_kg_m3 <= 0.0
            || self.tortuosity_inf < 1.0
            || self.permeability_m2 <= 0.0
            || self.viscosity_pa_s < 0.0
            || self.biot_willis_alpha < self.porosity
            || self.biot_willis_alpha > 1.0
            || self.biot_modulus_m_pa <= 0.0
            || self.frame_bulk_modulus_pa <= 0.0
            || self.frame_shear_modulus_pa <= 0.0
        {
            return Err(SignalError::InvalidParameter);
        }
        Ok(())
    }

    #[inline]
    pub fn bulk_density_kg_m3(&self) -> f64 {
        (1.0 - self.porosity) * self.rho_solid_kg_m3 + self.porosity * self.rho_fluid_kg_m3
    }

    #[inline]
    pub fn undrained_p_modulus_h_pa(&self) -> f64 {
        self.frame_bulk_modulus_pa
            + (4.0 / 3.0) * self.frame_shear_modulus_pa
            + self.biot_willis_alpha * self.biot_willis_alpha * self.biot_modulus_m_pa
    }
}

/// Phase velocities (m/s) and attenuation coefficients (Np/m) for Biot fast ($P_1$), slow ($P_2$), and shear ($S$) waves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiotWaveDispersion {
    pub omega_rad_s: f64,
    pub c_p1_m_s: f64,
    pub alpha_p1_np_m: f64,
    pub c_p2_m_s: f64,
    pub alpha_p2_np_m: f64,
    pub c_s_m_s: f64,
    pub alpha_s_np_m: f64,
}

/// Solve the Biot (1956/1962) complex quadratic dispersion relation at angular frequency $\omega > 0$.
pub fn solve_biot_poroelastic_dispersion(
    params: &BiotPoroelasticParams,
    omega_rad_s: f64,
) -> Result<BiotWaveDispersion, SignalError> {
    params.validate()?;
    if !omega_rad_s.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if omega_rad_s <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }

    let (rho, rho_f) = (params.bulk_density_kg_m3(), params.rho_fluid_kg_m3);
    let h_mod = params.undrained_p_modulus_h_pa();
    let c_mod = params.biot_willis_alpha * params.biot_modulus_m_pa;
    let m_mod = params.biot_modulus_m_pa;

    let rho_f_inf = params.tortuosity_inf * rho_f / params.porosity;
    let rho_tilde = if params.viscosity_pa_s > 0.0 {
        let omega_c = (params.viscosity_pa_s * params.porosity)
            / (params.permeability_m2 * rho_f * params.tortuosity_inf);
        let jca = Complex64::new(1.0, 0.5 * omega_rad_s / omega_c).sqrt_principal();
        let drag = Complex64::new(
            0.0,
            params.viscosity_pa_s / (omega_rad_s * params.permeability_m2),
        )
        .mul(jca);
        Complex64::new(rho_f_inf, 0.0).add(drag)
    } else {
        Complex64::new(rho_f_inf, 0.0)
    };

    let s_num = Complex64::new(rho, 0.0).sub(Complex64::new(rho_f * rho_f, 0.0).div(rho_tilde)?);
    let s_slowness = s_num
        .scale(1.0 / params.frame_shear_modulus_pa)
        .sqrt_principal();
    if s_slowness.re <= 0.0 {
        return Err(SignalError::NumericRange);
    }

    let a_quad = h_mod * m_mod - c_mod * c_mod;
    if a_quad <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    let b_quad = rho_tilde
        .scale(h_mod)
        .add(Complex64::new(rho * m_mod - 2.0 * rho_f * c_mod, 0.0));
    let c_quad = rho_tilde.scale(rho).sub(Complex64::new(rho_f * rho_f, 0.0));
    let disc = b_quad
        .mul(b_quad)
        .sub(c_quad.scale(4.0 * a_quad))
        .sqrt_principal();
    let inv_2a = 0.5 / a_quad;
    let s1 = b_quad.sub(disc).scale(inv_2a).sqrt_principal();
    let s2 = b_quad.add(disc).scale(inv_2a).sqrt_principal();
    if s1.re <= 0.0 || s2.re <= 0.0 {
        return Err(SignalError::NumericRange);
    }
    let (fast, slow) = if s1.re <= s2.re { (s1, s2) } else { (s2, s1) };

    Ok(BiotWaveDispersion {
        omega_rad_s,
        c_p1_m_s: 1.0 / fast.re,
        alpha_p1_np_m: (omega_rad_s * fast.im).max(0.0),
        c_p2_m_s: 1.0 / slow.re,
        alpha_p2_np_m: (omega_rad_s * slow.im).max(0.0),
        c_s_m_s: 1.0 / s_slowness.re,
        alpha_s_np_m: (omega_rad_s * s_slowness.im).max(0.0),
    })
}

/// Output of the 21-constant anisotropic elastodynamic Christoffel solver (`qP`, `qSV`, `qSH`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChristoffelSolution {
    pub direction_unit: [f64; 3],
    pub eigenvalues_pa: [f64; 3],
    pub phase_velocities_m_s: [f64; 3],
    pub polarizations: [[f64; 3]; 3],
    pub group_velocities_m_s: [[f64; 3]; 3],
}

#[inline]
const fn voigt_index(i: usize, j: usize) -> usize {
    match (i, j) {
        (0, 0) => 0,
        (1, 1) => 1,
        (2, 2) => 2,
        (1, 2) | (2, 1) => 3,
        (0, 2) | (2, 0) => 4,
        _ => 5,
    }
}

/// Verify that a $6 \times 6$ Voigt stiffness matrix is finite, symmetric, and positive-definite via Cholesky.
pub fn validate_voigt_stiffness_6x6(c_voigt: &[[f64; 6]; 6]) -> Result<(), SignalError> {
    let mut max_abs = 1.0_f64;
    for r in 0..6 {
        for c in 0..6 {
            let v = c_voigt[r][c];
            if !v.is_finite() {
                return Err(SignalError::NonFiniteInput);
            }
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }
    }
    for r in 0..6 {
        for c in (r + 1)..6 {
            if (c_voigt[r][c] - c_voigt[c][r]).abs() > 1e-9 * max_abs {
                return Err(SignalError::InvalidParameter);
            }
        }
    }
    let mut l_mat = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = c_voigt[i][j];
            for k in 0..j {
                sum -= l_mat[i][k] * l_mat[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(SignalError::InvalidParameter);
                }
                l_mat[i][j] = sqrt_f64(sum);
            } else {
                l_mat[i][j] = sum / l_mat[j][j];
            }
        }
    }
    Ok(())
}

/// Solve the 3D anisotropic Christoffel eigenvalue problem $\Gamma_{ik} p_k = \rho v_p^2 p_i$
/// and compute group velocity vectors $v_{g,j}^{(m)} = \frac{1}{\rho v_{p,m}} C_{ijkl} p_i^{(m)} p_k^{(m)} n_l$.
pub fn solve_christoffel_anisotropic_21(
    c_voigt: &[[f64; 6]; 6],
    rho_kg_m3: f64,
    direction: [f64; 3],
) -> Result<ChristoffelSolution, SignalError> {
    validate_voigt_stiffness_6x6(c_voigt)?;
    if !rho_kg_m3.is_finite() || direction.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    if rho_kg_m3 <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    let norm_sq =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    if norm_sq <= 1e-24 {
        return Err(SignalError::InvalidParameter);
    }
    let inv_norm = 1.0 / sqrt_f64(norm_sq);
    let n = [
        direction[0] * inv_norm,
        direction[1] * inv_norm,
        direction[2] * inv_norm,
    ];

    let mut a = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            let mut acc = 0.0_f64;
            for j in 0..3 {
                let ij = voigt_index(i, j);
                for l in 0..3 {
                    acc += c_voigt[ij][voigt_index(k, l)] * n[j] * n[l];
                }
            }
            a[i][k] = acc;
        }
    }

    let mut v_eig = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..16 {
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() <= 1e-15 * (a[p][p].abs() + a[q][q].abs()) {
                continue;
            }
            let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + sqrt_f64(1.0 + tau * tau))
            } else {
                -1.0 / (-tau + sqrt_f64(1.0 + tau * tau))
            };
            let c = 1.0 / sqrt_f64(1.0 + t * t);
            let s = t * c;
            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            let r = 3 - p - q;
            let (apr, aqr) = (a[p][r], a[q][r]);
            a[p][r] = c * apr - s * aqr;
            a[r][p] = a[p][r];
            a[q][r] = s * apr + c * aqr;
            a[r][q] = a[q][r];
            for row in 0..3 {
                let (vpr, vqr) = (v_eig[row][p], v_eig[row][q]);
                v_eig[row][p] = c * vpr - s * vqr;
                v_eig[row][q] = s * vpr + c * vqr;
            }
        }
    }

    let mut order = [0usize, 1, 2];
    if a[order[1]][order[1]] > a[order[0]][order[0]] {
        order.swap(0, 1);
    }
    if a[order[2]][order[2]] > a[order[1]][order[1]] {
        order.swap(1, 2);
    }
    if a[order[1]][order[1]] > a[order[0]][order[0]] {
        order.swap(0, 1);
    }

    let mut eigenvalues_pa = [0.0_f64; 3];
    let mut phase_velocities_m_s = [0.0_f64; 3];
    let mut polarizations = [[0.0_f64; 3]; 3];
    let mut group_velocities_m_s = [[0.0_f64; 3]; 3];

    for (m, &col) in order.iter().enumerate() {
        let lam = a[col][col];
        if lam <= 0.0 {
            return Err(SignalError::NumericRange);
        }
        let vp = sqrt_f64(lam / rho_kg_m3);
        let p_vec = [v_eig[0][col], v_eig[1][col], v_eig[2][col]];
        eigenvalues_pa[m] = lam;
        phase_velocities_m_s[m] = vp;
        polarizations[m] = p_vec;

        let inv_rho_vp = 1.0 / (rho_kg_m3 * vp);
        for j in 0..3 {
            let mut vg_j = 0.0_f64;
            for i in 0..3 {
                let ij = voigt_index(i, j);
                for k in 0..3 {
                    for l in 0..3 {
                        vg_j += c_voigt[ij][voigt_index(k, l)] * p_vec[i] * p_vec[k] * n[l];
                    }
                }
            }
            group_velocities_m_s[m][j] = vg_j * inv_rho_vp;
        }
    }

    Ok(ChristoffelSolution {
        direction_unit: n,
        eigenvalues_pa,
        phase_velocities_m_s,
        polarizations,
        group_velocities_m_s,
    })
}

/// Evaluate the UTD Fresnel transition function $F(X) = 2 i \sqrt{X} e^{i X} \int_{\sqrt{X}}^\infty e^{-i \tau^2}\,d\tau$ for $X \ge 0$.
pub fn utd_fresnel_transition_f(x: f64) -> Result<Complex64, SignalError> {
    if !x.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if x < 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    if x == 0.0 {
        return Ok(Complex64::new(0.0, 0.0));
    }
    if x >= 8.0 {
        let inv_2x = 0.5 / x;
        let inv_2x2 = inv_2x * inv_2x;
        return Ok(Complex64::new(
            1.0 - 3.0 * inv_2x2 + 105.0 * inv_2x2 * inv_2x2,
            inv_2x * (1.0 - 15.0 * inv_2x2),
        ));
    }
    let sqrt_x = sqrt_f64(x);
    let h = sqrt_x / 64.0;
    let (mut int_re, mut int_im) = (0.0_f64, 0.0_f64);
    for k in 0..64 {
        let t_mid = ((k as f64) + 0.5) * h;
        let (s, c) = sin_cos_f64(t_mid * t_mid);
        int_re += c * h;
        int_im -= s * h;
    }
    let tail_sqrt_pi_half = 0.5 * sqrt_f64(PI);
    let rem = Complex64::new(
        tail_sqrt_pi_half * cos_f64(FRAC_PI_4) - int_re,
        -tail_sqrt_pi_half * sin_f64(FRAC_PI_4) - int_im,
    );
    let (sx, cx) = sin_cos_f64(x);
    Ok(Complex64::new(-2.0 * sqrt_x * sx, 2.0 * sqrt_x * cx).mul(rem))
}

fn utd_cot_fresnel_term(
    n_wedge: f64,
    kl: f64,
    beta: f64,
    sign_pi: f64,
) -> Result<Complex64, SignalError> {
    let n_int = round_symmetric_f64((beta + sign_pi * PI) / (2.0 * n_wedge * PI));
    let delta = 2.0 * n_wedge * PI * n_int - beta;
    let cos_half = cos_f64(0.5 * delta);
    let x = kl * 2.0 * cos_half * cos_half;
    let (s_arg, c_arg) = sin_cos_f64((PI + sign_pi * beta) / (2.0 * n_wedge));
    if s_arg.abs() < 1e-5 {
        let eps = delta - sign_pi * PI;
        let sgn = if eps >= 0.0 { 1.0 } else { -1.0 };
        let root_term = -n_wedge * sqrt_f64(2.0 * PI * kl) * sgn;
        return Ok(Complex64::new(
            root_term * cos_f64(FRAC_PI_4) - 2.0 * kl * n_wedge * n_wedge * eps,
            root_term * sin_f64(FRAC_PI_4),
        ));
    }
    Ok(utd_fresnel_transition_f(x)?.scale(c_arg / s_arg))
}

/// Evaluate the Kouyoumjian-Pathak UTD wedge diffraction coefficient $D(\phi, \phi'; k, L)$
/// across shadow ($\phi - \phi' = \pi$) and reflection ($\phi + \phi' = \pi$) boundaries.
#[allow(clippy::too_many_arguments)]
pub fn compute_utd_wedge_diffraction(
    n_wedge: f64,
    k_wavenumber_m1: f64,
    r_source_m: f64,
    r_receiver_m: f64,
    phi_incident_rad: f64,
    phi_observer_rad: f64,
    refl_face_0: f64,
    refl_face_n: f64,
) -> Result<Complex64, SignalError> {
    for v in [
        n_wedge,
        k_wavenumber_m1,
        r_source_m,
        r_receiver_m,
        phi_incident_rad,
        phi_observer_rad,
        refl_face_0,
        refl_face_n,
    ] {
        if !v.is_finite() {
            return Err(SignalError::NonFiniteInput);
        }
    }
    if !(1.0 < n_wedge && n_wedge <= 2.0)
        || k_wavenumber_m1 <= 0.0
        || r_source_m <= 0.0
        || r_receiver_m <= 0.0
        || phi_incident_rad <= 0.0
        || phi_incident_rad >= n_wedge * PI
        || phi_observer_rad <= 0.0
        || phi_observer_rad >= n_wedge * PI
    {
        return Err(SignalError::InvalidParameter);
    }
    let kl = k_wavenumber_m1 * (r_source_m * r_receiver_m) / (r_source_m + r_receiver_m);
    let (beta_minus, beta_plus) = (
        phi_observer_rad - phi_incident_rad,
        phi_observer_rad + phi_incident_rad,
    );
    let d1 = utd_cot_fresnel_term(n_wedge, kl, beta_minus, 1.0)?;
    let d2 = utd_cot_fresnel_term(n_wedge, kl, beta_minus, -1.0)?;
    let d3 = utd_cot_fresnel_term(n_wedge, kl, beta_plus, 1.0)?;
    let d4 = utd_cot_fresnel_term(n_wedge, kl, beta_plus, -1.0)?;
    let sum = d1
        .add(d2)
        .add(d3.scale(refl_face_n))
        .add(d4.scale(refl_face_0));
    let scale = -1.0 / (2.0 * n_wedge * sqrt_f64(2.0 * PI * k_wavenumber_m1));
    Ok(Complex64::new(scale * cos_f64(FRAC_PI_4), scale * sin_f64(FRAC_PI_4)).mul(sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biot_gassmann_low_vs_high_frequency_and_christoffel_and_utd() {
        let p = BiotPoroelasticParams {
            porosity: 0.2,
            rho_fluid_kg_m3: 1000.0,
            rho_solid_kg_m3: 2650.0,
            tortuosity_inf: 2.0,
            permeability_m2: 1e-12,
            viscosity_pa_s: 1e-3,
            biot_willis_alpha: 0.8,
            biot_modulus_m_pa: 1.2e10,
            frame_bulk_modulus_pa: 1.5e10,
            frame_shear_modulus_pa: 1.0e10,
        };
        let low = solve_biot_poroelastic_dispersion(&p, 10.0).unwrap();
        let high = solve_biot_poroelastic_dispersion(&p, 1e7).unwrap();
        let gassmann_p1 = sqrt_f64(p.undrained_p_modulus_h_pa() / p.bulk_density_kg_m3());
        assert!((low.c_p1_m_s - gassmann_p1).abs() / gassmann_p1 < 1e-3);
        assert!(
            high.c_p1_m_s > low.c_p1_m_s
                && high.c_p2_m_s > low.c_p2_m_s * 10.0
                && high.c_s_m_s > low.c_s_m_s
        );

        let mut c_iso = [[0.0_f64; 6]; 6];
        let (lam, mu) = (2.0e10, 1.0e10);
        for i in 0..3 {
            for j in 0..3 {
                c_iso[i][j] = if i == j { lam + 2.0 * mu } else { lam };
            }
            c_iso[i + 3][i + 3] = mu;
        }
        let ch = solve_christoffel_anisotropic_21(&c_iso, 2500.0, [1.0, 2.0, 3.0]).unwrap();
        assert!((ch.phase_velocities_m_s[0] - sqrt_f64((lam + 2.0 * mu) / 2500.0)).abs() < 1e-6);
        assert!((ch.phase_velocities_m_s[1] - sqrt_f64(mu / 2500.0)).abs() < 1e-6);
        let vg_dot_n = ch.group_velocities_m_s[0][0] * ch.direction_unit[0]
            + ch.group_velocities_m_s[0][1] * ch.direction_unit[1]
            + ch.group_velocities_m_s[0][2] * ch.direction_unit[2];
        assert!((vg_dot_n - ch.phase_velocities_m_s[0]).abs() < 1e-6);

        let d_sb =
            compute_utd_wedge_diffraction(1.5, 20.0, 2.0, 2.0, 0.5, 0.5 + PI, 1.0, 1.0).unwrap();
        let d_near =
            compute_utd_wedge_diffraction(1.5, 20.0, 2.0, 2.0, 0.5, 0.5 + PI - 1e-6, 1.0, 1.0)
                .unwrap();
        assert!(d_sb.is_finite() && d_near.is_finite() && d_sb.sub(d_near).abs() < 1e-2);
    }
}
