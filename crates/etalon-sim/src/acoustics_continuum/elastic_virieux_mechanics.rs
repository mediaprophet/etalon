//! 3D Elastic Continuum Mechanics: Constitutive Medium, Tensors, Quaternions & Interfaces.

use super::super::fdtd_fluid::NumericalError;

/// Isotropic linear elastic solid (or explicit $\mu=0$ fluid-limit) medium parameters (Schema 200).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticMedium {
    /// Mass density $\rho > 0$ in $\text{kg}\cdot\text{m}^{-3}$.
    pub density_kg_m3: f64,
    /// Bulk modulus $K > 0$ in $\text{Pa}$.
    pub bulk_modulus_pa: f64,
    /// Shear modulus $\mu \ge 0$ in $\text{Pa}$ ($\mu > 0$ for solid; $\mu = 0$ fluid limit).
    pub shear_modulus_pa: f64,
    /// Material reference ID.
    pub material_ref: u32,
}

impl ElasticMedium {
    /// Construct and validate an isotropic linear elastic solid ($\mu > 0$).
    pub fn new_solid(
        density_kg_m3: f64,
        bulk_modulus_pa: f64,
        shear_modulus_pa: f64,
        material_ref: u32,
    ) -> Result<Self, NumericalError> {
        if !density_kg_m3.is_finite() || density_kg_m3 <= 0.0 {
            return Err(NumericalError::InvalidMedium(
                "elastic density_kg_m3 must be finite and > 0 (vacuum is a boundary mask, not rho=0)",
            ));
        }
        if !bulk_modulus_pa.is_finite() || bulk_modulus_pa <= 0.0 {
            return Err(NumericalError::InvalidMedium("elastic bulk_modulus_pa must be finite and > 0"));
        }
        if !shear_modulus_pa.is_finite() || shear_modulus_pa <= 0.0 {
            return Err(NumericalError::InvalidMedium(
                "elastic solid requires finite shear_modulus_pa > 0 (use new_fluid_limit for mu=0)",
            ));
        }
        Ok(Self { density_kg_m3, bulk_modulus_pa, shear_modulus_pa, material_ref })
    }

    /// Construct the explicitly admitted $\mu = 0$ fluid limit of the elastic equations (C02-C06).
    pub fn new_fluid_limit(
        density_kg_m3: f64,
        bulk_modulus_pa: f64,
        material_ref: u32,
    ) -> Result<Self, NumericalError> {
        if !density_kg_m3.is_finite() || density_kg_m3 <= 0.0 {
            return Err(NumericalError::InvalidMedium("fluid-limit density_kg_m3 must be finite and > 0"));
        }
        if !bulk_modulus_pa.is_finite() || bulk_modulus_pa <= 0.0 {
            return Err(NumericalError::InvalidMedium("fluid-limit bulk_modulus_pa must be finite and > 0"));
        }
        Ok(Self { density_kg_m3, bulk_modulus_pa, shear_modulus_pa: 0.0, material_ref })
    }

    /// First Lamé parameter $\lambda = K - \frac{2}{3}\mu$ in $\text{Pa}$.
    /// Note (C02-C06): negative $\lambda$ is physically admissible when $K > 0$ and $\mu > 0$.
    #[inline]
    pub fn lame_lambda_pa(&self) -> f64 {
        self.bulk_modulus_pa - (2.0 / 3.0) * self.shear_modulus_pa
    }

    /// Longitudinal / P-wave modulus $M = \lambda + 2\mu = K + \frac{4}{3}\mu$ in $\text{Pa}$.
    #[inline]
    pub fn p_wave_modulus_pa(&self) -> f64 {
        self.bulk_modulus_pa + (4.0 / 3.0) * self.shear_modulus_pa
    }

    /// Compressional (P-wave) phase speed $c_P = \sqrt{(K + \frac{4}{3}\mu)/\rho}$ in $\text{m}\cdot\text{s}^{-1}$.
    #[inline]
    pub fn p_wave_speed_m_s(&self) -> f64 {
        (self.p_wave_modulus_pa() / self.density_kg_m3).sqrt()
    }

    /// Shear (S-wave) phase speed $c_S = \sqrt{\mu/\rho}$ in $\text{m}\cdot\text{s}^{-1}$.
    #[inline]
    pub fn s_wave_speed_m_s(&self) -> f64 {
        (self.shear_modulus_pa / self.density_kg_m3).sqrt()
    }

    /// Longitudinal acoustic impedance $Z_P = \rho c_P = \sqrt{\rho(K + \frac{4}{3}\mu)}$ in Rayl.
    #[inline]
    pub fn p_wave_impedance_rayl(&self) -> f64 {
        (self.density_kg_m3 * self.p_wave_modulus_pa()).sqrt()
    }

    /// Shear acoustic impedance $Z_S = \rho c_S = \sqrt{\rho \mu}$ in Rayl.
    #[inline]
    pub fn s_wave_impedance_rayl(&self) -> f64 {
        (self.density_kg_m3 * self.shear_modulus_pa).sqrt()
    }

    /// Evaluate isotropic Hooke's law $\boldsymbol{\sigma} = \lambda\operatorname{tr}(\boldsymbol{\varepsilon})\mathbf{I} + 2\mu\boldsymbol{\varepsilon}$
    /// from tensor small strain $(\varepsilon_{xx},\varepsilon_{yy},\varepsilon_{zz},\varepsilon_{xy},\varepsilon_{yz},\varepsilon_{xz})$.
    #[inline]
    pub fn stress_from_tensor_strain(&self, strain: &TensorStrain6) -> CauchyStress6 {
        let lambda = self.lame_lambda_pa();
        let two_mu = 2.0 * self.shear_modulus_pa;
        let tr = strain.trace();
        let [exx, eyy, ezz, exy, eyz, exz] = strain.components;
        CauchyStress6 {
            components: [
                lambda * tr + two_mu * exx,
                lambda * tr + two_mu * eyy,
                lambda * tr + two_mu * ezz,
                two_mu * exy,
                two_mu * eyz,
                two_mu * exz,
            ],
        }
    }

    /// Elastic strain + kinetic energy density
    /// $e = \frac{1}{2}\rho|\mathbf{v}|^2 + \frac{1}{2}\boldsymbol{\sigma}:\boldsymbol{\varepsilon}$ in $\text{J}\cdot\text{m}^{-3}$.
    #[inline]
    pub fn energy_density_j_m3(
        &self,
        stress: &CauchyStress6,
        strain: &TensorStrain6,
        v: [f64; 3],
    ) -> f64 {
        let v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let [sxx, syy, szz, sxy, syz, sxz] = stress.components;
        let [exx, eyy, ezz, exy, eyz, exz] = strain.components;
        let u_strain = 0.5 * (sxx * exx + syy * eyy + szz * ezz + 2.0 * (sxy * exy + syz * eyz + sxz * exz));
        0.5 * self.density_kg_m3 * v_sq + u_strain
    }
}

/// Six-component symmetric small-strain tensor in canonical QAC order:
/// `[\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \varepsilon_{xy}, \varepsilon_{yz}, \varepsilon_{xz}]`.
///
/// Off-diagonal components are true tensor shear strains $\varepsilon_{ij} = \frac{1}{2}(\partial_i u_j + \partial_j u_i)$,
/// NOT engineering shear strains $\gamma_{ij} = 2\varepsilon_{ij}$ (C02-C04).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorStrain6 {
    /// Components `[xx, yy, zz, xy, yz, xz]`.
    pub components: [f64; 6],
}

impl TensorStrain6 {
    /// Construct from explicit tensor strain components `[xx, yy, zz, xy, yz, xz]`.
    #[inline]
    pub const fn new(components: [f64; 6]) -> Self {
        Self { components }
    }

    /// Construct from engineering Voigt strain `[exx, eyy, ezz, gamma_xy, gamma_yz, gamma_xz]`
    /// by halving the three engineering shear components ($\varepsilon_{ij} = \frac{1}{2}\gamma_{ij}$).
    #[inline]
    pub fn from_engineering_voigt(v: [f64; 6]) -> Self {
        Self { components: [v[0], v[1], v[2], 0.5 * v[3], 0.5 * v[4], 0.5 * v[5]] }
    }

    /// Convert to engineering Voigt strain `[exx, eyy, ezz, 2*exy, 2*eyz, 2*exz]`.
    #[inline]
    pub fn to_engineering_voigt(&self) -> [f64; 6] {
        let c = self.components;
        [c[0], c[1], c[2], 2.0 * c[3], 2.0 * c[4], 2.0 * c[5]]
    }

    /// Volumetric dilatation $\operatorname{tr}(\boldsymbol{\varepsilon}) = \varepsilon_{xx} + \varepsilon_{yy} + \varepsilon_{zz}$.
    #[inline]
    pub fn trace(&self) -> f64 {
        self.components[0] + self.components[1] + self.components[2]
    }
}

/// Six-component symmetric Cauchy stress tensor in canonical QAC order:
/// `[\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{yz}, \sigma_{xz}]` in Pascals (tensile-positive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CauchyStress6 {
    /// Components `[xx, yy, zz, xy, yz, xz]` in Pa.
    pub components: [f64; 6],
}

impl CauchyStress6 {
    /// Construct from `[xx, yy, zz, xy, yz, xz]` in Pa.
    #[inline]
    pub const fn new(components: [f64; 6]) -> Self {
        Self { components }
    }

    /// Stress trace $\operatorname{tr}(\boldsymbol{\sigma}) = \sigma_{xx} + \sigma_{yy} + \sigma_{zz}$ in Pa.
    #[inline]
    pub fn trace(&self) -> f64 {
        self.components[0] + self.components[1] + self.components[2]
    }

    /// Mechanical compressive pressure $p = -\frac{1}{3}\operatorname{tr}(\boldsymbol{\sigma})$ in Pa (C02-C04).
    #[inline]
    pub fn mechanical_pressure_pa(&self) -> f64 {
        -self.trace() / 3.0
    }

    /// Convert to a full $3\times 3$ symmetric matrix.
    #[inline]
    pub const fn to_matrix3(&self) -> [[f64; 3]; 3] {
        let [sxx, syy, szz, sxy, syz, sxz] = self.components;
        [[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]]
    }

    /// Construct from a symmetric $3\times 3$ matrix.
    #[inline]
    pub fn from_symmetric_matrix3(m: [[f64; 3]; 3]) -> Self {
        Self {
            components: [
                m[0][0],
                m[1][1],
                m[2][2],
                0.5 * (m[0][1] + m[1][0]),
                0.5 * (m[1][2] + m[2][1]),
                0.5 * (m[0][2] + m[2][0]),
            ],
        }
    }

    /// Surface traction vector $\mathbf{t} = \boldsymbol{\sigma}\cdot\mathbf{n}$ on a boundary with unit normal $\mathbf{n}$.
    #[inline]
    pub fn traction_vector(&self, n: [f64; 3]) -> [f64; 3] {
        let m = self.to_matrix3();
        [
            m[0][0] * n[0] + m[0][1] * n[1] + m[0][2] * n[2],
            m[1][0] * n[0] + m[1][1] * n[1] + m[1][2] * n[2],
            m[2][0] * n[0] + m[2][1] * n[1] + m[2][2] * n[2],
        ]
    }

    /// Enforce free-surface vacuum boundary ($\boldsymbol{\sigma}\cdot\mathbf{n} = \mathbf{0}$) along an axis-aligned normal
    /// (`axis = 0` for $x$, `1` for $y$, `2` for $z$).
    #[inline]
    pub fn project_free_surface_axis(&self, axis: usize) -> Self {
        self.project_fluid_interface_axis(axis, 0.0)
    }

    /// Enforce fluid-solid interface traction condition for a fluid with acoustic pressure `p_fluid_pa`
    /// along an axis-aligned normal (`axis = 0, 1, 2`):
    /// normal stress $\sigma_{nn} = -p_{\text{fluid}}$, tangential shear stresses $\sigma_{nt} = 0$.
    #[inline]
    pub fn project_fluid_interface_axis(&self, axis: usize, p_fluid_pa: f64) -> Self {
        let [mut sxx, mut syy, mut szz, mut sxy, mut syz, mut sxz] = self.components;
        match axis {
            0 => {
                sxx = -p_fluid_pa;
                sxy = 0.0;
                sxz = 0.0;
            }
            1 => {
                syy = -p_fluid_pa;
                sxy = 0.0;
                syz = 0.0;
            }
            _ => {
                szz = -p_fluid_pa;
                syz = 0.0;
                sxz = 0.0;
            }
        }
        Self { components: [sxx, syy, szz, sxy, syz, sxz] }
    }

    /// Principal invariants $(I_1, I_2, I_3)$ of the Cauchy stress tensor, which remain invariant
    /// under any proper orthogonal rotation $\mathbf{R}\boldsymbol{\sigma}\mathbf{R}^T$.
    #[inline]
    pub fn principal_invariants(&self) -> (f64, f64, f64) {
        let [sxx, syy, szz, sxy, syz, sxz] = self.components;
        let i1 = sxx + syy + szz;
        let i2 = (sxx * syy + syy * szz + szz * sxx) - (sxy * sxy + syz * syz + sxz * sxz);
        let i3 = sxx * (syy * szz - syz * syz)
            - sxy * (sxy * szz - syz * sxz)
            + sxz * (sxy * syz - syy * sxz);
        (i1, i2, i3)
    }
}

/// Active local-to-parent Hamilton quaternion `[x, y, z, w]` (C02-C04 / CROSS-A01).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HamiltonQuaternionXyzw {
    /// Raw authored quaternion components `[x, y, z, w]`.
    pub raw_xyzw: [f64; 4],
    /// Derived normalized quaternion `[x/n, y/n, z/n, w/n]`.
    derived_unit_xyzw: [f64; 4],
    /// Proper orthogonal $3\times 3$ active rotation matrix $\mathbf{R}$.
    rotation_matrix: [[f64; 3]; 3],
}

impl HamiltonQuaternionXyzw {
    /// Ordered squared-norm sum `(((x*x + y*y) + z*z) + w*w)` mandated by C02-C04.
    #[inline]
    pub fn ordered_norm_squared(xyzw: [f64; 4]) -> f64 {
        let [x, y, z, w] = xyzw;
        let xy = x * x + y * y;
        let xyz = xy + z * z;
        xyz + w * w
    }

    /// Admit a raw `[x, y, z, w]` quaternion and derive its orthogonal rotation matrix.
    pub fn admit(raw_xyzw: [f64; 4]) -> Result<Self, NumericalError> {
        if raw_xyzw.iter().any(|v| !v.is_finite()) {
            return Err(NumericalError::InvalidRotation("quaternion contains non-finite component"));
        }
        let n2 = Self::ordered_norm_squared(raw_xyzw);
        if (n2 - 1.0).abs() > 1e-12 {
            return Err(NumericalError::InvalidRotation(
                "quaternion squared norm deviation exceeds 1e-12 admission tolerance",
            ));
        }
        let n = n2.sqrt();
        let [x, y, z, w] = [raw_xyzw[0] / n, raw_xyzw[1] / n, raw_xyzw[2] / n, raw_xyzw[3] / n];
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        let r = [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ];
        Ok(Self { raw_xyzw, derived_unit_xyzw: [x, y, z, w], rotation_matrix: r })
    }

    /// Derived unit quaternion `[x, y, z, w]`.
    #[inline]
    pub const fn derived_unit_xyzw(&self) -> [f64; 4] {
        self.derived_unit_xyzw
    }

    /// Active $3\times 3$ rotation matrix $\mathbf{R}$.
    #[inline]
    pub const fn rotation_matrix(&self) -> [[f64; 3]; 3] {
        self.rotation_matrix
    }

    /// Maximum entrywise residual $\|\mathbf{R}\mathbf{R}^T - \mathbf{I}\|_\infty$.
    pub fn orthogonality_residual(&self) -> f64 {
        let r = self.rotation_matrix;
        let mut max_err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                let dot = r[i][0] * r[j][0] + r[i][1] * r[j][1] + r[i][2] * r[j][2];
                let target = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((dot - target).abs());
            }
        }
        max_err
    }

    /// Active vector rotation $\mathbf{v}_{\text{parent}} = \mathbf{R}\mathbf{v}_{\text{local}}$.
    #[inline]
    pub fn rotate_vector(&self, v: [f64; 3]) -> [f64; 3] {
        let r = self.rotation_matrix;
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    }

    /// Active Cauchy stress rotation $\boldsymbol{\sigma}' = \mathbf{R}\boldsymbol{\sigma}\mathbf{R}^T$.
    pub fn rotate_stress(&self, stress: &CauchyStress6) -> CauchyStress6 {
        let r = self.rotation_matrix;
        let s = stress.to_matrix3();
        let mut rs = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rs[i][j] = r[i][0] * s[0][j] + r[i][1] * s[1][j] + r[i][2] * s[2][j];
            }
        }
        let mut out = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = rs[i][0] * r[j][0] + rs[i][1] * r[j][1] + rs[i][2] * r[j][2];
            }
        }
        CauchyStress6::from_symmetric_matrix3(out)
    }

    /// Active Hamilton composition `q_parent * q_local` (apply `local` then `parent`).
    pub fn compose(&self, local: &Self) -> Result<Self, NumericalError> {
        let [x1, y1, z1, w1] = self.derived_unit_xyzw;
        let [x2, y2, z2, w2] = local.derived_unit_xyzw;
        Self::admit([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])
    }
}

/// Normal-incidence planar acoustic / longitudinal impedance reflection and transmission
/// across an interface from medium 1 ($Z_1 > 0$) into medium 2 ($Z_2 \ge 0$).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalIncidenceInterface {
    /// Pressure amplitude reflection coefficient $r_p = \frac{Z_2 - Z_1}{Z_2 + Z_1}$.
    pub pressure_reflection: f64,
    /// Pressure amplitude transmission coefficient $t_p = \frac{2 Z_2}{Z_2 + Z_1}$.
    pub pressure_transmission: f64,
    /// Particle-velocity amplitude reflection coefficient $r_v = \frac{Z_1 - Z_2}{Z_1 + Z_2} = -r_p$.
    pub velocity_reflection: f64,
    /// Particle-velocity amplitude transmission coefficient $t_v = \frac{2 Z_1}{Z_1 + Z_2}$.
    pub velocity_transmission: f64,
    /// Power / energy reflectance $R_E = r_p^2$.
    pub energy_reflectance: f64,
    /// Power / energy transmittance $T_E = \frac{Z_1}{Z_2} t_p^2 = \frac{4 Z_1 Z_2}{(Z_1 + Z_2)^2}$.
    pub energy_transmittance: f64,
}

impl NormalIncidenceInterface {
    /// Compute normal-incidence reflection and transmission from impedances $Z_1 > 0$ and $Z_2 > 0$.
    pub fn from_impedances(z1_rayl: f64, z2_rayl: f64) -> Result<Self, NumericalError> {
        if !z1_rayl.is_finite() || z1_rayl <= 0.0 || !z2_rayl.is_finite() || z2_rayl <= 0.0 {
            return Err(NumericalError::InvalidMedium(
                "interface impedances Z1 and Z2 must be finite and > 0",
            ));
        }
        let sum = z1_rayl + z2_rayl;
        let r_p = (z2_rayl - z1_rayl) / sum;
        let t_p = (2.0 * z2_rayl) / sum;
        Ok(Self {
            pressure_reflection: r_p,
            pressure_transmission: t_p,
            velocity_reflection: -r_p,
            velocity_transmission: (2.0 * z1_rayl) / sum,
            energy_reflectance: r_p * r_p,
            energy_transmittance: (4.0 * z1_rayl * z2_rayl) / (sum * sum),
        })
    }

    /// Ideal rigid (sound-hard) boundary limit ($Z_2 \to \infty$): $r_p = +1, t_p = 2, r_v = -1, t_v = 0, R_E = 1, T_E = 0$.
    pub const fn rigid_limit() -> Self {
        Self {
            pressure_reflection: 1.0,
            pressure_transmission: 2.0,
            velocity_reflection: -1.0,
            velocity_transmission: 0.0,
            energy_reflectance: 1.0,
            energy_transmittance: 0.0,
        }
    }

    /// Ideal pressure-release (free-surface / vacuum) boundary limit ($Z_2 = 0$):
    /// $r_p = -1, t_p = 0, r_v = +1, t_v = 2, R_E = 1, T_E = 0$.
    pub const fn pressure_release_limit() -> Self {
        Self {
            pressure_reflection: -1.0,
            pressure_transmission: 0.0,
            velocity_reflection: 1.0,
            velocity_transmission: 2.0,
            energy_reflectance: 1.0,
            energy_transmittance: 0.0,
        }
    }
}
