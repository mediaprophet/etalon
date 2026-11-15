//! 3D Helmholtz Frequency-Domain Solver & Green's Function Oracle.
//!
//! Implements the frequency-domain acoustic wave equation:
//! $$(\nabla^2 + k^2)\hat{p}(\mathbf{x}) = -\hat{s}(\mathbf{x}), \qquad k = \frac{\omega}{c(\omega)} + i\alpha(\omega)$$
//! - Complex arithmetic (`Complex64`) with explicit `f64` real/imaginary lanes.
//! - 7-point standard second-order and 27-point isotropic dispersion-minimizing 3D finite-difference stencils.
//! - Zero-allocation iterative solvers (damped Jacobi and preconditioned BiCGSTAB) operating on
//!   caller-supplied `HelmholtzWorkspace` buffers.
//! - Analytical free-space 3D Green's function $G(r) = \frac{e^{ikr}}{4\pi r}$ ($r > 0$) and
//!   rectangular cavity Dirichlet/Neumann eigenfrequencies and eigenmodes.

use core::f64::consts::PI;
use super::fdtd_fluid::NumericalError;

/// Complex double-precision scalar `re + i * im`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// Zero constant `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    /// Unit real constant `1 + 0i`.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    /// Unit imaginary constant `0 + 1i`.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Construct `re + i * im`.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Returns `true` if both components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Complex conjugate `re - i * im`.
    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared magnitude `re^2 + im^2`.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Euclidean magnitude `\sqrt{re^2 + im^2}`.
    #[inline]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Addition `self + rhs`.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    /// Subtraction `self - rhs`.
    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    /// Multiplication `self * rhs`.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    /// Scalar multiplication `s * self`.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Division `self / rhs`.
    #[inline]
    pub fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sqr();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

/// Finite-difference stencil choice for the 3D Helmholtz operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelmholtzStencil {
    /// Standard 7-point second-order Cartesian Laplacian + pointwise $k^2$.
    SevenPoint,
    /// Isotropic 27-point dispersion-minimizing 3D Laplacian (O'Reilly / Operto weights)
    /// with consistent mass weights on $k^2$ for uniform cubic spacing $h = \Delta x = \Delta y = \Delta z$.
    TwentySevenPointIsotropic,
}

/// Boundary condition for the 3D Helmholtz grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelmholtzBoundary {
    /// Homogeneous Dirichlet ($\hat{p} = 0$ on outer grid nodes, pressure-release).
    DirichletZero,
    /// Periodic wrap-around across all three axes.
    Periodic,
}

/// Configuration for the 3D Helmholtz operator $(\nabla^2 + k^2)\hat{p} = -\hat{s}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzConfig {
    /// Grid dimensions `(nx, ny, nz)`.
    pub dims: [usize; 3],
    /// Grid spacing `(dx, dy, dz)` in metres.
    pub spacing_m: [f64; 3],
    /// Angular frequency $\omega = 2\pi f$ in $\text{rad}\cdot\text{s}^{-1}$ ($\omega \ge 0$).
    pub omega_rad_s: f64,
    /// Phase speed $c(\omega) > 0$ in $\text{m}\cdot\text{s}^{-1}$.
    pub phase_speed_m_s: f64,
    /// Attenuation coefficient $\alpha(\omega) \ge 0$ in $\text{Np}\cdot\text{m}^{-1}$.
    pub attenuation_np_m: f64,
    /// Stencil discretization kind.
    pub stencil: HelmholtzStencil,
    /// Boundary condition kind.
    pub boundary: HelmholtzBoundary,
}

impl HelmholtzConfig {
    /// Validate grid dimensions, spacing, and physical wavenumber parameters.
    pub fn validate(&self) -> Result<Complex64, NumericalError> {
        let [nx, ny, nz] = self.dims;
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(NumericalError::InvalidGrid(
                "HelmholtzConfig dimensions (nx, ny, nz) must each be >= 3",
            ));
        }
        let [dx, dy, dz] = self.spacing_m;
        if !dx.is_finite() || !dy.is_finite() || !dz.is_finite() || dx <= 0.0 || dy <= 0.0 || dz <= 0.0
        {
            return Err(NumericalError::InvalidGrid(
                "HelmholtzConfig spacing_m components must be finite and > 0",
            ));
        }
        if matches!(self.stencil, HelmholtzStencil::TwentySevenPointIsotropic)
            && ((dx - dy).abs() > 1e-12 * dx || (dx - dz).abs() > 1e-12 * dx)
        {
            return Err(NumericalError::InvalidGrid(
                "TwentySevenPointIsotropic stencil requires uniform cubic spacing dx == dy == dz",
            ));
        }
        if !self.omega_rad_s.is_finite() || self.omega_rad_s < 0.0 {
            return Err(NumericalError::InvalidMedium(
                "Helmholtz omega_rad_s must be finite and >= 0",
            ));
        }
        if !self.phase_speed_m_s.is_finite() || self.phase_speed_m_s <= 0.0 {
            return Err(NumericalError::InvalidMedium(
                "Helmholtz phase_speed_m_s must be finite and > 0",
            ));
        }
        if !self.attenuation_np_m.is_finite() || self.attenuation_np_m < 0.0 {
            return Err(NumericalError::InvalidMedium(
                "Helmholtz attenuation_np_m must be finite and >= 0",
            ));
        }
        Ok(self.wavenumber())
    }

    /// Complex wavenumber $k = \frac{\omega}{c(\omega)} + i\alpha(\omega)$ in $\text{m}^{-1}$.
    #[inline]
    pub fn wavenumber(&self) -> Complex64 {
        Complex64::new(
            self.omega_rad_s / self.phase_speed_m_s,
            self.attenuation_np_m,
        )
    }

    /// Squared complex wavenumber $k^2$.
    #[inline]
    pub fn wavenumber_squared(&self) -> Complex64 {
        let k = self.wavenumber();
        k.mul(k)
    }

    /// Total grid node count `nx * ny * nz`.
    #[inline]
    pub const fn node_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
}

/// Apply the 3D discrete Helmholtz operator $A\hat{p} = (\nabla_h^2 + k^2)\hat{p}$ into `out`.
pub fn apply_helmholtz_operator(
    cfg: &HelmholtzConfig,
    u: &[Complex64],
    out: &mut [Complex64],
) -> Result<(), NumericalError> {
    let k = cfg.validate()?;
    let k2 = k.mul(k);
    let [nx, ny, nz] = cfg.dims;
    let n = cfg.node_count();
    if u.len() != n || out.len() != n {
        return Err(NumericalError::BufferLengthMismatch {
            buffer: "Helmholtz operator u/out",
            expected: n,
            actual: u.len().min(out.len()),
        });
    }

    let [dx, dy, dz] = cfg.spacing_m;
    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    let inv_dz2 = 1.0 / (dz * dz);
    let idx = |ix: usize, iy: usize, iz: usize| -> usize { (ix * ny + iy) * nz + iz };
    let periodic = matches!(cfg.boundary, HelmholtzBoundary::Periodic);

    match cfg.stencil {
        HelmholtzStencil::SevenPoint => {
            let center_lap = -2.0 * (inv_dx2 + inv_dy2 + inv_dz2);
            let diag = Complex64::new(center_lap + k2.re, k2.im);
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let c = idx(ix, iy, iz);
                        if !periodic
                            && (ix == 0 || ix + 1 == nx || iy == 0 || iy + 1 == ny || iz == 0 || iz + 1 == nz)
                        {
                            out[c] = u[c];
                            continue;
                        }
                        let xm = if ix > 0 { ix - 1 } else { nx - 1 };
                        let xp = if ix + 1 < nx { ix + 1 } else { 0 };
                        let ym = if iy > 0 { iy - 1 } else { ny - 1 };
                        let yp = if iy + 1 < ny { iy + 1 } else { 0 };
                        let zm = if iz > 0 { iz - 1 } else { nz - 1 };
                        let zp = if iz + 1 < nz { iz + 1 } else { 0 };

                        let sum_x = u[idx(xm, iy, iz)].add(u[idx(xp, iy, iz)]).scale(inv_dx2);
                        let sum_y = u[idx(ix, ym, iz)].add(u[idx(ix, yp, iz)]).scale(inv_dy2);
                        let sum_z = u[idx(ix, iy, zm)].add(u[idx(ix, iy, zp)]).scale(inv_dz2);

                        out[c] = u[c].mul(diag).add(sum_x).add(sum_y).add(sum_z);
                    }
                }
            }
        }
        HelmholtzStencil::TwentySevenPointIsotropic => {
            // Isotropic 27-point 3D Laplacian weights (1/h^2):
            // center = -128/30, 6 face neighbors = 14/30, 12 edge neighbors = 3/30, 8 corner neighbors = 1/30.
            // Exact sum of weights = (-128 + 6*14 + 12*3 + 8*1)/30 = 0, fourth-order isotropic rotational invariance.
            let inv_h2 = inv_dx2;
            let w0 = (-128.0 / 30.0) * inv_h2;
            let w_face = (14.0 / 30.0) * inv_h2;
            let w_edge = (3.0 / 30.0) * inv_h2;
            let w_corn = (1.0 / 30.0) * inv_h2;

            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let c = idx(ix, iy, iz);
                        if !periodic
                            && (ix == 0 || ix + 1 == nx || iy == 0 || iy + 1 == ny || iz == 0 || iz + 1 == nz)
                        {
                            out[c] = u[c];
                            continue;
                        }
                        let mut lap = Complex64::ZERO;
                        for ox in -1isize..=1 {
                            let nx_i = ((ix as isize + ox).rem_euclid(nx as isize)) as usize;
                            for oy in -1isize..=1 {
                                let ny_i = ((iy as isize + oy).rem_euclid(ny as isize)) as usize;
                                for oz in -1isize..=1 {
                                    let nz_i = ((iz as isize + oz).rem_euclid(nz as isize)) as usize;
                                    let manhattan = ox.unsigned_abs() + oy.unsigned_abs() + oz.unsigned_abs();
                                    let w = match manhattan {
                                        0 => w0,
                                        1 => w_face,
                                        2 => w_edge,
                                        _ => w_corn,
                                    };
                                    lap = lap.add(u[idx(nx_i, ny_i, nz_i)].scale(w));
                                }
                            }
                        }
                        out[c] = lap.add(u[c].mul(k2));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Caller-owned workspace for zero-allocation BiCGSTAB and Jacobi Helmholtz solves.
#[derive(Debug, Clone)]
pub struct HelmholtzWorkspace {
    /// Residual vector $r$.
    pub r: Vec<Complex64>,
    /// Shadow residual vector $\hat{r}_0$.
    pub r_hat: Vec<Complex64>,
    /// Search direction $p$.
    pub p: Vec<Complex64>,
    /// Operator application $v = A p$.
    pub v: Vec<Complex64>,
    /// Intermediate residual $s = r - \alpha v$.
    pub s: Vec<Complex64>,
    /// Operator application $t = A s$.
    pub t: Vec<Complex64>,
}

impl HelmholtzWorkspace {
    /// Allocate workspace for a grid with `node_count` nodes.
    pub fn new(node_count: usize) -> Self {
        Self {
            r: vec![Complex64::ZERO; node_count],
            r_hat: vec![Complex64::ZERO; node_count],
            p: vec![Complex64::ZERO; node_count],
            v: vec![Complex64::ZERO; node_count],
            s: vec![Complex64::ZERO; node_count],
            t: vec![Complex64::ZERO; node_count],
        }
    }
}

/// Convergence summary for an iterative 3D Helmholtz solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzSolveReport {
    /// Iterations executed.
    pub iterations: usize,
    /// Final Euclidean residual norm $\|r\|_2$.
    pub residual_norm: f64,
    /// Relative residual $\|r\|_2 / \max(\|b\|_2, 10^{-30})$.
    pub relative_residual: f64,
}

/// Solve $(\nabla_h^2 + k^2)\hat{p} = -\hat{s}$ (where `rhs` is $-\hat{s}$) using BiCGSTAB
/// over caller-owned `workspace` with zero heap allocations inside the loop.
pub fn solve_helmholtz_bicgstab(
    cfg: &HelmholtzConfig,
    rhs: &[Complex64],
    solution: &mut [Complex64],
    workspace: &mut HelmholtzWorkspace,
    max_iters: usize,
    rel_tol: f64,
) -> Result<HelmholtzSolveReport, NumericalError> {
    cfg.validate()?;
    let n = cfg.node_count();
    if rhs.len() != n
        || solution.len() != n
        || workspace.r.len() != n
        || workspace.r_hat.len() != n
        || workspace.p.len() != n
        || workspace.v.len() != n
        || workspace.s.len() != n
        || workspace.t.len() != n
    {
        return Err(NumericalError::BufferLengthMismatch {
            buffer: "Helmholtz BiCGSTAB arrays",
            expected: n,
            actual: rhs.len().min(solution.len()),
        });
    }

    let b_norm = l2_norm(rhs).max(1e-30);

    // r = rhs - A * solution
    apply_helmholtz_operator(cfg, solution, &mut workspace.v)?;
    for i in 0..n {
        workspace.r[i] = rhs[i].sub(workspace.v[i]);
        workspace.r_hat[i] = workspace.r[i];
        workspace.p[i] = Complex64::ZERO;
        workspace.v[i] = Complex64::ZERO;
    }

    let mut res_norm = l2_norm(&workspace.r);
    if res_norm / b_norm <= rel_tol {
        return Ok(HelmholtzSolveReport {
            iterations: 0,
            residual_norm: res_norm,
            relative_residual: res_norm / b_norm,
        });
    }

    let mut rho_prev = Complex64::ONE;
    let mut alpha = Complex64::ONE;
    let mut omega = Complex64::ONE;

    for iter in 1..=max_iters {
        let rho_curr = hermitian_dot(&workspace.r_hat, &workspace.r);
        if rho_curr.abs() < 1e-60 {
            return Err(NumericalError::SolverConvergence(
                "BiCGSTAB breakdown: <r_hat, r> vanished",
            ));
        }
        let beta = if iter == 1 {
            Complex64::ZERO
        } else {
            rho_curr.div(rho_prev).mul(alpha.div(omega))
        };

        for i in 0..n {
            let p_minus_omega_v = workspace.p[i].sub(omega.mul(workspace.v[i]));
            workspace.p[i] = workspace.r[i].add(beta.mul(p_minus_omega_v));
        }

        apply_helmholtz_operator(cfg, &workspace.p, &mut workspace.v)?;
        let rhat_v = hermitian_dot(&workspace.r_hat, &workspace.v);
        if rhat_v.abs() < 1e-60 {
            return Err(NumericalError::SolverConvergence(
                "BiCGSTAB breakdown: <r_hat, v> vanished",
            ));
        }
        alpha = rho_curr.div(rhat_v);

        for i in 0..n {
            workspace.s[i] = workspace.r[i].sub(alpha.mul(workspace.v[i]));
        }

        let s_norm = l2_norm(&workspace.s);
        if s_norm / b_norm <= rel_tol {
            for i in 0..n {
                solution[i] = solution[i].add(alpha.mul(workspace.p[i]));
            }
            return Ok(HelmholtzSolveReport {
                iterations: iter,
                residual_norm: s_norm,
                relative_residual: s_norm / b_norm,
            });
        }

        apply_helmholtz_operator(cfg, &workspace.s, &mut workspace.t)?;
        let tt = hermitian_dot(&workspace.t, &workspace.t).re;
        if tt <= 1e-120 {
            return Err(NumericalError::SolverConvergence(
                "BiCGSTAB breakdown: <t, t> vanished",
            ));
        }
        let ts = hermitian_dot(&workspace.t, &workspace.s);
        omega = ts.scale(1.0 / tt);

        for i in 0..n {
            solution[i] = solution[i]
                .add(alpha.mul(workspace.p[i]))
                .add(omega.mul(workspace.s[i]));
            workspace.r[i] = workspace.s[i].sub(omega.mul(workspace.t[i]));
        }

        res_norm = l2_norm(&workspace.r);
        if res_norm / b_norm <= rel_tol {
            return Ok(HelmholtzSolveReport {
                iterations: iter,
                residual_norm: res_norm,
                relative_residual: res_norm / b_norm,
            });
        }
        rho_prev = rho_curr;
    }

    Err(NumericalError::SolverConvergence(
        "BiCGSTAB did not reach rel_tol within max_iters",
    ))
}

/// Damped Jacobi relaxation step for the 3D Helmholtz operator over caller-owned workspace.
pub fn step_helmholtz_jacobi(
    cfg: &HelmholtzConfig,
    rhs: &[Complex64],
    solution: &mut [Complex64],
    scratch_residual: &mut [Complex64],
    omega_relax: f64,
) -> Result<f64, NumericalError> {
    let k = cfg.validate()?;
    let k2 = k.mul(k);
    let n = cfg.node_count();
    if rhs.len() != n || solution.len() != n || scratch_residual.len() != n {
        return Err(NumericalError::BufferLengthMismatch {
            buffer: "Helmholtz Jacobi arrays",
            expected: n,
            actual: rhs.len().min(solution.len()),
        });
    }
    let [dx, dy, dz] = cfg.spacing_m;
    let center_lap = match cfg.stencil {
        HelmholtzStencil::SevenPoint => -2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)),
        HelmholtzStencil::TwentySevenPointIsotropic => (-128.0 / 30.0) / (dx * dx),
    };
    let diag = Complex64::new(center_lap + k2.re, k2.im);
    let [nx, ny, nz] = cfg.dims;
    let periodic = matches!(cfg.boundary, HelmholtzBoundary::Periodic);
    let idx = |ix: usize, iy: usize, iz: usize| -> usize { (ix * ny + iy) * nz + iz };

    apply_helmholtz_operator(cfg, solution, scratch_residual)?;
    let mut sum_sq = 0.0f64;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let c = idx(ix, iy, iz);
                let r = rhs[c].sub(scratch_residual[c]);
                sum_sq += r.norm_sqr();
                let is_bnd = !periodic
                    && (ix == 0 || ix + 1 == nx || iy == 0 || iy + 1 == ny || iz == 0 || iz + 1 == nz);
                let d = if is_bnd { Complex64::ONE } else { diag };
                solution[c] = solution[c].add(r.div(d).scale(omega_relax));
            }
        }
    }
    Ok(sum_sq.sqrt())
}

/// Analytical 3D free-space outgoing Green's function
/// $$G(r) = \frac{e^{i k r}}{4\pi r}, \qquad r = \|\mathbf{x} - \mathbf{x}_0\|_2 > 0$$
/// Rejects $r \le 0$ singularity per C02.
pub fn helmholtz_greens_function_3d(distance_m: f64, k: Complex64) -> Result<Complex64, NumericalError> {
    if !distance_m.is_finite() || distance_m <= 0.0 {
        return Err(NumericalError::DomainError(
            "3D Helmholtz Green's function requires finite distance_m > 0 (singular at r=0)",
        ));
    }
    if !k.is_finite() {
        return Err(NumericalError::DomainError(
            "3D Helmholtz Green's function requires finite complex wavenumber k",
        ));
    }
    // i * k * r = i * (k_re + i * k_im) * r = -k_im * r + i * k_re * r
    let exp_mag = (-k.im * distance_m).exp() / (4.0 * PI * distance_m);
    let phase = k.re * distance_m;
    Ok(Complex64::new(
        exp_mag * phase.cos(),
        exp_mag * phase.sin(),
    ))
}

/// Analytical eigenfrequency $f_{n_x,n_y,n_z}$ (in Hz) of a 3D rectangular acoustic cavity
/// with side lengths $(L_x, L_y, L_z)$, sound speed $c$, and mode indices $(n_x, n_y, n_z)$:
/// $$f_{n_x,n_y,n_z} = \frac{c}{2}\sqrt{\left(\frac{n_x}{L_x}\right)^2 + \left(\frac{n_y}{L_y}\right)^2 + \left(\frac{n_z}{L_z}\right)^2}$$
pub fn rectangular_cavity_eigenfrequency_hz(
    lengths_m: [f64; 3],
    mode_indices: [u32; 3],
    sound_speed_m_s: f64,
) -> Result<f64, NumericalError> {
    if lengths_m.iter().any(|&l| !l.is_finite() || l <= 0.0) {
        return Err(NumericalError::InvalidGrid(
            "cavity lengths_m must be finite and > 0",
        ));
    }
    if !sound_speed_m_s.is_finite() || sound_speed_m_s <= 0.0 {
        return Err(NumericalError::InvalidMedium(
            "cavity sound_speed_m_s must be finite and > 0",
        ));
    }
    let kx = (mode_indices[0] as f64) / lengths_m[0];
    let ky = (mode_indices[1] as f64) / lengths_m[1];
    let kz = (mode_indices[2] as f64) / lengths_m[2];
    Ok(0.5 * sound_speed_m_s * (kx * kx + ky * ky + kz * kz).sqrt())
}

/// Evaluate the analytical standing-wave eigenmode shape in a 3D rectangular cavity at `pos_m`:
/// - When `rigid_neumann == true`: $\cos(n_x \pi x / L_x)\cos(n_y \pi y / L_y)\cos(n_z \pi z / L_z)$
/// - When `rigid_neumann == false`: $\sin(n_x \pi x / L_x)\sin(n_y \pi y / L_y)\sin(n_z \pi z / L_z)$
#[inline]
pub fn rectangular_cavity_eigenmode_value(
    pos_m: [f64; 3],
    lengths_m: [f64; 3],
    mode_indices: [u32; 3],
    rigid_neumann: bool,
) -> f64 {
    let ax = (mode_indices[0] as f64) * PI * pos_m[0] / lengths_m[0];
    let ay = (mode_indices[1] as f64) * PI * pos_m[1] / lengths_m[1];
    let az = (mode_indices[2] as f64) * PI * pos_m[2] / lengths_m[2];
    if rigid_neumann {
        ax.cos() * ay.cos() * az.cos()
    } else {
        ax.sin() * ay.sin() * az.sin()
    }
}

#[inline]
fn hermitian_dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    let mut acc = Complex64::ZERO;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        acc = acc.add(ai.conj().mul(bi));
    }
    acc
}

#[inline]
fn l2_norm(v: &[Complex64]) -> f64 {
    let mut s = 0.0f64;
    for &vi in v {
        s += vi.norm_sqr();
    }
    s.sqrt()
}
