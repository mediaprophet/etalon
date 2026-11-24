//! 3D Staggered & Collocated Electromagnetic Vector Calculus Operators (`etalon-sim`).
//!
//! Provides second-order and fourth-order finite-difference Yee staggered-grid
//! curl, divergence, double-curl, Gauss-law residuals, discrete charge conservation,
//! Stokes-theorem face circulation/flux verification, and analytical Fourier-mode
//! spectral differential operators. All hot-path routines execute with zero heap
//! allocations over caller-supplied slices and are `wasm32`-compatible.

use super::{BoundaryCondition, EmfOperatorError, Grid3dSpec};

#[inline]
fn check_slice_len(slice: &[f64], expected: usize) -> Result<(), EmfOperatorError> {
    if slice.len() != expected {
        return Err(EmfOperatorError::SliceLengthMismatch {
            expected,
            provided: slice.len(),
        });
    }
    if slice.iter().any(|v| !v.is_finite()) {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    Ok(())
}

#[inline]
fn check_mut_slice_len(slice: &[f64], expected: usize) -> Result<(), EmfOperatorError> {
    if slice.len() != expected {
        return Err(EmfOperatorError::SliceLengthMismatch {
            expected,
            provided: slice.len(),
        });
    }
    Ok(())
}

#[inline]
fn step_plus(idx: usize, dim: usize, delta: usize) -> usize {
    (idx + delta) % dim
}

#[inline]
fn step_minus(idx: usize, dim: usize, delta: usize) -> usize {
    (idx + dim - (delta % dim)) % dim
}

#[inline]
fn sample_shifted(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    field: &[f64],
    ix: usize,
    iy: usize,
    iz: usize,
    sx: isize,
    sy: isize,
    sz: isize,
) -> f64 {
    match bc {
        BoundaryCondition::Periodic => {
            let nx = if sx >= 0 {
                step_plus(ix, grid.nx, sx as usize)
            } else {
                step_minus(ix, grid.nx, (-sx) as usize)
            };
            let ny = if sy >= 0 {
                step_plus(iy, grid.ny, sy as usize)
            } else {
                step_minus(iy, grid.ny, (-sy) as usize)
            };
            let nz = if sz >= 0 {
                step_plus(iz, grid.nz, sz as usize)
            } else {
                step_minus(iz, grid.nz, (-sz) as usize)
            };
            field[grid.idx(nx, ny, nz)]
        }
        BoundaryCondition::DirichletPec => {
            let nx = ix as isize + sx;
            let ny = iy as isize + sy;
            let nz = iz as isize + sz;
            if nx < 0
                || nx >= grid.nx as isize
                || ny < 0
                || ny >= grid.ny as isize
                || nz < 0
                || nz >= grid.nz as isize
            {
                0.0
            } else {
                field[grid.idx(nx as usize, ny as usize, nz as usize)]
            }
        }
    }
}

/// Second-order staggered Yee forward curl $\nabla \times \mathbf{E}$ mapping edge-staggered
/// electric fields $(E_x, E_y, E_z)$ to face-staggered magnetic flux rates.
#[allow(clippy::too_many_arguments)]
pub fn yee_curl_e_2nd(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    ex: &[f64],
    ey: &[f64],
    ez: &[f64],
    curl_x: &mut [f64],
    curl_y: &mut [f64],
    curl_z: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(ex, n)?;
    check_slice_len(ey, n)?;
    check_slice_len(ez, n)?;
    check_mut_slice_len(curl_x, n)?;
    check_mut_slice_len(curl_y, n)?;
    check_mut_slice_len(curl_z, n)?;

    let inv_dx = 1.0 / grid.dx;
    let inv_dy = 1.0 / grid.dy;
    let inv_dz = 1.0 / grid.dz;

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let ex_c = ex[c];
                let ey_c = ey[c];
                let ez_c = ez[c];

                let ez_yp = sample_shifted(grid, bc, ez, ix, iy, iz, 0, 1, 0);
                let ey_zp = sample_shifted(grid, bc, ey, ix, iy, iz, 0, 0, 1);
                let ex_zp = sample_shifted(grid, bc, ex, ix, iy, iz, 0, 0, 1);
                let ez_xp = sample_shifted(grid, bc, ez, ix, iy, iz, 1, 0, 0);
                let ey_xp = sample_shifted(grid, bc, ey, ix, iy, iz, 1, 0, 0);
                let ex_yp = sample_shifted(grid, bc, ex, ix, iy, iz, 0, 1, 0);

                curl_x[c] = (ez_yp - ez_c) * inv_dy - (ey_zp - ey_c) * inv_dz;
                curl_y[c] = (ex_zp - ex_c) * inv_dz - (ez_xp - ez_c) * inv_dx;
                curl_z[c] = (ey_xp - ey_c) * inv_dx - (ex_yp - ex_c) * inv_dy;
            }
        }
    }
    Ok(())
}

/// Second-order staggered Yee backward curl $\nabla \times \mathbf{H}$ mapping face-staggered
/// magnetic fields $(H_x, H_y, H_z)$ to edge-staggered electric displacement rates.
#[allow(clippy::too_many_arguments)]
pub fn yee_curl_h_2nd(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    hx: &[f64],
    hy: &[f64],
    hz: &[f64],
    curl_x: &mut [f64],
    curl_y: &mut [f64],
    curl_z: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(hx, n)?;
    check_slice_len(hy, n)?;
    check_slice_len(hz, n)?;
    check_mut_slice_len(curl_x, n)?;
    check_mut_slice_len(curl_y, n)?;
    check_mut_slice_len(curl_z, n)?;

    let inv_dx = 1.0 / grid.dx;
    let inv_dy = 1.0 / grid.dy;
    let inv_dz = 1.0 / grid.dz;

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let hx_c = hx[c];
                let hy_c = hy[c];
                let hz_c = hz[c];

                let hz_ym = sample_shifted(grid, bc, hz, ix, iy, iz, 0, -1, 0);
                let hy_zm = sample_shifted(grid, bc, hy, ix, iy, iz, 0, 0, -1);
                let hx_zm = sample_shifted(grid, bc, hx, ix, iy, iz, 0, 0, -1);
                let hz_xm = sample_shifted(grid, bc, hz, ix, iy, iz, -1, 0, 0);
                let hy_xm = sample_shifted(grid, bc, hy, ix, iy, iz, -1, 0, 0);
                let hx_ym = sample_shifted(grid, bc, hx, ix, iy, iz, 0, -1, 0);

                curl_x[c] = (hz_c - hz_ym) * inv_dy - (hy_c - hy_zm) * inv_dz;
                curl_y[c] = (hx_c - hx_zm) * inv_dz - (hz_c - hz_xm) * inv_dx;
                curl_z[c] = (hy_c - hy_xm) * inv_dx - (hx_c - hx_ym) * inv_dy;
            }
        }
    }
    Ok(())
}

/// Fourth-order staggered forward curl $\nabla^{(4)} \times \mathbf{E}$ (Fang-type 4th-order stencil
/// with weights $c_1 = 9/8$ and $c_2 = -1/24$).
#[allow(clippy::too_many_arguments)]
pub fn yee_curl_e_4th(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    ex: &[f64],
    ey: &[f64],
    ez: &[f64],
    curl_x: &mut [f64],
    curl_y: &mut [f64],
    curl_z: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(ex, n)?;
    check_slice_len(ey, n)?;
    check_slice_len(ez, n)?;
    check_mut_slice_len(curl_x, n)?;
    check_mut_slice_len(curl_y, n)?;
    check_mut_slice_len(curl_z, n)?;

    const C1: f64 = 9.0 / 8.0;
    const C2: f64 = -1.0 / 24.0;
    let inv_dx = 1.0 / grid.dx;
    let inv_dy = 1.0 / grid.dy;
    let inv_dz = 1.0 / grid.dz;

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let ex_c = ex[c];
                let ey_c = ey[c];
                let ez_c = ez[c];

                let dez_dy = (C1 * (sample_shifted(grid, bc, ez, ix, iy, iz, 0, 1, 0) - ez_c)
                    + C2 * (sample_shifted(grid, bc, ez, ix, iy, iz, 0, 2, 0)
                        - sample_shifted(grid, bc, ez, ix, iy, iz, 0, -1, 0)))
                    * inv_dy;
                let dey_dz = (C1 * (sample_shifted(grid, bc, ey, ix, iy, iz, 0, 0, 1) - ey_c)
                    + C2 * (sample_shifted(grid, bc, ey, ix, iy, iz, 0, 0, 2)
                        - sample_shifted(grid, bc, ey, ix, iy, iz, 0, 0, -1)))
                    * inv_dz;

                let dex_dz = (C1 * (sample_shifted(grid, bc, ex, ix, iy, iz, 0, 0, 1) - ex_c)
                    + C2 * (sample_shifted(grid, bc, ex, ix, iy, iz, 0, 0, 2)
                        - sample_shifted(grid, bc, ex, ix, iy, iz, 0, 0, -1)))
                    * inv_dz;
                let dez_dx = (C1 * (sample_shifted(grid, bc, ez, ix, iy, iz, 1, 0, 0) - ez_c)
                    + C2 * (sample_shifted(grid, bc, ez, ix, iy, iz, 2, 0, 0)
                        - sample_shifted(grid, bc, ez, ix, iy, iz, -1, 0, 0)))
                    * inv_dx;

                let dey_dx = (C1 * (sample_shifted(grid, bc, ey, ix, iy, iz, 1, 0, 0) - ey_c)
                    + C2 * (sample_shifted(grid, bc, ey, ix, iy, iz, 2, 0, 0)
                        - sample_shifted(grid, bc, ey, ix, iy, iz, -1, 0, 0)))
                    * inv_dx;
                let dex_dy = (C1 * (sample_shifted(grid, bc, ex, ix, iy, iz, 0, 1, 0) - ex_c)
                    + C2 * (sample_shifted(grid, bc, ex, ix, iy, iz, 0, 2, 0)
                        - sample_shifted(grid, bc, ex, ix, iy, iz, 0, -1, 0)))
                    * inv_dy;

                curl_x[c] = dez_dy - dey_dz;
                curl_y[c] = dex_dz - dez_dx;
                curl_z[c] = dey_dx - dex_dy;
            }
        }
    }
    Ok(())
}

/// Fourth-order staggered backward curl $\nabla^{(4)} \times \mathbf{H}$.
#[allow(clippy::too_many_arguments)]
pub fn yee_curl_h_4th(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    hx: &[f64],
    hy: &[f64],
    hz: &[f64],
    curl_x: &mut [f64],
    curl_y: &mut [f64],
    curl_z: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(hx, n)?;
    check_slice_len(hy, n)?;
    check_slice_len(hz, n)?;
    check_mut_slice_len(curl_x, n)?;
    check_mut_slice_len(curl_y, n)?;
    check_mut_slice_len(curl_z, n)?;

    const C1: f64 = 9.0 / 8.0;
    const C2: f64 = -1.0 / 24.0;
    let inv_dx = 1.0 / grid.dx;
    let inv_dy = 1.0 / grid.dy;
    let inv_dz = 1.0 / grid.dz;

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let hx_c = hx[c];
                let hy_c = hy[c];
                let hz_c = hz[c];

                let dhz_dy = (C1 * (hz_c - sample_shifted(grid, bc, hz, ix, iy, iz, 0, -1, 0))
                    + C2 * (sample_shifted(grid, bc, hz, ix, iy, iz, 0, 1, 0)
                        - sample_shifted(grid, bc, hz, ix, iy, iz, 0, -2, 0)))
                    * inv_dy;
                let dhy_dz = (C1 * (hy_c - sample_shifted(grid, bc, hy, ix, iy, iz, 0, 0, -1))
                    + C2 * (sample_shifted(grid, bc, hy, ix, iy, iz, 0, 0, 1)
                        - sample_shifted(grid, bc, hy, ix, iy, iz, 0, 0, -2)))
                    * inv_dz;

                let dhx_dz = (C1 * (hx_c - sample_shifted(grid, bc, hx, ix, iy, iz, 0, 0, -1))
                    + C2 * (sample_shifted(grid, bc, hx, ix, iy, iz, 0, 0, 1)
                        - sample_shifted(grid, bc, hx, ix, iy, iz, 0, 0, -2)))
                    * inv_dz;
                let dhz_dx = (C1 * (hz_c - sample_shifted(grid, bc, hz, ix, iy, iz, -1, 0, 0))
                    + C2 * (sample_shifted(grid, bc, hz, ix, iy, iz, 1, 0, 0)
                        - sample_shifted(grid, bc, hz, ix, iy, iz, -2, 0, 0)))
                    * inv_dx;

                let dhy_dx = (C1 * (hy_c - sample_shifted(grid, bc, hy, ix, iy, iz, -1, 0, 0))
                    + C2 * (sample_shifted(grid, bc, hy, ix, iy, iz, 1, 0, 0)
                        - sample_shifted(grid, bc, hy, ix, iy, iz, -2, 0, 0)))
                    * inv_dx;
                let dhx_dy = (C1 * (hx_c - sample_shifted(grid, bc, hx, ix, iy, iz, 0, -1, 0))
                    + C2 * (sample_shifted(grid, bc, hx, ix, iy, iz, 0, 1, 0)
                        - sample_shifted(grid, bc, hx, ix, iy, iz, 0, -2, 0)))
                    * inv_dy;

                curl_x[c] = dhz_dy - dhy_dz;
                curl_y[c] = dhx_dz - dhz_dx;
                curl_z[c] = dhy_dx - dhx_dy;
            }
        }
    }
    Ok(())
}

/// Second-order staggered vertex divergence $\nabla \cdot \mathbf{D}$ for edge-staggered fields.
pub fn yee_div_d_2nd(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    dx_field: &[f64],
    dy_field: &[f64],
    dz_field: &[f64],
    div_out: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(dx_field, n)?;
    check_slice_len(dy_field, n)?;
    check_slice_len(dz_field, n)?;
    check_mut_slice_len(div_out, n)?;

    let inv_dx = 1.0 / grid.dx;
    let inv_dy = 1.0 / grid.dy;
    let inv_dz = 1.0 / grid.dz;

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let d_xm = sample_shifted(grid, bc, dx_field, ix, iy, iz, -1, 0, 0);
                let d_ym = sample_shifted(grid, bc, dy_field, ix, iy, iz, 0, -1, 0);
                let d_zm = sample_shifted(grid, bc, dz_field, ix, iy, iz, 0, 0, -1);

                div_out[c] = (dx_field[c] - d_xm) * inv_dx
                    + (dy_field[c] - d_ym) * inv_dy
                    + (dz_field[c] - d_zm) * inv_dz;
            }
        }
    }
    Ok(())
}

/// Second-order staggered cell-center divergence $\nabla \cdot \mathbf{B}$ for face-staggered fields.
pub fn yee_div_b_2nd(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    bx_field: &[f64],
    by_field: &[f64],
    bz_field: &[f64],
    div_out: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(bx_field, n)?;
    check_slice_len(by_field, n)?;
    check_slice_len(bz_field, n)?;
    check_mut_slice_len(div_out, n)?;

    let inv_dx = 1.0 / grid.dx;
    let inv_dy = 1.0 / grid.dy;
    let inv_dz = 1.0 / grid.dz;

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let b_xp = sample_shifted(grid, bc, bx_field, ix, iy, iz, 1, 0, 0);
                let b_yp = sample_shifted(grid, bc, by_field, ix, iy, iz, 0, 1, 0);
                let b_zp = sample_shifted(grid, bc, bz_field, ix, iy, iz, 0, 0, 1);

                div_out[c] = (b_xp - bx_field[c]) * inv_dx
                    + (b_yp - by_field[c]) * inv_dy
                    + (b_zp - bz_field[c]) * inv_dz;
            }
        }
    }
    Ok(())
}

/// Compute the electric Gauss-law residual $r_D = \nabla \cdot \mathbf{D} - \rho_v$.
pub fn gauss_electric_residual(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    dx_field: &[f64],
    dy_field: &[f64],
    dz_field: &[f64],
    rho_v: &[f64],
    residual_out: &mut [f64],
) -> Result<f64, EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(rho_v, n)?;
    yee_div_d_2nd(grid, bc, dx_field, dy_field, dz_field, residual_out)?;
    let mut max_res = 0.0_f64;
    for i in 0..n {
        residual_out[i] -= rho_v[i];
        max_res = max_res.max(residual_out[i].abs());
    }
    Ok(max_res)
}

/// Compute the magnetic Gauss-law residual $r_B = \nabla \cdot \mathbf{B}$.
pub fn gauss_magnetic_residual(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    bx_field: &[f64],
    by_field: &[f64],
    bz_field: &[f64],
    residual_out: &mut [f64],
) -> Result<f64, EmfOperatorError> {
    yee_div_b_2nd(grid, bc, bx_field, by_field, bz_field, residual_out)?;
    let mut max_res = 0.0_f64;
    for &val in residual_out.iter() {
        max_res = max_res.max(val.abs());
    }
    Ok(max_res)
}

/// Verify discrete charge conservation on the Yee lattice:
/// $\frac{(\nabla \cdot \mathbf{D})^{n+1} - (\nabla \cdot \mathbf{D})^n}{\Delta t} + \nabla \cdot \mathbf{J}^{n+1/2} = 0$.
#[allow(clippy::too_many_arguments)]
pub fn verify_discrete_charge_conservation(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    dt: f64,
    dx_n: &[f64],
    dy_n: &[f64],
    dz_n: &[f64],
    dx_np1: &[f64],
    dy_np1: &[f64],
    dz_np1: &[f64],
    jx_half: &[f64],
    jy_half: &[f64],
    jz_half: &[f64],
    scratch_div_n: &mut [f64],
    scratch_div_np1: &mut [f64],
    scratch_div_j: &mut [f64],
) -> Result<f64, EmfOperatorError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    yee_div_d_2nd(grid, bc, dx_n, dy_n, dz_n, scratch_div_n)?;
    yee_div_d_2nd(grid, bc, dx_np1, dy_np1, dz_np1, scratch_div_np1)?;
    yee_div_d_2nd(grid, bc, jx_half, jy_half, jz_half, scratch_div_j)?;

    let inv_dt = 1.0 / dt;
    let mut max_residual = 0.0_f64;
    for i in 0..scratch_div_n.len() {
        let d_div_dt = (scratch_div_np1[i] - scratch_div_n[i]) * inv_dt;
        let continuity_res = (d_div_dt + scratch_div_j[i]).abs();
        max_residual = max_residual.max(continuity_res);
    }
    Ok(max_residual)
}

/// Evaluate the staggered double-curl operator $\nabla \times (\mu^{-1} \nabla \times \mathbf{E})$.
#[allow(clippy::too_many_arguments)]
pub fn yee_double_curl_e_2nd(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    inv_mu: &[f64],
    ex: &[f64],
    ey: &[f64],
    ez: &[f64],
    scratch_fx: &mut [f64],
    scratch_fy: &mut [f64],
    scratch_fz: &mut [f64],
    out_x: &mut [f64],
    out_y: &mut [f64],
    out_z: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(inv_mu, n)?;
    yee_curl_e_2nd(grid, bc, ex, ey, ez, scratch_fx, scratch_fy, scratch_fz)?;
    for i in 0..n {
        scratch_fx[i] *= inv_mu[i];
        scratch_fy[i] *= inv_mu[i];
        scratch_fz[i] *= inv_mu[i];
    }
    yee_curl_h_2nd(
        grid, bc, scratch_fx, scratch_fy, scratch_fz, out_x, out_y, out_z,
    )
}

/// Verify the exact staggered discrete vector identity $\nabla \cdot (\nabla \times \mathbf{A}) = 0$.
#[allow(clippy::too_many_arguments)]
pub fn verify_div_curl_zero_2nd(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    ex: &[f64],
    ey: &[f64],
    ez: &[f64],
    hx: &[f64],
    hy: &[f64],
    hz: &[f64],
    scratch_cx: &mut [f64],
    scratch_cy: &mut [f64],
    scratch_cz: &mut [f64],
    scratch_div: &mut [f64],
) -> Result<(f64, f64), EmfOperatorError> {
    yee_curl_e_2nd(grid, bc, ex, ey, ez, scratch_cx, scratch_cy, scratch_cz)?;
    let max_e = gauss_magnetic_residual(grid, bc, scratch_cx, scratch_cy, scratch_cz, scratch_div)?;

    yee_curl_h_2nd(grid, bc, hx, hy, hz, scratch_cx, scratch_cy, scratch_cz)?;
    yee_div_d_2nd(grid, bc, scratch_cx, scratch_cy, scratch_cz, scratch_div)?;
    let mut max_h = 0.0_f64;
    for &v in scratch_div.iter() {
        max_h = max_h.max(v.abs());
    }
    Ok((max_e, max_h))
}

/// Verify integral Stokes theorem $\oint_{\partial S} \mathbf{E} \cdot d\mathbf{l} = \iint_S (\nabla \times \mathbf{E}) \cdot d\mathbf{A}$
/// on a single Yee cell `(ix, iy, iz)`.
#[allow(clippy::too_many_arguments)]
pub fn verify_stokes_circulation_cell(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    ex: &[f64],
    ey: &[f64],
    ez: &[f64],
    curl_x: &[f64],
    curl_y: &[f64],
    curl_z: &[f64],
    ix: usize,
    iy: usize,
    iz: usize,
) -> Result<[f64; 3], EmfOperatorError> {
    let n = grid.validate()?;
    check_slice_len(ex, n)?;
    check_slice_len(ey, n)?;
    check_slice_len(ez, n)?;
    check_slice_len(curl_x, n)?;
    check_slice_len(curl_y, n)?;
    check_slice_len(curl_z, n)?;

    let c = grid.idx(ix, iy, iz);
    let ex_c = ex[c];
    let ey_c = ey[c];
    let ez_c = ez[c];

    let ez_yp = sample_shifted(grid, bc, ez, ix, iy, iz, 0, 1, 0);
    let ey_zp = sample_shifted(grid, bc, ey, ix, iy, iz, 0, 0, 1);
    let ex_zp = sample_shifted(grid, bc, ex, ix, iy, iz, 0, 0, 1);
    let ez_xp = sample_shifted(grid, bc, ez, ix, iy, iz, 1, 0, 0);
    let ey_xp = sample_shifted(grid, bc, ey, ix, iy, iz, 1, 0, 0);
    let ex_yp = sample_shifted(grid, bc, ex, ix, iy, iz, 0, 1, 0);

    let circ_yz = (ey_c - ey_zp) * grid.dy + (ez_yp - ez_c) * grid.dz;
    let flux_yz = curl_x[c] * (grid.dy * grid.dz);

    let circ_zx = (ez_c - ez_xp) * grid.dz + (ex_zp - ex_c) * grid.dx;
    let flux_zx = curl_y[c] * (grid.dz * grid.dx);

    let circ_xy = (ex_c - ex_yp) * grid.dx + (ey_xp - ey_c) * grid.dy;
    let flux_xy = curl_z[c] * (grid.dx * grid.dy);

    Ok([
        (circ_yz - flux_yz).abs(),
        (circ_zx - flux_zx).abs(),
        (circ_xy - flux_xy).abs(),
    ])
}

/// Exact spectral Fourier-mode curl oracle for a real plane wave
/// $\mathbf{A}(\mathbf{r}) = \mathbf{A}_0 \cos(\mathbf{k}\cdot\mathbf{r} + \phi_0)$.
#[inline]
pub fn spectral_plane_wave_curl(
    k_vec: [f64; 3],
    a0: [f64; 3],
    r_m: [f64; 3],
    phase0_rad: f64,
) -> [f64; 3] {
    let phase = k_vec[0] * r_m[0] + k_vec[1] * r_m[1] + k_vec[2] * r_m[2] + phase0_rad;
    let s = -phase.sin();
    let k_cross_a = [
        k_vec[1] * a0[2] - k_vec[2] * a0[1],
        k_vec[2] * a0[0] - k_vec[0] * a0[2],
        k_vec[0] * a0[1] - k_vec[1] * a0[0],
    ];
    [k_cross_a[0] * s, k_cross_a[1] * s, k_cross_a[2] * s]
}

/// Exact spectral Fourier-mode divergence oracle for a real plane wave
/// $\mathbf{A}(\mathbf{r}) = \mathbf{A}_0 \cos(\mathbf{k}\cdot\mathbf{r} + \phi_0)$.
#[inline]
pub fn spectral_plane_wave_divergence(
    k_vec: [f64; 3],
    a0: [f64; 3],
    r_m: [f64; 3],
    phase0_rad: f64,
) -> f64 {
    let phase = k_vec[0] * r_m[0] + k_vec[1] * r_m[1] + k_vec[2] * r_m[2] + phase0_rad;
    let k_dot_a = k_vec[0] * a0[0] + k_vec[1] * a0[1] + k_vec[2] * a0[2];
    -k_dot_a * phase.sin()
}
