//! 3D Cartesian Yee FDTD Electromagnetic Leapfrog Step Kernel.
//!
//! Provides the certified WGSL compute shader [`MAXWELL_YEE_STEP_WGSL`], the exact
//! `f32` CPU reference oracle [`maxwell_yee_step_cpu`], buffer capacity validator
//! [`maxwell_yee_gpu_capacity`], and GPU execution harnesses
//! [`maxwell_yee_step_with_context`] and [`maxwell_yee_step_gpu`].
//!
//! State is represented as a flat `f32` buffer with [`MAXWELL_YEE_STRIDE`] (`9`) scalars
//! per voxel `(ix, iy, iz)` in row-major `[ix * ny * nz + iy * nz + iz]` order:
//! `[ex, ey, ez, hx, hy, hz, jx, jy, jz]`.
//!
//! To guarantee race-free execution across arbitrary workgroup scheduling, one full
//! leapfrog step $(\mathbf{E}^n, \mathbf{H}^{n-1/2}) \to (\mathbf{E}^{n+1}, \mathbf{H}^{n+1/2})$
//! executes as two out-of-place dispatches of [`MAXWELL_YEE_STEP_ENTRY`]:
//! 1. `stage = 0.0` ($\mathbf{H}$-field Faraday half-step): reads $\mathbf{E}^n$ from
//!    `state_in` (`read`) and writes updated $\mathbf{H}^{n+1/2}$ (preserving $\mathbf{E}^n, \mathbf{J}^n$)
//!    to `state_out` (`read_write`).
//! 2. `stage = 1.0` ($\mathbf{E}$-field Ampère-Maxwell step): reads $\mathbf{H}^{n+1/2}$
//!    from the first pass output (`read`) and writes updated $\mathbf{E}^{n+1}$ along with
//!    $\mathbf{H}^{n+1/2}, \mathbf{J}^n$ to `state_out` (`read_write`).

use crate::numerical::ForgeError;

/// Entry-point name of [`MAXWELL_YEE_STEP_WGSL`].
pub const MAXWELL_YEE_STEP_ENTRY: &str = "maxwell_yee_step";

/// Scalars per Yee cell in the flat state buffer: `[ex, ey, ez, hx, hy, hz, jx, jy, jz]`.
pub const MAXWELL_YEE_STRIDE: usize = 9;

/// Vacuum speed of light $c_0$ ($\text{m/s}$).
pub const C0_F32: f32 = 299_792_458.0;

/// WGSL compute shader for 3D periodic staggered Yee FDTD leapfrog updates.
///
/// Bindings:
/// - `@group(0) @binding(0) var<storage, read> state_in: array<f32>` (`9 * nx * ny * nz`)
/// - `@group(0) @binding(1) var<storage, read_write> state_out: array<f32>` (`9 * nx * ny * nz`)
/// - `@group(0) @binding(2) var<storage, read> params: array<f32>` (12 scalars):
///   `[nx, ny, nz, inv_dx, inv_dy, inv_dz, dt_over_mu, c_ee, c_eh, stage, 0.0, 0.0]`
pub const MAXWELL_YEE_STEP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> state_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;

fn cell_base(ix: u32, iy: u32, iz: u32, ny: u32, nz: u32) -> u32 {
    return ((ix * ny + iy) * nz + iz) * 9u;
}

@compute @workgroup_size(64)
fn maxwell_yee_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = u32(params[0]);
    let ny = u32(params[1]);
    let nz = u32(params[2]);
    let total_cells = nx * ny * nz;
    let idx = gid.x;
    if (idx >= total_cells) {
        return;
    }

    let yz = ny * nz;
    let ix = idx / yz;
    let rem = idx - ix * yz;
    let iy = rem / nz;
    let iz = rem - iy * nz;

    let inv_dx = params[3];
    let inv_dy = params[4];
    let inv_dz = params[5];
    let dt_over_mu = params[6];
    let c_ee = params[7];
    let c_eh = params[8];
    let stage = u32(params[9]);

    let base = idx * 9u;
    let ex = state_in[base + 0u];
    let ey = state_in[base + 1u];
    let ez = state_in[base + 2u];
    let hx = state_in[base + 3u];
    let hy = state_in[base + 4u];
    let hz = state_in[base + 5u];
    let jx = state_in[base + 6u];
    let jy = state_in[base + 7u];
    let jz = state_in[base + 8u];

    if (stage == 0u) {
        // Forward differences for curl(E) with periodic wrapping
        let ixp = select(ix + 1u, 0u, ix + 1u >= nx);
        let iyp = select(iy + 1u, 0u, iy + 1u >= ny);
        let izp = select(iz + 1u, 0u, iz + 1u >= nz);

        let bx = cell_base(ixp, iy, iz, ny, nz);
        let by = cell_base(ix, iyp, iz, ny, nz);
        let bz = cell_base(ix, iy, izp, ny, nz);

        let dey_dz = (state_in[bz + 1u] - ey) * inv_dz;
        let dez_dy = (state_in[by + 2u] - ez) * inv_dy;
        let dez_dx = (state_in[bx + 2u] - ez) * inv_dx;
        let dex_dz = (state_in[bz + 0u] - ex) * inv_dz;
        let dex_dy = (state_in[by + 0u] - ex) * inv_dy;
        let dey_dx = (state_in[bx + 1u] - ey) * inv_dx;

        let curl_ex = dez_dy - dey_dz;
        let curl_ey = dex_dz - dez_dx;
        let curl_ez = dey_dx - dex_dy;

        state_out[base + 0u] = ex;
        state_out[base + 1u] = ey;
        state_out[base + 2u] = ez;
        state_out[base + 3u] = hx - dt_over_mu * curl_ex;
        state_out[base + 4u] = hy - dt_over_mu * curl_ey;
        state_out[base + 5u] = hz - dt_over_mu * curl_ez;
        state_out[base + 6u] = jx;
        state_out[base + 7u] = jy;
        state_out[base + 8u] = jz;
    } else {
        // Backward differences for curl(H) with periodic wrapping
        let ixm = select(ix - 1u, nx - 1u, ix == 0u);
        let iym = select(iy - 1u, ny - 1u, iy == 0u);
        let izm = select(iz - 1u, nz - 1u, iz == 0u);

        let bxm = cell_base(ixm, iy, iz, ny, nz);
        let bym = cell_base(ix, iym, iz, ny, nz);
        let bzm = cell_base(ix, iy, izm, ny, nz);

        let dhy_dz = (hy - state_in[bzm + 4u]) * inv_dz;
        let dhz_dy = (hz - state_in[bym + 5u]) * inv_dy;
        let dhz_dx = (hz - state_in[bxm + 5u]) * inv_dx;
        let dhx_dz = (hx - state_in[bzm + 3u]) * inv_dz;
        let dhx_dy = (hx - state_in[bym + 3u]) * inv_dy;
        let dhy_dx = (hy - state_in[bxm + 4u]) * inv_dx;

        let curl_hx = dhz_dy - dhy_dz;
        let curl_hy = dhx_dz - dhz_dx;
        let curl_hz = dhy_dx - dhx_dy;

        state_out[base + 0u] = c_ee * ex + c_eh * (curl_hx - jx);
        state_out[base + 1u] = c_ee * ey + c_eh * (curl_hy - jy);
        state_out[base + 2u] = c_ee * ez + c_eh * (curl_hz - jz);
        state_out[base + 3u] = hx;
        state_out[base + 4u] = hy;
        state_out[base + 5u] = hz;
        state_out[base + 6u] = jx;
        state_out[base + 7u] = jy;
        state_out[base + 8u] = jz;
    }
}
"#;

/// Parameters for a 3D Yee FDTD leapfrog step on a periodic Cartesian grid.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxwellYeeParams {
    pub dims: [usize; 3],
    pub spacing: [f32; 3],
    pub dt: f32,
    pub eps: f32,
    pub mu: f32,
    pub sigma: f32,
}

impl MaxwellYeeParams {
    /// Total number of Yee cells `nx * ny * nz`.
    #[inline]
    pub fn cell_count(&self) -> Option<usize> {
        let [nx, ny, nz] = self.dims;
        if nx == 0 || ny == 0 || nz == 0 {
            return None;
        }
        nx.checked_mul(ny)?.checked_mul(nz)
    }

    /// Required `f32` buffer length `9 * nx * ny * nz`.
    #[inline]
    pub fn state_len(&self) -> Option<usize> {
        self.cell_count()?.checked_mul(MAXWELL_YEE_STRIDE)
    }

    /// Computes update coefficients `[inv_dx, inv_dy, inv_dz, dt_over_mu, c_ee, c_eh]`.
    #[inline]
    pub fn coefficients(&self) -> [f32; 6] {
        let inv_dx = 1.0 / self.spacing[0];
        let inv_dy = 1.0 / self.spacing[1];
        let inv_dz = 1.0 / self.spacing[2];
        let dt_over_mu = self.dt / self.mu;
        let loss = 0.5 * self.sigma * self.dt / self.eps;
        let denom = 1.0 + loss;
        let c_ee = (1.0 - loss) / denom;
        let c_eh = (self.dt / self.eps) / denom;
        [inv_dx, inv_dy, inv_dz, dt_over_mu, c_ee, c_eh]
    }

    /// Packs the 12-float shader uniform/storage parameter array for the given `stage` (`0` or `1`).
    #[inline]
    pub fn to_shader_params(&self, stage: u32) -> [f32; 12] {
        let [inv_dx, inv_dy, inv_dz, dt_over_mu, c_ee, c_eh] = self.coefficients();
        [
            self.dims[0] as f32,
            self.dims[1] as f32,
            self.dims[2] as f32,
            inv_dx,
            inv_dy,
            inv_dz,
            dt_over_mu,
            c_ee,
            c_eh,
            stage as f32,
            0.0,
            0.0,
        ]
    }
}

/// Validates `state` and `params` (including 3D Courant-Friedrichs-Lewy stability) and
/// returns the required transient slab byte capacity for [`maxwell_yee_step_with_context`].
pub fn maxwell_yee_gpu_capacity(
    state: &[f32],
    params: &MaxwellYeeParams,
) -> Result<usize, ForgeError> {
    let expected_len = params.state_len().ok_or_else(|| {
        ForgeError::GpuValidation("maxwell_yee: zero or overflowing grid dimensions".into())
    })?;
    if state.len() != expected_len {
        return Err(ForgeError::GpuValidation(format!(
            "maxwell_yee: state length {} does not match 9 * nx * ny * nz = {expected_len}",
            state.len()
        )));
    }
    let [dx, dy, dz] = params.spacing;
    if !(dx.is_finite()
        && dy.is_finite()
        && dz.is_finite()
        && dx > 0.0
        && dy > 0.0
        && dz > 0.0
        && params.dt.is_finite()
        && params.dt > 0.0
        && params.eps.is_finite()
        && params.eps > 0.0
        && params.mu.is_finite()
        && params.mu > 0.0
        && params.sigma.is_finite()
        && params.sigma >= 0.0)
    {
        return Err(ForgeError::GpuValidation(
            "maxwell_yee: spacing, dt, eps, mu must be positive finite and sigma >= 0".into(),
        ));
    }

    // Enforce 3D CFL bound: dt <= 1 / (v * sqrt(dx^-2 + dy^-2 + dz^-2))
    let v = 1.0 / (params.eps * params.mu).sqrt();
    let inv_h = (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)).sqrt();
    let dt_cfl = 1.0 / (v * inv_h);
    if params.dt > dt_cfl * (1.0 + 1.0e-6) {
        return Err(ForgeError::GpuValidation(format!(
            "maxwell_yee: CFL violation dt={} exceeds dt_cfl={dt_cfl}",
            params.dt
        )));
    }

    let bytes = expected_len
        .checked_mul(4)
        .and_then(|b| b.checked_mul(4))
        .ok_or_else(|| ForgeError::GpuValidation("maxwell_yee: byte capacity overflow".into()))?;
    Ok(bytes.max(4 << 20))
}

fn maxwell_yee_stage_cpu(state_in: &[f32], params: &MaxwellYeeParams, stage: u32) -> Vec<f32> {
    let [nx, ny, nz] = params.dims;
    let [inv_dx, inv_dy, inv_dz, dt_over_mu, c_ee, c_eh] = params.coefficients();
    let mut out = vec![0.0f32; state_in.len()];
    let cell_base = |ix: usize, iy: usize, iz: usize| -> usize {
        ((ix * ny + iy) * nz + iz) * MAXWELL_YEE_STRIDE
    };

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let base = cell_base(ix, iy, iz);
                let ex = state_in[base];
                let ey = state_in[base + 1];
                let ez = state_in[base + 2];
                let hx = state_in[base + 3];
                let hy = state_in[base + 4];
                let hz = state_in[base + 5];
                let jx = state_in[base + 6];
                let jy = state_in[base + 7];
                let jz = state_in[base + 8];

                if stage == 0 {
                    let ixp = if ix + 1 >= nx { 0 } else { ix + 1 };
                    let iyp = if iy + 1 >= ny { 0 } else { iy + 1 };
                    let izp = if iz + 1 >= nz { 0 } else { iz + 1 };

                    let bx = cell_base(ixp, iy, iz);
                    let by = cell_base(ix, iyp, iz);
                    let bz = cell_base(ix, iy, izp);

                    let dey_dz = (state_in[bz + 1] - ey) * inv_dz;
                    let dez_dy = (state_in[by + 2] - ez) * inv_dy;
                    let dez_dx = (state_in[bx + 2] - ez) * inv_dx;
                    let dex_dz = (state_in[bz] - ex) * inv_dz;
                    let dex_dy = (state_in[by] - ex) * inv_dy;
                    let dey_dx = (state_in[bx + 1] - ey) * inv_dx;

                    let curl_ex = dez_dy - dey_dz;
                    let curl_ey = dex_dz - dez_dx;
                    let curl_ez = dey_dx - dex_dy;

                    out[base] = ex;
                    out[base + 1] = ey;
                    out[base + 2] = ez;
                    out[base + 3] = hx - dt_over_mu * curl_ex;
                    out[base + 4] = hy - dt_over_mu * curl_ey;
                    out[base + 5] = hz - dt_over_mu * curl_ez;
                    out[base + 6] = jx;
                    out[base + 7] = jy;
                    out[base + 8] = jz;
                } else {
                    let ixm = if ix == 0 { nx - 1 } else { ix - 1 };
                    let iym = if iy == 0 { ny - 1 } else { iy - 1 };
                    let izm = if iz == 0 { nz - 1 } else { iz - 1 };

                    let bxm = cell_base(ixm, iy, iz);
                    let bym = cell_base(ix, iym, iz);
                    let bzm = cell_base(ix, iy, izm);

                    let dhy_dz = (hy - state_in[bzm + 4]) * inv_dz;
                    let dhz_dy = (hz - state_in[bym + 5]) * inv_dy;
                    let dhz_dx = (hz - state_in[bxm + 5]) * inv_dx;
                    let dhx_dz = (hx - state_in[bzm + 3]) * inv_dz;
                    let dhx_dy = (hx - state_in[bym + 3]) * inv_dy;
                    let dhy_dx = (hy - state_in[bxm + 4]) * inv_dx;

                    let curl_hx = dhz_dy - dhy_dz;
                    let curl_hy = dhx_dz - dhz_dx;
                    let curl_hz = dhy_dx - dhx_dy;

                    out[base] = c_ee * ex + c_eh * (curl_hx - jx);
                    out[base + 1] = c_ee * ey + c_eh * (curl_hy - jy);
                    out[base + 2] = c_ee * ez + c_eh * (curl_hz - jz);
                    out[base + 3] = hx;
                    out[base + 4] = hy;
                    out[base + 5] = hz;
                    out[base + 6] = jx;
                    out[base + 7] = jy;
                    out[base + 8] = jz;
                }
            }
        }
    }
    out
}

/// Exact `f32` CPU oracle for one complete 3D Yee FDTD leapfrog step
/// ($\mathbf{H}$ half-step followed by $\mathbf{E}$ full-step), mirroring
/// [`MAXWELL_YEE_STEP_WGSL`] scalar-for-scalar.
pub fn maxwell_yee_step_cpu(
    state_in: &[f32],
    params: &MaxwellYeeParams,
) -> Result<Vec<f32>, ForgeError> {
    maxwell_yee_gpu_capacity(state_in, params)?;
    let mid = maxwell_yee_stage_cpu(state_in, params, 0);
    Ok(maxwell_yee_stage_cpu(&mid, params, 1))
}

/// Executes one complete 3D Yee FDTD leapfrog step on the GPU using a caller-supplied
/// [`WgpuComputeContext`](crate::numerical::execute::WgpuComputeContext).
pub fn maxwell_yee_step_with_context(
    ctx: &mut crate::numerical::execute::WgpuComputeContext,
    state_in: &[f32],
    params: &MaxwellYeeParams,
) -> Result<Vec<f32>, ForgeError> {
    use crate::numerical::execute::{BindingUsage, QualiaCompute, WgpuPipeline};
    use crate::numerical::Schedule;

    maxwell_yee_gpu_capacity(state_in, params)?;
    let cell_count = params.cell_count().unwrap();
    let len = state_in.len();
    let zeros = vec![0.0f32; len];
    let schedule = Schedule {
        workgroup_size: 64,
        ..Default::default()
    };

    // Pass 1: H-field half-step (stage = 0)
    let view_in_0 = ctx.allocate_and_write(
        bytemuck::cast_slice(state_in),
        0,
        0,
        BindingUsage::StorageRead,
    )?;
    let view_out_0 = ctx.allocate_and_write(
        bytemuck::cast_slice(&zeros),
        1,
        0,
        BindingUsage::StorageReadWrite,
    )?;
    let params_0 = params.to_shader_params(0);
    let view_params_0 = ctx.allocate_and_write(
        bytemuck::cast_slice(&params_0),
        2,
        0,
        BindingUsage::StorageRead,
    )?;

    {
        let pipeline = WgpuPipeline::compile(ctx, MAXWELL_YEE_STEP_WGSL, MAXWELL_YEE_STEP_ENTRY)?;
        pipeline.dispatch(
            &[view_in_0, view_out_0, view_params_0],
            &schedule,
            cell_count,
        )?;
    }
    let mut mid = ctx.read_buffer_f32(&view_out_0)?;
    mid.truncate(len);

    // Pass 2: E-field full-step (stage = 1)
    let view_in_1 =
        ctx.allocate_and_write(bytemuck::cast_slice(&mid), 0, 0, BindingUsage::StorageRead)?;
    let view_out_1 = ctx.allocate_and_write(
        bytemuck::cast_slice(&zeros),
        1,
        0,
        BindingUsage::StorageReadWrite,
    )?;
    let params_1 = params.to_shader_params(1);
    let view_params_1 = ctx.allocate_and_write(
        bytemuck::cast_slice(&params_1),
        2,
        0,
        BindingUsage::StorageRead,
    )?;

    {
        let pipeline = WgpuPipeline::compile(ctx, MAXWELL_YEE_STEP_WGSL, MAXWELL_YEE_STEP_ENTRY)?;
        pipeline.dispatch(
            &[view_in_1, view_out_1, view_params_1],
            &schedule,
            cell_count,
        )?;
    }
    let mut out = ctx.read_buffer_f32(&view_out_1)?;
    out.truncate(len);
    Ok(out)
}

/// Convenience wrapper that creates a transient [`WgpuComputeContext`](crate::numerical::execute::WgpuComputeContext)
/// and delegates to [`maxwell_yee_step_with_context`].
pub fn maxwell_yee_step_gpu(
    state_in: &[f32],
    params: &MaxwellYeeParams,
) -> Result<Vec<f32>, ForgeError> {
    use crate::numerical::execute::WgpuComputeContext;
    let capacity = maxwell_yee_gpu_capacity(state_in, params)?;
    let mut ctx = WgpuComputeContext::new(capacity)?;
    maxwell_yee_step_with_context(&mut ctx, state_in, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::validate::validate_wgsl;

    #[test]
    fn maxwell_yee_wgsl_validates() {
        let report =
            validate_wgsl(MAXWELL_YEE_STEP_WGSL).expect("maxwell_yee_step WGSL must naga-validate");
        assert!(
            report
                .entry_points
                .iter()
                .any(|e| e == MAXWELL_YEE_STEP_ENTRY),
            "expected entry point {MAXWELL_YEE_STEP_ENTRY}, got {:?}",
            report.entry_points
        );
        assert_eq!(report.binding_count, 3);
    }

    #[test]
    fn maxwell_yee_cfl_gate_rejects_super_courant_step() {
        let params = MaxwellYeeParams {
            dims: [4, 4, 4],
            spacing: [1.0, 1.0, 1.0],
            dt: 0.60, // > 1 / sqrt(3) = 0.57735026
            eps: 1.0,
            mu: 1.0,
            sigma: 0.0,
        };
        let state = vec![0.0f32; 4 * 4 * 4 * MAXWELL_YEE_STRIDE];
        assert!(maxwell_yee_gpu_capacity(&state, &params).is_err());
    }

    #[test]
    fn maxwell_yee_cpu_step_advances_plane_wave() {
        let nx = 8usize;
        let ny = 4usize;
        let nz = 4usize;
        let params = MaxwellYeeParams {
            dims: [nx, ny, nz],
            spacing: [1.0, 1.0, 1.0],
            dt: 0.5,
            eps: 1.0,
            mu: 1.0,
            sigma: 0.0,
        };
        let mut state = vec![0.0f32; nx * ny * nz * MAXWELL_YEE_STRIDE];
        let k = 2.0 * core::f32::consts::PI / (nx as f32);
        for ix in 0..nx {
            let phase = k * (ix as f32);
            let ey = phase.sin();
            let hz = (phase + 0.5 * k).sin();
            for iy in 0..ny {
                for iz in 0..nz {
                    let base = ((ix * ny + iy) * nz + iz) * MAXWELL_YEE_STRIDE;
                    state[base + 1] = ey;
                    state[base + 5] = hz;
                }
            }
        }
        let next = maxwell_yee_step_cpu(&state, &params).expect("valid CFL step");
        assert_eq!(next.len(), state.len());
        let diff_norm: f32 = state
            .iter()
            .zip(next.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_norm > 1.0e-3, "wave must advance over one timestep");
    }

    #[test]
    fn maxwell_yee_gpu_matches_cpu_oracle() {
        if !crate::numerical::test_gpu_available() {
            return;
        }
        let nx = 6usize;
        let ny = 4usize;
        let nz = 4usize;
        let params = MaxwellYeeParams {
            dims: [nx, ny, nz],
            spacing: [1.0, 1.0, 1.0],
            dt: 0.45,
            eps: 1.0,
            mu: 1.0,
            sigma: 0.02,
        };
        let mut state = vec![0.0f32; nx * ny * nz * MAXWELL_YEE_STRIDE];
        for (i, v) in state.iter_mut().enumerate() {
            *v = ((i as f32) * 0.137).sin() * 0.5;
        }
        let cpu = maxwell_yee_step_cpu(&state, &params).unwrap();
        let gpu = maxwell_yee_step_gpu(&state, &params).unwrap();
        for (idx, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (c - g).abs() <= 1.0e-5,
                "CPU/GPU mismatch at scalar {idx}: cpu={c}, gpu={g}"
            );
        }
    }
}
