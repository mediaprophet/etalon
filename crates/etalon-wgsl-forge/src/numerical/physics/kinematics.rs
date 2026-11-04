//! Softened inverse-square N-body step as a certified forge kernel.
//!
//! Embeds [`shaders/kinematics.wgsl`](kinematics.wgsl) via
//! `include_str!` (single source of truth), grades it against the exact CPU oracle
//! [`nbody_step_cpu`], and runs it on any wgpu adapter via [`nbody_step_gpu`].
//!
//! State is a flat `f32` buffer, 8 scalars per particle:
//! `[px, py, pz, vx, vy, vz, mass, charge]`. The kernel is **double-buffered**: forces
//! are read from the input state only and written to a separate output, so the result is
//! independent of invocation order (no read/write race). The pairwise force on `i` is
//! `F_i = coupling · Σ_{j≠i} q_i q_j (x_i−x_j) / (|x_i−x_j|² + soft)^{3/2}` (Plummer
//! softening, no singular skip branch), then a symplectic-Euler update
//! `v ← v + (F/m)dt`, `x ← x + v·dt`. `coupling` selects the law: `+k` electrostatic
//! (repulsive like-charges), `−G` gravitational (put mass in the charge slot).

use crate::numerical::ForgeError;

/// The N-body step kernel source (embedded from the canonical `.wgsl`).
pub const KIN_STEP_WGSL: &str = include_str!("kinematics.wgsl");
/// Entry-point name of [`KIN_STEP_WGSL`].
pub const KIN_STEP_ENTRY: &str = "nbody_step";
/// Scalars per particle in the flat state buffer.
pub const KIN_STRIDE: usize = 8;

/// Exact CPU oracle for one N-body step. Reads `state_in`, returns the new state (same
/// length), mirroring the WGSL scalar-for-scalar: forces accumulated in increasing-`j`
/// order (skipping self), `1/(r²+soft)^{3/2}` via `r2 * sqrt(r2)`, then `v` then `x`
/// using the new `v`. Particles with `mass == 0` take a zero inverse mass.
pub fn nbody_step_cpu(state_in: &[f32], dt: f32, soft: f32, coupling: f32) -> Vec<f32> {
    let count = state_in.len() / KIN_STRIDE;
    let mut out = vec![0.0f32; state_in.len()];
    for i in 0..count {
        let bi = i * KIN_STRIDE;
        let pix = state_in[bi];
        let piy = state_in[bi + 1];
        let piz = state_in[bi + 2];
        let qi = state_in[bi + 7];

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;
        for j in 0..count {
            if j == i {
                continue;
            }
            let bj = j * KIN_STRIDE;
            let rx = pix - state_in[bj];
            let ry = piy - state_in[bj + 1];
            let rz = piz - state_in[bj + 2];
            let r2 = rx * rx + ry * ry + rz * rz + soft;
            let inv = coupling * qi * state_in[bj + 7] / (r2 * r2.sqrt());
            fx += rx * inv;
            fy += ry * inv;
            fz += rz * inv;
        }

        let mass = state_in[bi + 6];
        let inv_m = if mass != 0.0 { 1.0 / mass } else { 0.0 };
        let vx = state_in[bi + 3] + fx * inv_m * dt;
        let vy = state_in[bi + 4] + fy * inv_m * dt;
        let vz = state_in[bi + 5] + fz * inv_m * dt;

        out[bi] = pix + vx * dt;
        out[bi + 1] = piy + vy * dt;
        out[bi + 2] = piz + vz * dt;
        out[bi + 3] = vx;
        out[bi + 4] = vy;
        out[bi + 5] = vz;
        out[bi + 6] = mass;
        out[bi + 7] = qi;
    }
    out
}

/// Run one N-body step on the GPU and read back the new flat state. Builds a transient
/// wgpu context, uploads `state_in` (binding 0, read), a zeroed output (binding 1,
/// read_write) and `params = [dt, soft, coupling]` (binding 2, read), dispatches one
/// invocation per particle, and reads the output back. Returns the same length as
/// `state_in`.
pub fn nbody_step_gpu(
    state_in: &[f32],
    dt: f32,
    soft: f32,
    coupling: f32,
) -> Result<Vec<f32>, ForgeError> {
    use crate::numerical::execute::{
        BindingUsage, QualiaCompute, WgpuComputeContext, WgpuPipeline,
    };
    use crate::numerical::Schedule;

    if state_in.is_empty() || state_in.len() % KIN_STRIDE != 0 {
        return Err(ForgeError::GpuValidation(format!(
            "nbody_step_gpu: state length {} is not a non-zero multiple of {KIN_STRIDE}",
            state_in.len()
        )));
    }
    let count = state_in.len() / KIN_STRIDE;
    let capacity = (state_in.len() * 8).max(4 << 20);
    let mut ctx = WgpuComputeContext::new(capacity)?;

    let view_in = ctx.allocate_and_write(
        bytemuck::cast_slice(state_in),
        0,
        0,
        BindingUsage::StorageRead,
    )?;
    let zeros = vec![0.0f32; state_in.len()];
    let view_out = ctx.allocate_and_write(
        bytemuck::cast_slice(&zeros),
        1,
        0,
        BindingUsage::StorageReadWrite,
    )?;
    let params = [dt, soft, coupling];
    let view_params = ctx.allocate_and_write(
        bytemuck::cast_slice(&params),
        2,
        0,
        BindingUsage::StorageRead,
    )?;

    let buffers = vec![view_in, view_out, view_params];
    let pipeline = WgpuPipeline::compile(&ctx, KIN_STEP_WGSL, KIN_STEP_ENTRY)?;
    let schedule = Schedule {
        workgroup_size: 64,
        ..Default::default()
    };
    pipeline.dispatch(&buffers, &schedule, count)?;
    let mut out = ctx.read_buffer_f32(&view_out)?;
    out.truncate(state_in.len());
    Ok(out)
}

#[cfg(test)]
#[path = "kinematics_tests.rs"]
mod tests;
