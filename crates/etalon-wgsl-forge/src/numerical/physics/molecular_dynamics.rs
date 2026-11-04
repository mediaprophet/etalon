//! Velocity-Verlet molecular-dynamics step as a certified forge kernel.
//!
//! Embeds [`shaders/molecular_dynamics.wgsl`](molecular_dynamics.wgsl)
//! via `include_str!` (single source of truth), grades it against the exact CPU oracle
//! [`md_step_cpu`], and runs it on any wgpu adapter via [`md_step_gpu`].
//!
//! State is a flat `f32` buffer, 10 scalars per molecule:
//! `[px, py, pz, vx, vy, vz, fx, fy, fz, mass]`. For a force held constant over the step
//! `dt`, velocity-Verlet is the exact closed form
//! `x ← x + v·dt + ½(f/m)dt²`, `v ← v + (f/m)dt`, followed by per-axis PBC wrapping into
//! `[0, box)`. One invocation updates one molecule's own slots only, so the in-place
//! buffer is race-free and the GPU result is order-independent — exactly reproducible by
//! the oracle within f32 tolerance.

use crate::numerical::ForgeError;

/// The MD step kernel source (embedded from the canonical `.wgsl`).
pub const MD_STEP_WGSL: &str = include_str!("molecular_dynamics.wgsl");
/// Entry-point name of [`MD_STEP_WGSL`].
pub const MD_STEP_ENTRY: &str = "md_step";
/// Scalars per molecule in the flat state buffer.
pub const MD_STRIDE: usize = 10;

/// Wrap `v` into `[0, b)` for `b > 0` (PBC); a non-positive box leaves `v` untouched.
/// Matches the WGSL `wrap` exactly (`v − b·floor(v/b)`).
#[inline]
fn wrap(v: f32, b: f32) -> f32 {
    if b <= 0.0 {
        v
    } else {
        v - b * (v / b).floor()
    }
}

/// Exact CPU oracle for one MD step — mutates `state` in place, mirroring the WGSL
/// arithmetic scalar-for-scalar (including reading the old velocity for the position
/// update before overwriting it). `box_size` is the PBC box; `dt` the timestep.
/// Molecules whose trailing slot (`mass`) is `0` take a zero acceleration (`inv_m = 0`).
pub fn md_step_cpu(state: &mut [f32], box_size: [f32; 3], dt: f32) {
    let count = state.len() / MD_STRIDE;
    let half_dt2 = 0.5 * dt * dt;
    for i in 0..count {
        let base = i * MD_STRIDE;
        let mass = state[base + 9];
        let inv_m = if mass != 0.0 { 1.0 / mass } else { 0.0 };
        let ax = state[base + 6] * inv_m;
        let ay = state[base + 7] * inv_m;
        let az = state[base + 8] * inv_m;

        let px = state[base] + state[base + 3] * dt + ax * half_dt2;
        let py = state[base + 1] + state[base + 4] * dt + ay * half_dt2;
        let pz = state[base + 2] + state[base + 5] * dt + az * half_dt2;

        state[base + 3] += ax * dt;
        state[base + 4] += ay * dt;
        state[base + 5] += az * dt;

        state[base] = wrap(px, box_size[0]);
        state[base + 1] = wrap(py, box_size[1]);
        state[base + 2] = wrap(pz, box_size[2]);
    }
}

/// Run one MD step on the GPU and read back the updated flat state. Builds a transient
/// wgpu context, uploads `state` (binding 0, read_write) and `params = [box_x, box_y,
/// box_z, dt]` (binding 1, read), dispatches one invocation per molecule, and reads the
/// state buffer back. Returns the same length as `state`.
pub fn md_step_gpu(state: &[f32], box_size: [f32; 3], dt: f32) -> Result<Vec<f32>, ForgeError> {
    use crate::numerical::execute::{
        BindingUsage, QualiaCompute, WgpuComputeContext, WgpuPipeline,
    };
    use crate::numerical::Schedule;

    if state.is_empty() || state.len() % MD_STRIDE != 0 {
        return Err(ForgeError::GpuValidation(format!(
            "md_step_gpu: state length {} is not a non-zero multiple of {MD_STRIDE}",
            state.len()
        )));
    }
    let count = state.len() / MD_STRIDE;
    let capacity = (state.len() * 4).max(4 << 20);
    let mut ctx = WgpuComputeContext::new(capacity)?;

    let view_state = ctx.allocate_and_write(
        bytemuck::cast_slice(state),
        0,
        0,
        BindingUsage::StorageReadWrite,
    )?;
    let params = [box_size[0], box_size[1], box_size[2], dt];
    let view_params = ctx.allocate_and_write(
        bytemuck::cast_slice(&params),
        1,
        0,
        BindingUsage::StorageRead,
    )?;

    let buffers = vec![view_state, view_params];
    let pipeline = WgpuPipeline::compile(&ctx, MD_STEP_WGSL, MD_STEP_ENTRY)?;
    let schedule = Schedule {
        workgroup_size: 64,
        ..Default::default()
    };
    pipeline.dispatch(&buffers, &schedule, count)?;
    let mut out = ctx.read_buffer_f32(&view_state)?;
    out.truncate(state.len());
    Ok(out)
}

#[cfg(test)]
#[path = "molecular_dynamics_tests.rs"]
mod tests;
