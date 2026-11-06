//! Tetrahedral XPBD Volume Conservation & Neo-Hookean Hydrostatic Strain (§7.3.1).
//!
//! Provides the certified WGSL compute kernel [`XPBD_VOLUME_CONSTRAINT_WGSL`], the exact
//! zero-heap CPU reference oracle [`xpbd_volume_step_cpu`], single-tetrahedron solver
//! [`solve_tet_volume_constraint`], and GPU execution harness [`xpbd_volume_step_gpu`].
//!
//! Flat state layout (`XPBD_TET_STRIDE = 18` `f32` scalars per independent tetrahedron):
//! `[x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3, w0,w1,w2,w3, rest_volume, lambda]`.

use crate::numerical::ForgeError;

/// WGSL compute kernel source for tetrahedral XPBD volume conservation.
pub const XPBD_VOLUME_CONSTRAINT_WGSL: &str = r#"// Tetrahedral XPBD Volume Conservation & Neo-Hookean Hydrostatic Strain.
//
// Each tetrahedron occupies 18 contiguous f32 scalars in `tets`:
//   [0..3]   x0 (vec3)
//   [3..6]   x1 (vec3)
//   [6..9]   x2 (vec3)
//   [9..12]  x3 (vec3)
//   [12..16] w0, w1, w2, w3 (inverse masses)
//   [16]     rest_volume (V_rest)
//   [17]     lambda (Lagrange multiplier accumulator)
//
// `params[0]` = alpha_tilde = alpha / (dt * dt), where alpha = 1 / (K * V_rest)
// for Neo-Hookean bulk modulus K.

@group(0) @binding(0) var<storage, read_write> tets: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;

@compute @workgroup_size(64)
fn xpbd_volume_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = 18u;
    let count = arrayLength(&tets) / stride;
    let i = gid.x;
    if (i >= count) {
        return;
    }

    let base = i * stride;
    let x0 = vec3<f32>(tets[base + 0u], tets[base + 1u], tets[base + 2u]);
    let x1 = vec3<f32>(tets[base + 3u], tets[base + 4u], tets[base + 5u]);
    let x2 = vec3<f32>(tets[base + 6u], tets[base + 7u], tets[base + 8u]);
    let x3 = vec3<f32>(tets[base + 9u], tets[base + 10u], tets[base + 11u]);

    let w0 = tets[base + 12u];
    let w1 = tets[base + 13u];
    let w2 = tets[base + 14u];
    let w3 = tets[base + 15u];
    let v_rest = tets[base + 16u];
    let lambda_prev = tets[base + 17u];
    let alpha_tilde = max(params[0], 0.0);

    let e1 = x1 - x0;
    let e2 = x2 - x0;
    let e3 = x3 - x0;

    let one_sixth = 1.0 / 6.0;
    let g1 = cross(e2, e3) * one_sixth;
    let g2 = cross(e3, e1) * one_sixth;
    let g3 = cross(e1, e2) * one_sixth;
    let g0 = -(g1 + g2 + g3);

    let volume = dot(g3, e3);
    let c_vol = volume - v_rest;

    let w_sum = w0 * dot(g0, g0)
              + w1 * dot(g1, g1)
              + w2 * dot(g2, g2)
              + w3 * dot(g3, g3)
              + alpha_tilde;

    let d_lambda = select(
        0.0,
        (-c_vol - alpha_tilde * lambda_prev) / w_sum,
        w_sum > 1e-12
    );

    let nx0 = x0 + g0 * (w0 * d_lambda);
    let nx1 = x1 + g1 * (w1 * d_lambda);
    let nx2 = x2 + g2 * (w2 * d_lambda);
    let nx3 = x3 + g3 * (w3 * d_lambda);

    tets[base + 0u] = nx0.x;
    tets[base + 1u] = nx0.y;
    tets[base + 2u] = nx0.z;
    tets[base + 3u] = nx1.x;
    tets[base + 4u] = nx1.y;
    tets[base + 5u] = nx1.z;
    tets[base + 6u] = nx2.x;
    tets[base + 7u] = nx2.y;
    tets[base + 8u] = nx2.z;
    tets[base + 9u] = nx3.x;
    tets[base + 10u] = nx3.y;
    tets[base + 11u] = nx3.z;
    tets[base + 17u] = lambda_prev + d_lambda;
}
"#;

/// Entry-point name of [`XPBD_VOLUME_CONSTRAINT_WGSL`].
pub const XPBD_VOLUME_CONSTRAINT_ENTRY: &str = "xpbd_volume_step";
/// Scalars per tetrahedron in the flat XPBD state buffer.
pub const XPBD_TET_STRIDE: usize = 18;

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Computes the signed volume of a tetrahedron:
/// $$V(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3) = \frac{1}{6}\det[\mathbf{x}_1 - \mathbf{x}_0, \mathbf{x}_2 - \mathbf{x}_0, \mathbf{x}_3 - \mathbf{x}_0]$$
#[inline]
pub fn tet_signed_volume(
    x0: [f32; 3],
    x1: [f32; 3],
    x2: [f32; 3],
    x3: [f32; 3],
) -> f32 {
    let e1 = sub3(x1, x0);
    let e2 = sub3(x2, x0);
    let e3 = sub3(x3, x0);
    dot3(cross3(e1, e2), e3) * (1.0 / 6.0)
}

/// Computes the analytical volume gradients $[\nabla_{\mathbf{x}_0} C_{\text{vol}}, \nabla_{\mathbf{x}_1} C_{\text{vol}}, \nabla_{\mathbf{x}_2} C_{\text{vol}}, \nabla_{\mathbf{x}_3} C_{\text{vol}}]$.
#[inline]
pub fn tet_volume_gradients(
    x0: [f32; 3],
    x1: [f32; 3],
    x2: [f32; 3],
    x3: [f32; 3],
) -> [[f32; 3]; 4] {
    let e1 = sub3(x1, x0);
    let e2 = sub3(x2, x0);
    let e3 = sub3(x3, x0);
    let one_sixth = 1.0 / 6.0;
    let g1 = scale3(cross3(e2, e3), one_sixth);
    let g2 = scale3(cross3(e3, e1), one_sixth);
    let g3 = scale3(cross3(e1, e2), one_sixth);
    let g0 = [
        -(g1[0] + g2[0] + g3[0]),
        -(g1[1] + g2[1] + g3[1]),
        -(g1[2] + g2[2] + g3[2]),
    ];
    [g0, g1, g2, g3]
}

/// Computes the time-scaled XPBD compliance $\tilde{\alpha} = \frac{\alpha}{\Delta t^2}$ for a
/// Neo-Hookean hydrostatic bulk modulus $\kappa$ and rest volume $V_{\text{rest}}$.
#[inline]
pub fn neo_hookean_hydrostatic_compliance_tilde(
    bulk_modulus: f32,
    rest_volume: f32,
    dt: f32,
) -> f32 {
    let denom = bulk_modulus.max(1.0e-12) * rest_volume.abs().max(1.0e-12) * (dt * dt).max(1.0e-12);
    1.0 / denom
}

/// Solves a single XPBD tetrahedral volume constraint step in place (zero-heap).
/// Returns $\Delta \lambda$.
#[inline]
pub fn solve_tet_volume_constraint(
    positions: &mut [[f32; 3]; 4],
    inv_masses: [f32; 4],
    rest_volume: f32,
    alpha_tilde: f32,
    lambda: &mut f32,
) -> f32 {
    let alpha_t = alpha_tilde.max(0.0);
    let grads = tet_volume_gradients(positions[0], positions[1], positions[2], positions[3]);
    let e3 = sub3(positions[3], positions[0]);
    let volume = dot3(grads[3], e3);
    let c_vol = volume - rest_volume;

    let w_sum = inv_masses[0] * dot3(grads[0], grads[0])
        + inv_masses[1] * dot3(grads[1], grads[1])
        + inv_masses[2] * dot3(grads[2], grads[2])
        + inv_masses[3] * dot3(grads[3], grads[3])
        + alpha_t;

    let d_lambda = if w_sum > 1.0e-12 {
        (-c_vol - alpha_t * (*lambda)) / w_sum
    } else {
        0.0
    };

    for i in 0..4 {
        let s = inv_masses[i] * d_lambda;
        positions[i][0] += grads[i][0] * s;
        positions[i][1] += grads[i][1] * s;
        positions[i][2] += grads[i][2] * s;
    }
    *lambda += d_lambda;
    d_lambda
}

/// Exact CPU oracle for a batch of tetrahedral XPBD volume constraints (`XPBD_TET_STRIDE = 18`).
pub fn xpbd_volume_step_cpu(tets: &mut [f32], alpha_tilde: f32) {
    let count = tets.len() / XPBD_TET_STRIDE;
    for i in 0..count {
        let base = i * XPBD_TET_STRIDE;
        let mut pos = [
            [tets[base], tets[base + 1], tets[base + 2]],
            [tets[base + 3], tets[base + 4], tets[base + 5]],
            [tets[base + 6], tets[base + 7], tets[base + 8]],
            [tets[base + 9], tets[base + 10], tets[base + 11]],
        ];
        let inv_masses = [
            tets[base + 12],
            tets[base + 13],
            tets[base + 14],
            tets[base + 15],
        ];
        let v_rest = tets[base + 16];
        let mut lambda = tets[base + 17];

        solve_tet_volume_constraint(&mut pos, inv_masses, v_rest, alpha_tilde, &mut lambda);

        for v in 0..4 {
            tets[base + v * 3] = pos[v][0];
            tets[base + v * 3 + 1] = pos[v][1];
            tets[base + v * 3 + 2] = pos[v][2];
        }
        tets[base + 17] = lambda;
    }
}

/// Dispatches one XPBD volume constraint pass on the GPU and returns the updated buffer.
pub fn xpbd_volume_step_gpu(tets: &[f32], alpha_tilde: f32) -> Result<Vec<f32>, ForgeError> {
    use crate::numerical::execute::{
        BindingUsage, QualiaCompute, WgpuComputeContext, WgpuPipeline,
    };
    use crate::numerical::Schedule;

    if tets.is_empty() || tets.len() % XPBD_TET_STRIDE != 0 {
        return Err(ForgeError::GpuValidation(format!(
            "xpbd_volume_step_gpu: buffer length {} is not a non-zero multiple of {XPBD_TET_STRIDE}",
            tets.len()
        )));
    }
    let count = tets.len() / XPBD_TET_STRIDE;
    let capacity = (tets.len() * 4).max(4 << 20);
    let mut ctx = WgpuComputeContext::new(capacity)?;

    let view_tets = ctx.allocate_and_write(
        bytemuck::cast_slice(tets),
        0,
        0,
        BindingUsage::StorageReadWrite,
    )?;
    let params = [alpha_tilde, 0.0, 0.0, 0.0];
    let view_params = ctx.allocate_and_write(
        bytemuck::cast_slice(&params),
        1,
        0,
        BindingUsage::StorageRead,
    )?;

    let buffers = vec![view_tets, view_params];
    let pipeline = WgpuPipeline::compile(&ctx, XPBD_VOLUME_CONSTRAINT_WGSL, XPBD_VOLUME_CONSTRAINT_ENTRY)?;
    let schedule = Schedule {
        workgroup_size: 64,
        ..Default::default()
    };
    pipeline.dispatch(&buffers, &schedule, count)?;
    let mut out = ctx.read_buffer_f32(&view_tets)?;
    out.truncate(tets.len());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xpbd_volume_wgsl_validates() {
        let report = crate::validate_wgsl(XPBD_VOLUME_CONSTRAINT_WGSL)
            .expect("XPBD volume WGSL must naga-validate");
        assert!(
            report.entry_points.iter().any(|e| e == XPBD_VOLUME_CONSTRAINT_ENTRY),
            "expected entry point {XPBD_VOLUME_CONSTRAINT_ENTRY}, got {:?}",
            report.entry_points
        );
        assert_eq!(report.binding_count, 2);
    }

    #[test]
    fn compressed_and_stretched_tetrahedra_relax_toward_rest_volume() {
        let rest_pos = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let v_rest = tet_signed_volume(rest_pos[0], rest_pos[1], rest_pos[2], rest_pos[3]);
        assert!((v_rest - 1.0 / 6.0).abs() < 1.0e-6);

        // 1. Compressed tetrahedron (scaled by 0.7 -> V < V_rest)
        let mut compressed = rest_pos.map(|p| [p[0] * 0.7, p[1] * 0.7, p[2] * 0.7]);
        let mut lambda_c = 0.0;
        let v_init_c = tet_signed_volume(compressed[0], compressed[1], compressed[2], compressed[3]);
        assert!(v_init_c < v_rest);
        for _ in 0..8 {
            solve_tet_volume_constraint(&mut compressed, [1.0; 4], v_rest, 1.0e-6, &mut lambda_c);
        }
        let v_final_c = tet_signed_volume(compressed[0], compressed[1], compressed[2], compressed[3]);
        assert!(
            (v_final_c - v_rest).abs() < 1.0e-5,
            "compressed tet did not relax to rest volume: {v_final_c} vs {v_rest}"
        );

        // 2. Stretched tetrahedron (scaled by 1.4 -> V > V_rest)
        let mut stretched = rest_pos.map(|p| [p[0] * 1.4, p[1] * 1.4, p[2] * 1.4]);
        let mut lambda_s = 0.0;
        let v_init_s = tet_signed_volume(stretched[0], stretched[1], stretched[2], stretched[3]);
        assert!(v_init_s > v_rest);
        for _ in 0..8 {
            solve_tet_volume_constraint(&mut stretched, [1.0; 4], v_rest, 1.0e-6, &mut lambda_s);
        }
        let v_final_s = tet_signed_volume(stretched[0], stretched[1], stretched[2], stretched[3]);
        assert!(
            (v_final_s - v_rest).abs() < 1.0e-5,
            "stretched tet did not relax to rest volume: {v_final_s} vs {v_rest}"
        );
    }

    #[test]
    #[serial_test::serial(gpu)]
    fn xpbd_volume_gpu_matches_cpu_oracle() {
        if !crate::numerical::test_gpu_available() {
            return;
        }
        let mut state = vec![
            0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2, 1.0, 1.0, 1.0, 1.0,
            0.16666667, 0.0,
        ];
        let gpu_out = xpbd_volume_step_gpu(&state, 1.0e-4).expect("xpbd_volume_step_gpu");
        xpbd_volume_step_cpu(&mut state, 1.0e-4);
        for (g, e) in gpu_out.iter().zip(state.iter()) {
            assert!((g - e).abs() <= 1.0e-4, "GPU/CPU mismatch: {g} vs {e}");
        }
    }
}
