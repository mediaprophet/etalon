//! Regime II & V: Geometrically Exact Discrete Cosserat Rod Solver (`cosserat_rod.rs`).
//!
//! Models compliant axial spines, teleost fin rays, flexible tails, hydrostatic tentacles,
//! and botanical FSPM metamers with centerline positions, $SO(3)$ material frames, Darboux
//! bending/twist strain, internal hydrostatic cavity pressure $P_{\text{cav}}\pi r^2$,
//! anisotropic fluid/wind drag, gravitropic/phototropic auxin tropism, and WGSL compute kernel.

use super::SkeletalError;

pub const MAX_COSSERAT_NODES: usize = 256;

/// Centerline node state for a discrete Cosserat rod.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosseratNode {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub mass: f64,
    pub radius_m: f64,
    pub pinned: bool,
}

impl Default for CosseratNode {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            vel: [0.0; 3],
            mass: 0.05,
            radius_m: 0.02,
            pinned: false,
        }
    }
}

/// Material-frame segment state connecting node `i` to node `i + 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosseratElement {
    /// Material frame unit quaternion `[w, x, y, z]` (`d3` = local `+Z` tangent).
    pub q: [f64; 4],
    /// Material-frame angular velocity (rad/s).
    pub omega: [f64; 3],
    pub rest_len: f64,
    pub rest_darboux: [f64; 3],
    pub k_bend_twist: [f64; 3],
    pub k_shear_stretch: [f64; 3],
    /// Internal hydrostatic / turgor cavity pressure in Pa ($F_{\text{hydro}} = P_{\text{cav}} \pi r^2$).
    pub cavity_pressure_pa: f64,
    /// Auxin hormone concentration modulating differential growth tropism.
    pub auxin_concentration: f64,
    pub gravitropic_gain: f64,
    pub phototropic_gain: f64,
    pub drag_normal: f64,
    pub drag_tangent: f64,
}

impl Default for CosseratElement {
    fn default() -> Self {
        Self {
            q: [1.0, 0.0, 0.0, 0.0],
            omega: [0.0; 3],
            rest_len: 0.1,
            rest_darboux: [0.0; 3],
            k_bend_twist: [2.5, 2.5, 1.8],
            k_shear_stretch: [120.0, 120.0, 450.0],
            cavity_pressure_pa: 0.0,
            auxin_concentration: 0.0,
            gravitropic_gain: 0.0,
            phototropic_gain: 0.0,
            drag_normal: 1.2,
            drag_tangent: 0.15,
        }
    }
}

/// Environmental forcing parameters for Cosserat rods (gravity, fluid/wind, light direction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosseratEnvironment {
    pub gravity: [f64; 3],
    pub fluid_velocity: [f64; 3],
    pub fluid_density_kg_m3: f64,
    pub light_dir: [f64; 3],
    pub linear_damping: f64,
    pub angular_damping: f64,
}

impl Default for CosseratEnvironment {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.80665, 0.0],
            fluid_velocity: [0.0; 3],
            fluid_density_kg_m3: 1.225,
            light_dir: [0.0, 1.0, 0.0],
            linear_damping: 0.992,
            angular_damping: 0.985,
        }
    }
}

/// GPU-aligned 64-byte Cosserat element buffer matching `COSSERAT_ROD_WGSL`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CosseratElementGpu {
    pub pos_radius: [f32; 4],
    pub quat_wxyz: [f32; 4],
    pub k_bend_pressure: [f32; 4],
    pub rest_darboux_auxin: [f32; 4],
}

const _: [(); 64] = [(); core::mem::size_of::<CosseratElementGpu>()];

pub const COSSERAT_ROD_WGSL: &str = r#"
struct CosseratParams {
    node_count: u32,
    dt: f32,
    gravity_y: f32,
    auxin_rate: f32,
    light_dir: vec4<f32>,
};

struct CosseratElementGpu {
    pos_radius: vec4<f32>,
    quat_wxyz: vec4<f32>,
    k_bend_pressure: vec4<f32>,
    rest_darboux_auxin: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: CosseratParams;
@group(0) @binding(1) var<storage, read_write> elements: array<CosseratElementGpu>;

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(q.x, -q.y, -q.z, -q.w);
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

fn rotate_z(q: vec4<f32>) -> vec3<f32> {
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let w = q.x;
    return normalize(vec3<f32>(
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y)
    ));
}

@compute @workgroup_size(64)
fn step_cosserat_rod(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx + 1u >= params.node_count) {
        return;
    }
    var cur = elements[idx];
    let nxt = elements[idx + 1u];
    let d3 = rotate_z(cur.quat_wxyz);
    let r = max(cur.pos_radius.w, 1e-4);
    let hydro_force = cur.k_bend_pressure.w * 3.14159265 * r * r;
    let dq = quat_mul(quat_conj(cur.quat_wxyz), nxt.quat_wxyz);
    let sign_w = select(-1.0, 1.0, dq.x >= 0.0);
    let darboux = (2.0 / max(0.01, length(nxt.pos_radius.xyz - cur.pos_radius.xyz))) * sign_w * dq.yzw;
    let bend_err = darboux - cur.rest_darboux_auxin.xyz;
    let tropism = cross(d3, normalize(params.light_dir.xyz)) * cur.rest_darboux_auxin.w * params.auxin_rate;
    let delta_axis = (-cur.k_bend_pressure.xyz * bend_err + tropism) * params.dt;
    let step_q = normalize(vec4<f32>(1.0, 0.5 * delta_axis.x, 0.5 * delta_axis.y, 0.5 * delta_axis.z));
    cur.quat_wxyz = normalize(quat_mul(cur.quat_wxyz, step_q));
    if (idx > 0u) {
        cur.pos_radius = vec4<f32>(
            cur.pos_radius.xyz + d3 * (hydro_force * params.dt * 1e-4) + vec3<f32>(0.0, params.gravity_y * params.dt * params.dt, 0.0),
            r
        );
    }
    elements[idx] = cur;
}
"#;

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn normalize3_or(a: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let l = len3(a);
    if l > 1e-12 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        fallback
    }
}

#[inline]
fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

#[inline]
fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

#[inline]
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

#[inline]
fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[1], q[2], q[3]];
    let s = q[0];
    let uv = cross3(u, v);
    let uuv = cross3(u, uv);
    add3(v, add3(scale3(uv, 2.0 * s), scale3(uuv, 2.0)))
}

#[inline]
fn quat_inv_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    quat_rotate(quat_conj(q), v)
}

/// Computes material Darboux bending/twist vector $\boldsymbol{\Omega}$ between adjacent frames.
#[inline]
pub fn compute_darboux_vector(q_i: [f64; 4], q_next: [f64; 4], ds_avg: f64) -> [f64; 3] {
    let dq = quat_mul(quat_conj(quat_normalize(q_i)), quat_normalize(q_next));
    let sign = if dq[0] < 0.0 { -1.0 } else { 1.0 };
    let scale = 2.0 * sign / ds_avg.max(1e-6);
    [dq[1] * scale, dq[2] * scale, dq[3] * scale]
}

/// Computes internal hydrostatic cavity axial thrust $F_{\text{cav}} = P_{\text{cav}} \pi r^2$.
#[inline]
pub fn hydrostatic_cavity_axial_force(cavity_pressure_pa: f64, radius_m: f64) -> f64 {
    if !cavity_pressure_pa.is_finite() || !radius_m.is_finite() || radius_m <= 0.0 {
        return 0.0;
    }
    cavity_pressure_pa * std::f64::consts::PI * radius_m * radius_m
}

/// Advances a discrete Cosserat rod by `dt` seconds with zero heap allocations (`HotZeroHeap`).
pub fn step_cosserat_rod_zero_heap(
    nodes: &mut [CosseratNode],
    elements: &mut [CosseratElement],
    env: CosseratEnvironment,
    dt: f64,
) -> Result<(), SkeletalError> {
    let n_nodes = nodes.len();
    if n_nodes < 2
        || n_nodes > MAX_COSSERAT_NODES
        || elements.len() != n_nodes - 1
        || !dt.is_finite()
        || dt <= 0.0
        || env.gravity.iter().any(|v| !v.is_finite())
        || env.fluid_velocity.iter().any(|v| !v.is_finite())
        || env.light_dir.iter().any(|v| !v.is_finite())
    {
        return Err(SkeletalError::InvalidInput);
    }

    for node in nodes.iter() {
        if !node.mass.is_finite()
            || node.mass <= 0.0
            || !node.radius_m.is_finite()
            || node.radius_m <= 0.0
            || node.pos.iter().any(|v| !v.is_finite())
            || node.vel.iter().any(|v| !v.is_finite())
        {
            return Err(SkeletalError::InvalidInput);
        }
    }
    for el in elements.iter() {
        if !el.rest_len.is_finite()
            || el.rest_len <= 0.0
            || !el.cavity_pressure_pa.is_finite()
            || !el.auxin_concentration.is_finite()
            || el.q.iter().any(|v| !v.is_finite())
        {
            return Err(SkeletalError::InvalidInput);
        }
    }

    const SUBSTEPS: usize = 32;
    let sub_dt = dt / (SUBSTEPS as f64);
    let lin_damp = env
        .linear_damping
        .clamp(0.0, 1.0)
        .powf(1.0 / (SUBSTEPS as f64));
    let ang_damp = env
        .angular_damping
        .clamp(0.0, 1.0)
        .powf(1.0 / (SUBSTEPS as f64));

    let n_elem = elements.len();
    let mut node_forces = [[0.0f64; 3]; MAX_COSSERAT_NODES];
    let mut elem_torques = [[0.0f64; 3]; MAX_COSSERAT_NODES];

    let g_up = normalize3_or(scale3(env.gravity, -1.0), [0.0, 1.0, 0.0]);
    let light_u = normalize3_or(env.light_dir, [0.0, 1.0, 0.0]);

    for _sub in 0..SUBSTEPS {
        for f in node_forces[..n_nodes].iter_mut() {
            *f = [0.0; 3];
        }
        for t in elem_torques[..n_elem].iter_mut() {
            *t = [0.0; 3];
        }

        for i in 0..n_nodes {
            let fg = scale3(env.gravity, nodes[i].mass);
            node_forces[i] = add3(node_forces[i], fg);
        }

        // Internal shear-stretch forces, hydrostatic cavity thrust, fluid drag, and alignment torque.
        for i in 0..n_elem {
            let el = &mut elements[i];
            el.q = quat_normalize(el.q);
            let edge = sub3(nodes[i + 1].pos, nodes[i].pos);
            let d3_world = quat_rotate(el.q, [0.0, 0.0, 1.0]);
            let r_avg = 0.5 * (nodes[i].radius_m + nodes[i + 1].radius_m);

            // Strain Gamma = R^T (edge / ds) - e3.
            let edge_ds = scale3(edge, 1.0 / el.rest_len);
            let gamma_local = sub3(quat_inv_rotate(el.q, edge_ds), [0.0, 0.0, 1.0]);
            let f_local = [
                el.k_shear_stretch[0] * gamma_local[0],
                el.k_shear_stretch[1] * gamma_local[1],
                el.k_shear_stretch[2] * gamma_local[2],
            ];
            let f_elastic_world = quat_rotate(el.q, f_local);

            // Internal cavity pressure pushes outward along d3 on distal node and opposite on proximal.
            let f_hydro_mag = hydrostatic_cavity_axial_force(el.cavity_pressure_pa, r_avg);
            let f_hydro_world = scale3(d3_world, f_hydro_mag);

            node_forces[i] = add3(node_forces[i], sub3(f_elastic_world, f_hydro_world));
            node_forces[i + 1] = sub3(node_forces[i + 1], sub3(f_elastic_world, f_hydro_world));

            // Cosserat couple aligning material tangent d3 with centerline edge.
            let align_world = cross3(edge_ds, f_elastic_world);
            elem_torques[i] = add3(elem_torques[i], quat_inv_rotate(el.q, align_world));

            // Anisotropic hydrodynamic/aerodynamic drag.
            let v_mid = scale3(add3(nodes[i].vel, nodes[i + 1].vel), 0.5);
            let v_rel = sub3(env.fluid_velocity, v_mid);
            let v_tan_mag = dot3(v_rel, d3_world);
            let v_tan = scale3(d3_world, v_tan_mag);
            let v_norm = sub3(v_rel, v_tan);
            let area_proj = 2.0 * r_avg * el.rest_len;
            let f_drag_norm = scale3(
                v_norm,
                0.5 * env.fluid_density_kg_m3 * el.drag_normal * area_proj * len3(v_norm),
            );
            let f_drag_tan = scale3(
                v_tan,
                0.5 * env.fluid_density_kg_m3
                    * el.drag_tangent
                    * std::f64::consts::PI
                    * 2.0
                    * r_avg
                    * el.rest_len
                    * v_tan_mag.abs(),
            );
            let f_drag_half = scale3(add3(f_drag_norm, f_drag_tan), 0.5);
            node_forces[i] = add3(node_forces[i], f_drag_half);
            node_forces[i + 1] = add3(node_forces[i + 1], f_drag_half);

            // Regime V gravitropic & phototropic auxin tropism: shifts rest_darboux + active turgor cue.
            if el.auxin_concentration > 0.0
                && (el.gravitropic_gain != 0.0 || el.phototropic_gain != 0.0)
            {
                let trop_target = normalize3_or(
                    add3(
                        scale3(g_up, el.gravitropic_gain),
                        scale3(light_u, el.phototropic_gain),
                    ),
                    g_up,
                );
                let cue_world = cross3(d3_world, trop_target);
                let cue_local = quat_inv_rotate(el.q, cue_world);
                let trop_rate = el.auxin_concentration.clamp(0.0, 4.0);
                el.rest_darboux = add3(el.rest_darboux, scale3(cue_local, trop_rate * sub_dt));
                elem_torques[i] = add3(elem_torques[i], scale3(cue_local, trop_rate * 0.5));
            }
        }

        // Bending & twist Darboux moments across consecutive material frames.
        for i in 0..n_elem.saturating_sub(1) {
            let ds_avg = 0.5 * (elements[i].rest_len + elements[i + 1].rest_len);
            let omega_vec = compute_darboux_vector(elements[i].q, elements[i + 1].q, ds_avg);
            let r_avg = nodes[i + 1].radius_m;
            // Hydrostatic pressure stiffens flexural resistance proportional to P * pi * r^4.
            let p_eff = 0.5 * (elements[i].cavity_pressure_pa + elements[i + 1].cavity_pressure_pa);
            let p_stiff = (p_eff.max(0.0) * std::f64::consts::PI * r_avg.powi(4)) * 0.5;
            let k_bt = [
                elements[i].k_bend_twist[0] + p_stiff,
                elements[i].k_bend_twist[1] + p_stiff,
                elements[i].k_bend_twist[2],
            ];
            let rest_d = scale3(
                add3(elements[i].rest_darboux, elements[i + 1].rest_darboux),
                0.5,
            );
            let m_local = [
                k_bt[0] * (omega_vec[0] - rest_d[0]),
                k_bt[1] * (omega_vec[1] - rest_d[1]),
                k_bt[2] * (omega_vec[2] - rest_d[2]),
            ];
            let m_world = quat_rotate(elements[i].q, m_local);
            elem_torques[i] = add3(elem_torques[i], quat_inv_rotate(elements[i].q, m_world));
            elem_torques[i + 1] = sub3(
                elem_torques[i + 1],
                quat_inv_rotate(elements[i + 1].q, m_world),
            );
        }

        // Integrate linear velocities & positions.
        for i in 0..n_nodes {
            if nodes[i].pinned {
                nodes[i].vel = [0.0; 3];
                continue;
            }
            let inv_m = 1.0 / nodes[i].mass;
            nodes[i].vel = scale3(
                add3(nodes[i].vel, scale3(node_forces[i], inv_m * sub_dt)),
                lin_damp,
            );
            nodes[i].pos = add3(nodes[i].pos, scale3(nodes[i].vel, sub_dt));
        }

        // Integrate element rotational velocities & quaternion orientations.
        for i in 0..n_elem {
            let r_avg = 0.5 * (nodes[i].radius_m + nodes[i + 1].radius_m);
            let m_seg = 0.5 * (nodes[i].mass + nodes[i + 1].mass);
            let el = &mut elements[i];
            let inertia_rot = (m_seg * el.rest_len * el.rest_len / 12.0)
                .max(0.5 * m_seg * r_avg * r_avg)
                .max(1e-4);
            el.omega = scale3(
                add3(el.omega, scale3(elem_torques[i], sub_dt / inertia_rot)),
                ang_damp,
            );
            let half_w = [
                0.0,
                0.5 * el.omega[0] * sub_dt,
                0.5 * el.omega[1] * sub_dt,
                0.5 * el.omega[2] * sub_dt,
            ];
            let dq = quat_mul(el.q, half_w);
            el.q = quat_normalize([
                el.q[0] + dq[0],
                el.q[1] + dq[1],
                el.q[2] + dq[2],
                el.q[3] + dq[3],
            ]);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrostatic_cavity_pressure_generates_axial_extension_force() {
        let mut nodes_lo = [
            CosseratNode {
                pos: [0.0, 0.0, 0.0],
                pinned: true,
                radius_m: 0.03,
                ..Default::default()
            },
            CosseratNode {
                pos: [0.0, 0.0, 0.1],
                pinned: false,
                radius_m: 0.03,
                ..Default::default()
            },
        ];
        let mut nodes_hi = nodes_lo;
        let mut elems_lo = [CosseratElement {
            cavity_pressure_pa: 0.0,
            ..Default::default()
        }];
        let mut elems_hi = [CosseratElement {
            cavity_pressure_pa: 250_000.0,
            ..Default::default()
        }];
        let env = CosseratEnvironment {
            gravity: [0.0; 3],
            ..Default::default()
        };
        step_cosserat_rod_zero_heap(&mut nodes_lo, &mut elems_lo, env, 0.002).unwrap();
        step_cosserat_rod_zero_heap(&mut nodes_hi, &mut elems_hi, env, 0.002).unwrap();
        assert!(nodes_hi[1].pos[2] > nodes_lo[1].pos[2]);
    }

    #[test]
    fn auxin_phototropism_and_gravitropism_curve_rod_toward_cue() {
        // Horizontal cantilever along +Z with light & upward cue along +Y.
        let mut nodes = [CosseratNode::default(); 5];
        let mut elems = [CosseratElement::default(); 4];
        for i in 0..5 {
            nodes[i] = CosseratNode {
                pos: [0.0, 0.0, (i as f64) * 0.1],
                vel: [0.0; 3],
                mass: 0.04,
                radius_m: 0.015,
                pinned: i == 0,
            };
        }
        for el in elems.iter_mut() {
            el.auxin_concentration = 1.5;
            el.gravitropic_gain = 1.0;
            el.phototropic_gain = 1.0;
            el.k_bend_twist = [3.0, 3.0, 2.0];
        }
        let env = CosseratEnvironment {
            gravity: [0.0, -0.5, 0.0],
            light_dir: [0.0, 1.0, 0.0],
            ..Default::default()
        };
        for _ in 0..40 {
            step_cosserat_rod_zero_heap(&mut nodes, &mut elems, env, 0.005).unwrap();
        }
        assert!(
            nodes[4].pos[1] > 0.0,
            "tip should curve upward toward light/anti-gravity, got y={}",
            nodes[4].pos[1]
        );
    }

    #[test]
    fn cosserat_wgsl_shader_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(COSSERAT_ROD_WGSL).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
