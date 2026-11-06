// Holographic 3D Engine for Semantic Graphs — Procedural Volumetric Light Field.
//
// Architected per `docs/Holographic 3D Engine for Semantic Graphs.md`:
// 1. Proxy Bounding Volumes: Instanced unit cuboids act as "projector lenses".
//    The GPU processes the proxy instantly with zero CPU triangle meshing.
// 2. SVG-like Procedural Shading: Resolution-independent mathematical functions
//    calculate continuous contours, telemetry fringes, and density.
// 3. Volumetric Raymarching via Signed Distance Fields (SDFs):
//    Evaluates smooth-min organic primitives (spheres, capsules, rounded boxes,
//    metaball harmonics) directly along camera rays.
// 4. Wave Interference for N3Logic Reasoning & Epistemic Honesty Lens:
//    Constructive interference generates glowing focal cores and relational
//    bridges where assertions align. Epistemic Honesty Lens maps `epistemic_q`
//    across Empirical (0.0), Theoretical (0..999), Escrow (999.0), and Eigenstate (>=1000.0).
// 5. Logarithmic Depth Buffer + ADM 3+1 Geodesics + Non-Euclidean Portal Ray Transport.
//
// @group(0) Camera + Observer standpoint · @group(1) Holographic node storage

struct Camera {
    view_projection: mat4x4<f32>,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    tensor_mode: u32,
    _padding0: vec4<f32>, // frame time, eye xyz
    _padding1: vec4<f32>, // z_far, c_scale, schwarzschild_radius, adm_enabled
    _padding2: vec4<f32>, // mass_center xyz, unused
};

struct ObserverStandpoint {
    standpoint_hash: vec2<u32>,
    session_nonce: vec2<u32>,
    epistemic_q: f32,
    t_slice: f32,
    t_window: f32,
    deontic_lane: u32,
    standpoint_class: u32,
    fabric_gate: u32,
    _padding0: vec2<f32>,
    _padding1: vec4<f32>,
    _padding2: vec4<f32>,
    _padding3: vec4<f32>,
    _padding4: vec4<f32>,
    _padding5: vec4<f32>,
};

// 96-byte std140/std430 aligned GPU representation of a semantic holographic entity.
struct HolographicNodeGpu {
    // World space position xyz, bounding sphere / proxy scale w
    position_scale: vec4<f32>,
    // Half extents xyz, primitive type w (0: Sphere, 1: RoundedBox, 2: Capsule, 3: Metaball, 4: InterferenceBridge)
    half_extents_primitive: vec4<f32>,
    // Base frequency x, amplitude y, phase z, manifold curvature w
    frequency_amplitude: vec4<f32>,
    // 10D Manifold parameters: epistemic_q x, topological_spin y, temporal_decay z, density_threshold w
    manifold_tensor: vec4<f32>,
    // Target endpoint xyz (for relational bridges), coherence weight w
    connection_endpoint: vec4<f32>,
    // Emissive tint rgba
    emissive_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> observer: ObserverStandpoint;
@group(1) @binding(0) var<storage, read> nodes: array<HolographicNodeGpu>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) node_index: u32,
};

const CUBE_VERTICES: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(-1.0, -1.0, -1.0),
    vec3<f32>( 1.0, -1.0, -1.0),
    vec3<f32>( 1.0,  1.0, -1.0),
    vec3<f32>(-1.0,  1.0, -1.0),
    vec3<f32>(-1.0, -1.0,  1.0),
    vec3<f32>( 1.0, -1.0,  1.0),
    vec3<f32>( 1.0,  1.0,  1.0),
    vec3<f32>(-1.0,  1.0,  1.0)
);

const CUBE_INDICES: array<u32, 36> = array<u32, 36>(
    0u, 1u, 2u,  0u, 2u, 3u,
    4u, 6u, 5u,  4u, 7u, 6u,
    4u, 0u, 3u,  4u, 3u, 7u,
    1u, 5u, 6u,  1u, 6u, 2u,
    3u, 2u, 6u,  3u, 6u, 7u,
    4u, 5u, 1u,  4u, 1u, 0u
);

const TWO_PI: f32 = 6.283185307;

// ── Spectral CIE 1931 XYZ -> Linear sRGB ────────────────────────────────────

fn sigma_to_cie_xyz(sigma: f32) -> vec3<f32> {
    let s = fract(sigma);
    let lambda = 380.0 + (s * 370.0);
    let x1 = 1.056 * exp(-0.5 * pow((lambda - 599.8) / 43.2, 2.0));
    let x2 = 0.362 * exp(-0.5 * pow((lambda - 442.0) / 32.0, 2.0));
    let x3 = -0.065 * exp(-0.5 * pow((lambda - 501.1) / 20.4, 2.0));
    let X = x1 + x2 + x3;
    let y1 = 0.821 * exp(-0.5 * pow((lambda - 568.8) / 46.9, 2.0));
    let y2 = 0.286 * exp(-0.5 * pow((lambda - 530.9) / 16.3, 2.0));
    let Y = y1 + y2;
    let z1 = 1.217 * exp(-0.5 * pow((lambda - 437.0) / 11.8, 2.0));
    let z2 = 0.681 * exp(-0.5 * pow((lambda - 459.0) / 26.0, 2.0));
    let Z = z1 + z2;
    return vec3<f32>(X, Y, Z);
}

fn xyz_to_linear_srgb(xyz: vec3<f32>) -> vec3<f32> {
    let R = 3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z;
    let G = -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z;
    let B = 0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z;
    return max(vec3<f32>(R, G, B), vec3<f32>(0.0));
}

fn sigma_to_linear_rgb(sigma: f32) -> vec3<f32> {
    return xyz_to_linear_srgb(sigma_to_cie_xyz(sigma));
}

// ── Logarithmic Depth Buffer (§4.2.1) ───────────────────────────────────────

fn compute_log_depth(z_view: f32, z_far: f32, c_scale: f32) -> f32 {
    let c = max(c_scale, 1e-6);
    let numer = log2(max(1e-6, 1.0 + c * z_view));
    let denom = max(log2(1.0 + c * max(z_far, 1e-6)), 1e-6);
    return clamp(numer / denom, 0.0, 1.0);
}

fn invert_log_depth(z_log: f32, z_far: f32, c_scale: f32) -> f32 {
    let c = max(c_scale, 1e-6);
    let denom = max(log2(1.0 + c * max(z_far, 1e-6)), 1e-6);
    return (exp2(z_log * denom) - 1.0) / c;
}

// ── ADM 3+1 Gravitational Lensing Geodesic Step & Portal Transport (§4.2.2) ─

struct RayState {
    pos: vec3<f32>,
    dir: vec3<f32>,
};

fn adm_geodesic_deflect_ray(
    ray_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    mass_center: vec3<f32>,
    schwarzschild_radius: f32,
    step_ds: f32
) -> RayState {
    let unit_dir = normalize(ray_dir);
    if (schwarzschild_radius <= 0.0 || step_ds <= 0.0) {
        return RayState(ray_pos + unit_dir * max(step_ds, 0.0), unit_dir);
    }
    let rel = ray_pos - mass_center;
    let r = max(length(rel), max(schwarzschild_radius * 1.5, 1e-4));
    let r3 = r * r * r;
    let rel_perp = rel - unit_dir * dot(rel, unit_dir);
    let accel = (-2.0 * schwarzschild_radius / r3) * rel_perp;
    let next_dir = normalize(unit_dir + accel * step_ds);
    let next_pos = ray_pos + next_dir * step_ds;
    return RayState(next_pos, next_dir);
}

fn portal_transport_ray(
    ray_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    portal_in_inv: mat4x4<f32>,
    portal_out: mat4x4<f32>
) -> RayState {
    let local_pos = portal_in_inv * vec4<f32>(ray_pos, 1.0);
    let out_pos = (portal_out * local_pos).xyz;
    let local_dir = portal_in_inv * vec4<f32>(ray_dir, 0.0);
    let out_dir = normalize((portal_out * local_dir).xyz);
    return RayState(out_pos, out_dir);
}

// ── Epistemic Honesty Lens (§4.3.1) ─────────────────────────────────────────

fn apply_epistemic_honesty_lens(
    base_color: vec3<f32>,
    base_alpha: f32,
    epistemic_q: f32,
    topological_spin: f32,
    local_p: vec3<f32>,
    time: f32,
    latitude_ring: f32
) -> vec4<f32> {
    if (epistemic_q <= 0.0) {
        // q == 0.0 (Empirical Ground Truth): crisp, fully opaque physical surface rendering.
        return vec4<f32>(base_color, 1.0);
    } else if (abs(epistemic_q - 999.0) < 0.5) {
        // q == 999.0 (Epistemic Escrow): amber/gold quarantine phase-grid overlay.
        let grid = max(
            smoothstep(0.06, 0.0, abs(fract(local_p.x * 6.0) - 0.5)),
            smoothstep(0.06, 0.0, abs(fract(local_p.z * 6.0) - 0.5))
        );
        let amber = vec3<f32>(1.0, 0.72, 0.12);
        let color = mix(base_color * 0.65, amber * 1.35, clamp(grid * 0.75 + 0.25, 0.0, 1.0));
        return vec4<f32>(color, clamp(base_alpha * 0.85 + grid * 0.15, 0.25, 0.95));
    } else if (epistemic_q >= 1000.0) {
        // q >= 1000.0 (Fictional / Quantum Eigenstate): iridescent wave-interference fringes modulated by topological_spin.
        let fringe_phase = dot(local_p, vec3<f32>(4.5, 6.0, 4.5)) + topological_spin * 3.14159265 + time * 2.0;
        let iridescence = sigma_to_linear_rgb(fract(fringe_phase / TWO_PI));
        let fringe = 0.5 + 0.5 * sin(fringe_phase);
        let color = mix(base_color, iridescence * 1.4, 0.6 * fringe);
        return vec4<f32>(color, clamp(base_alpha * (0.45 + 0.45 * fringe), 0.15, 0.9));
    } else {
        // 0.0 < q < 999.0 (Theoretical Hypothesis): subtle holographic wireframe/band-pass phase shimmer proportional to (1.0 - confidence).
        let confidence = clamp(1.0 / (1.0 + epistemic_q * 0.1), 0.0, 1.0);
        let uncertainty = 1.0 - confidence;
        let shimmer = 0.5 + 0.5 * sin(local_p.y * 24.0 - time * 4.0 + epistemic_q);
        let wire = clamp(latitude_ring + uncertainty * shimmer * 0.5, 0.0, 1.0);
        let color = base_color * mix(1.0, 0.75 + 0.5 * wire, uncertainty);
        let alpha = mix(1.0, clamp(base_alpha, 0.2, 0.9), uncertainty);
        return vec4<f32>(color, alpha);
    }
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let node = nodes[instance_index];
    let corner = CUBE_VERTICES[CUBE_INDICES[vertex_index]];
    let proxy_extents = node.half_extents_primitive.xyz * 1.35 + vec3<f32>(0.08);
    let world_pos = node.position_scale.xyz + corner * proxy_extents;

    var output: VertexOutput;
    output.clip_position = camera.view_projection * vec4<f32>(world_pos, 1.0);
    output.world_pos = world_pos;
    output.node_index = instance_index;
    return output;
}

// ── Ray-Box Slab Intersection ───────────────────────────────────────────────

fn ray_box_intersect(ro: vec3<f32>, rd: vec3<f32>, half_size: vec3<f32>) -> vec2<f32> {
    let inv_d = 1.0 / rd;
    let t0 = (-half_size - ro) * inv_d;
    let t1 = (half_size - ro) * inv_d;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

// ── Signed Distance Field Primitives (sdf_ops.rs GPU analogues) ─────────────

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_rounded_box(p: vec3<f32>, half_extents: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - (half_extents - vec3<f32>(r));
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// Polynomial smooth-min (C¹ continuous blending from sdf_ops::sdf_smooth_union)
fn sdf_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    if (k <= 0.0) {
        return min(d1, d2);
    }
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// Evaluates the semantic SDF inside the local node coordinate frame.
fn evaluate_node_sdf(local_p: vec3<f32>, node: HolographicNodeGpu, time: f32) -> f32 {
    let prim_type = u32(round(node.half_extents_primitive.w));
    let half_dim = node.half_extents_primitive.xyz;
    let r = min(half_dim.x, min(half_dim.y, half_dim.z)) * 0.75;

    switch (prim_type) {
        case 0u: {
            let pulse = 0.04 * sin(node.frequency_amplitude.x * time + node.frequency_amplitude.z);
            return sdf_sphere(local_p, r + pulse);
        }
        case 1u: {
            return sdf_rounded_box(local_p, half_dim, r * 0.4);
        }
        case 2u: {
            let half_h = half_dim.y;
            let a = vec3<f32>(0.0, -half_h, 0.0);
            let b = vec3<f32>(0.0,  half_h, 0.0);
            return sdf_capsule(local_p, a, b, r * 0.5);
        }
        case 3u: {
            let spin = node.manifold_tensor.y * time * 0.8;
            let c0 = vec3<f32>(sin(spin) * r * 0.35, 0.0, cos(spin) * r * 0.35);
            let c1 = vec3<f32>(-c0.x, r * 0.25, -c0.z);
            let c2 = vec3<f32>(0.0, -r * 0.2, 0.0);
            let d0 = sdf_sphere(local_p - c0, r * 0.65);
            let d1 = sdf_sphere(local_p - c1, r * 0.55);
            let d2 = sdf_sphere(local_p - c2, r * 0.70);
            let blend_k = r * 0.6;
            return sdf_smooth_union(sdf_smooth_union(d0, d1, blend_k), d2, blend_k);
        }
        default: {
            let local_target = node.connection_endpoint.xyz - node.position_scale.xyz;
            let bridge = sdf_capsule(local_p, vec3<f32>(0.0), local_target, r * 0.35);
            let core = sdf_sphere(local_p, r * 0.5);
            return sdf_smooth_union(core, bridge, r * 0.4);
        }
    }
}

// ── Surface Normal Calculation via Tetrahedron Gradients ────────────────────

fn evaluate_normal(local_p: vec3<f32>, node: HolographicNodeGpu, time: f32) -> vec3<f32> {
    let eps = 0.0035;
    let k = vec2<f32>(1.0, -1.0);
    let n = k.xyy * evaluate_node_sdf(local_p + k.xyy * eps, node, time)
          + k.yyx * evaluate_node_sdf(local_p + k.yyx * eps, node, time)
          + k.yxy * evaluate_node_sdf(local_p + k.yxy * eps, node, time)
          + k.xxx * evaluate_node_sdf(local_p + k.xxx * eps, node, time);
    return normalize(n);
}

// ── Wave Interference & Holographic Emission ────────────────────────────────

fn evaluate_interference_field(
    local_p: vec3<f32>,
    node: HolographicNodeGpu,
    time: f32
) -> f32 {
    let freq = node.frequency_amplitude.x;
    let phase = node.frequency_amplitude.z;
    let k_vec1 = vec3<f32>(freq * 1.8, freq * 2.4, freq * 1.5);
    let wave1 = cos(dot(k_vec1, local_p) - freq * time * 2.0 + phase);

    // Secondary wave driven by epistemic certainty q
    let q = clamp(node.manifold_tensor.x, 0.0, 1.0);
    let k_vec2 = vec3<f32>(-freq * 1.5, freq * 2.1, -freq * 1.9);
    let wave2 = cos(dot(k_vec2, local_p) - freq * time * 3.14 + phase * 1.618);

    // Coherence superposition: constructive alignment where assertions agree
    let superposed = (wave1 + q * wave2) / (1.0 + q);
    return superposed;
}

// Maps phase (radians) to σ in [0, 1) for spectral coloring.
fn phase_to_sigma(phase: f32) -> f32 {
    return fract(phase / TWO_PI);
}

// ── .10d PROC1 3D Cellular Voronoi Micro-Relief ─────────────────────────────

fn hash33_voronoi(p: vec3<f32>) -> vec3<f32> {
    let q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453123);
}

fn proc_voronoi_f1_3d(p: vec3<f32>, cell_density: f32, jitter: f32) -> f32 {
    let sp = p * max(cell_density, 0.001);
    let ip = floor(sp);
    let fp = fract(sp);
    let clamped_jitter = clamp(jitter, 0.0, 1.0);
    var min_dist_sq = 100.0;

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let neighbor = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let rnd = hash33_voronoi(ip + neighbor);
                let offset = mix(vec3<f32>(0.5), rnd, clamped_jitter);
                let diff = neighbor + offset - fp;
                let d_sq = dot(diff, diff);
                min_dist_sq = min(min_dist_sq, d_sq);
            }
        }
    }
    return clamp(sqrt(min_dist_sq), 0.0, 1.0);
}

struct HoloFragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fragment_main(input: VertexOutput) -> HoloFragmentOut {
    let node = nodes[input.node_index];
    let time = camera._padding0.x;
    let camera_eye = camera._padding0.yzw;

    // Ray setup in local node coordinates (with optional ADM 3+1 geodesic deflection)
    let ro_world = camera_eye;
    let rd_initial = normalize(input.world_pos - ro_world);
    let deflected = adm_geodesic_deflect_ray(
        ro_world,
        rd_initial,
        camera._padding2.xyz,
        camera._padding1.z,
        camera._padding1.w
    );
    let local_ro = ro_world - node.position_scale.xyz;
    let local_rd = deflected.dir;

    let proxy_extents = node.half_extents_primitive.xyz * 1.35 + vec3<f32>(0.08);
    let hit_bounds = ray_box_intersect(local_ro, local_rd, proxy_extents);
    let t_near = max(hit_bounds.x, 0.0);
    let t_far = hit_bounds.y;

    if (t_far <= t_near) {
        discard;
    }

    // Sphere-tracing ray march through the bounding volume
    var t = t_near;
    var hit_p = vec3<f32>(0.0);
    var hit = false;
    let max_steps = 42u;
    let threshold = 0.003;

    for (var i = 0u; i < max_steps; i = i + 1u) {
        let p = local_ro + local_rd * t;
        let d = evaluate_node_sdf(p, node, time);
        if (d < threshold) {
            hit = true;
            hit_p = p;
            break;
        }
        t = t + max(d * 0.85, 0.008);
        if (t > t_far) {
            break;
        }
    }

    if (!hit) {
        discard;
    }

    // Exact world-space hit position and logarithmic or clip-space depth
    let hit_pos = node.position_scale.xyz + hit_p;
    let clip_pos = camera.view_projection * vec4<f32>(hit_pos, 1.0);
    let z_far = select(1.0e9, camera._padding1.x, camera._padding1.x > 0.0);
    let c_scale = select(1.0, camera._padding1.y, camera._padding1.y > 0.0);
    let exact_depth = select(
        clamp(clip_pos.z / max(clip_pos.w, 1e-6), 0.0, 1.0),
        compute_log_depth(max(clip_pos.w, 0.0), z_far, c_scale),
        camera._padding1.x > 0.0
    );

    // Surface normal & Fresnel rim glow
    let normal = evaluate_normal(hit_p, node, time);
    let view_dir = -local_rd;
    let ndotv = abs(dot(normal, view_dir));
    let curvature = clamp(node.frequency_amplitude.w, 0.1, 4.0);
    let fresnel = pow(1.0 - ndotv, 2.2) * curvature;

    // N3Logic Wave interference pattern + PROC1 cellular Voronoi micro-relief
    let interference = evaluate_interference_field(hit_p, node, time);
    let constructive = max(0.0, interference);
    let voronoi_f1 = proc_voronoi_f1_3d(hit_p, max(node.frequency_amplitude.x * 2.0, 1.0), 0.85);
    let cellular_relief = mix(1.0, 0.85 + 0.30 * (1.0 - voronoi_f1), clamp(node.manifold_tensor.w, 0.0, 1.0));

    // SVG-like procedural telemetry scanlines & latitude rings
    let scanline = 0.85 + 0.15 * sin(hit_p.y * 55.0 - time * 6.0);
    let latitude_ring = smoothstep(0.04, 0.0, abs(fract(hit_p.y * 4.0) - 0.5));

    // EMF spectral emission frequency
    let sigma_val = phase_to_sigma(node.frequency_amplitude.z + constructive * 0.25 + time * 0.05);
    let spectral_rgb = sigma_to_linear_rgb(sigma_val);

    // Epistemic Honesty Lens modulation
    let epistemic_q = node.manifold_tensor.x;
    let topological_spin = node.manifold_tensor.y;
    let conf = clamp(select(1.0 / (1.0 + max(epistemic_q, 0.0) * 0.1), 1.0, epistemic_q <= 0.0), 0.0, 1.0);
    let certainty_gain = mix(0.45, 1.35, conf);
    let amplitude = node.frequency_amplitude.y;

    let base_rgb = mix(node.emissive_color.rgb, spectral_rgb, 0.65);
    let emission_hdr = base_rgb * amplitude * (0.6 + fresnel * 1.5 + latitude_ring * 0.8)
                     * (1.0 + constructive * 0.7) * scanline * certainty_gain * cellular_relief;
    let base_alpha = clamp(0.35 + fresnel * 0.65 + latitude_ring * 0.35 + constructive * 0.25, 0.15, 1.0);
    let lens_rgba = apply_epistemic_honesty_lens(
        emission_hdr,
        base_alpha,
        epistemic_q,
        topological_spin,
        hit_p,
        time,
        latitude_ring
    );

    var out: HoloFragmentOut;
    out.color = vec4<f32>(lens_rgba.rgb, lens_rgba.a * node.emissive_color.a);
    out.depth = exact_depth;
    return out;
}
