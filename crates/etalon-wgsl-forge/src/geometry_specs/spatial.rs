//! Packed spatial shader specifications, without dispatch or CPU metric policy.
/// WGSL shader for batch pairwise distance computation.
///
/// Computes one row of the distance matrix per workgroup invocation.
pub const GPU_DISTANCE_KERNEL_WGSL: &str = r#"
// P8.7 — GPU batch pairwise distance kernel.
// Computes one row of the n×n distance matrix per invocation.

struct Points {
    data: array<f32>,  // n × 3 (x, y, z) packed
};

struct Distances {
    data: array<f32>,  // n × n distance matrix
};

struct Params {
    n: u32,
};

@group(0) @binding(0) var<storage, read> points: Points;
@group(0) @binding(1) var<storage, read_write> distances: Distances;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }

    let ai = i * 3u;
    let ax = points.data[ai];
    let ay = points.data[ai + 1u];
    let az = points.data[ai + 2u];

    for (var j = 0u; j < params.n; j = j + 1u) {
        let bj = j * 3u;
        let dx = ax - points.data[bj];
        let dy = ay - points.data[bj + 1u];
        let dz = az - points.data[bj + 2u];
        let d = sqrt(dx * dx + dy * dy + dz * dz);
        distances.data[i * params.n + j] = d;
    }
}
"#;

/// WGSL shader for batch circumradius computation.
pub const GPU_CIRCUMRADIUS_KERNEL_WGSL: &str = r#"
// P8.7 — GPU batch circumradius kernel.
// Computes circumradius for one triangle per invocation.

struct Points {
    data: array<f32>,  // n × 3 (x, y, z)
};

struct Triangles {
    data: array<u32>,  // m × 3 vertex indices
};

struct Radii {
    data: array<f32>,  // m circumradii
};

struct Params {
    n: u32,
    m: u32,
};

@group(0) @binding(0) var<storage, read> points: Points;
@group(0) @binding(1) var<storage, read> triangles: Triangles;
@group(0) @binding(2) var<storage, read_write> radii: Radii;
@group(0) @binding(3) var<uniform> params: Params;

fn dist(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return sqrt(dot(d, d));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.m) {
        return;
    }

    let ia = triangles.data[idx * 3u];
    let ib = triangles.data[idx * 3u + 1u];
    let ic = triangles.data[idx * 3u + 2u];

    let a = vec3<f32>(points.data[ia * 3u], points.data[ia * 3u + 1u], points.data[ia * 3u + 2u]);
    let b = vec3<f32>(points.data[ib * 3u], points.data[ib * 3u + 1u], points.data[ib * 3u + 2u]);
    let c = vec3<f32>(points.data[ic * 3u], points.data[ic * 3u + 1u], points.data[ic * 3u + 2u]);

    let d_ab = dist(a, b);
    let d_bc = dist(b, c);
    let d_ca = dist(c, a);

    let s = (d_ab + d_bc + d_ca) * 0.5;
    let area_sq = s * (s - d_ab) * (s - d_bc) * (s - d_ca);

    if (area_sq <= 0.0) {
        radii.data[idx] = 1e30; // Large value for degenerate triangles.
    } else {
        radii.data[idx] = (d_ab * d_bc * d_ca) / (4.0 * sqrt(area_sq));
    }
}
"#;
