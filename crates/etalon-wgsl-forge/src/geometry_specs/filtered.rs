//! Filtered geometry shader specifications; exact adjudication belongs to the caller.
pub const GPU_ORIENTATION_UNCERTAIN: i32 = 2;

/// GPU candidate-generation result flag: the pair is a definite overlap.
pub const GPU_OVERLAP_YES: i32 = 1;
/// GPU candidate-generation result flag: the pair is definitely not overlapping.
pub const GPU_OVERLAP_NO: i32 = 0;
/// GPU candidate-generation result flag: uncertain (near-boundary), needs CPU verification.
pub const GPU_OVERLAP_UNCERTAIN: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryGpuKernel {
    /// One invocation per packed `(a,b,c)` f32 point triple (6 f32s).
    Orientation2F32,
    /// One invocation per packed `(a,b,c,d)` f32 point quadruple (12 f32s).
    /// P1.9.
    Orient3dF32,
    /// One invocation per packed `(a,b,c,d)` f32 point quadruple (8 f32s).
    /// P1.9.
    IncircleF32,
    /// One invocation per AABB pair (12 f32s: amin\[3\], amax\[3\], bmin\[3\], bmax\[3\]).
    /// P3.6: GPU broad-phase overlap test with filtered error bound.
    AabbOverlapF32,
    /// One invocation per point-AABB pair (6 f32s: point\[3\], amin\[3\]) + 3 f32s (amax\[3\]).
    /// P3.6: GPU point-to-AABB distance squared for NN candidate filtering.
    PointAabbDistSqF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryGpuSchedule {
    pub workgroup_size: u32,
}

impl Default for GeometryGpuSchedule {
    fn default() -> Self {
        Self {
            workgroup_size: 128,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryGpuError {
    InvalidWorkgroupSize,
    InputLengthNotMultipleOfSix,
    InputLengthNotMultipleOfEight,
    InputLengthNotMultipleOfTwelve,
    InputLengthNotMultipleOfTwentyFour,
    OutputTooSmall {
        required: usize,
    },
    /// Candidate buffer too small for GPU-generated candidates.
    CandidateBufferTooSmall {
        required: usize,
    },
}

/// Deterministically emit a typed computational-geometry shader.
pub fn emit_geometry_wgsl(
    kernel: GeometryGpuKernel,
    schedule: GeometryGpuSchedule,
) -> Result<String, GeometryGpuError> {
    if !(32..=256).contains(&schedule.workgroup_size) || !schedule.workgroup_size.is_power_of_two()
    {
        return Err(GeometryGpuError::InvalidWorkgroupSize);
    }
    match kernel {
        GeometryGpuKernel::Orientation2F32 => Ok(format!(
            r#"// QualiaDB computational geometry: orientation_2_f32 v1
struct Params {{
    triple_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0)
var<storage, read> points: array<f32>;

@group(0) @binding(1)
var<storage, read_write> orientation: array<i32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn orientation_2_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.triple_count) {{
        return;
    }}
    let base = i * 6u;
    let ax = points[base];
    let ay = points[base + 1u];
    let bx = points[base + 2u];
    let by = points[base + 3u];
    let cx = points[base + 4u];
    let cy = points[base + 5u];
    let left = (ax - cx) * (by - cy);
    let right = (ay - cy) * (bx - cx);
    let det = left - right;
    // Filter only. Near-degenerate triples are resolved by the exact CPU/WASM oracle.
    let error_bound = (abs(left) + abs(right)) * 9.5367431640625e-7;
    if (abs(det) <= error_bound) {{
        orientation[i] = {uncertain};
    }} else if (det > 0.0) {{
        orientation[i] = 1;
    }} else {{
        orientation[i] = -1;
    }}
}}
"#,
            workgroup_size = schedule.workgroup_size,
            uncertain = GPU_ORIENTATION_UNCERTAIN,
        )),

        GeometryGpuKernel::Orient3dF32 => Ok(format!(
            r#"// QualiaDB computational geometry: orient_3d_f32 v1
struct Params {{
    quad_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0)
var<storage, read> points: array<f32>;

@group(0) @binding(1)
var<storage, read_write> orient: array<i32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn orient_3d_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.quad_count) {{
        return;
    }}
    let base = i * 12u;
    let ax = points[base];
    let ay = points[base + 1u];
    let az = points[base + 2u];
    let bx = points[base + 3u];
    let by = points[base + 4u];
    let bz = points[base + 5u];
    let cx = points[base + 6u];
    let cy = points[base + 7u];
    let cz = points[base + 8u];
    let dx = points[base + 9u];
    let dy = points[base + 10u];
    let dz = points[base + 11u];
    // det = (b-a) · ((c-a) × (d-a))
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let acx = cx - ax;
    let acy = cy - ay;
    let acz = cz - az;
    let adx = dx - ax;
    let ady = dy - ay;
    let adz = dz - az;
    // cross = (c-a) × (d-a)
    let cx0 = acy * adz - acz * ady;
    let cy0 = acz * adx - acx * adz;
    let cz0 = acx * ady - acy * adx;
    // dot = (b-a) · cross
    let det = abx * cx0 + aby * cy0 + abz * cz0;
    // Filtered error bound: sum of absolute products × f32 epsilon.
    let perm = abs(abx) * (abs(acy) * abs(adz) + abs(acz) * abs(ady))
             + abs(aby) * (abs(acx) * abs(adz) + abs(acz) * abs(adx))
             + abs(abz) * (abs(acx) * abs(ady) + abs(acy) * abs(adx));
    let error_bound = perm * 1.52587890625e-5;
    if (abs(det) <= error_bound) {{
        orient[i] = {uncertain};
    }} else if (det > 0.0) {{
        orient[i] = 1;
    }} else {{
        orient[i] = -1;
    }}
}}
"#,
            workgroup_size = schedule.workgroup_size,
            uncertain = GPU_ORIENTATION_UNCERTAIN,
        )),

        GeometryGpuKernel::IncircleF32 => Ok(format!(
            r#"// QualiaDB computational geometry: incircle_f32 v1
struct Params {{
    quad_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0)
var<storage, read> points: array<f32>;

@group(0) @binding(1)
var<storage, read_write> incircle_out: array<i32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn incircle_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.quad_count) {{
        return;
    }}
    let base = i * 8u;
    let ax = points[base];
    let ay = points[base + 1u];
    let bx = points[base + 2u];
    let by = points[base + 3u];
    let cx = points[base + 4u];
    let cy = points[base + 5u];
    let dx = points[base + 6u];
    let dy = points[base + 7u];
    // Translate by d
    let adx = ax - dx;
    let ady = ay - dy;
    let bdx = bx - dx;
    let bdy = by - dy;
    let cdx = cx - dx;
    let cdy = cy - dy;
    let ad2 = adx * adx + ady * ady;
    let bd2 = bdx * bdx + bdy * bdy;
    let cd2 = cdx * cdx + cdy * cdy;
    // det = adx*(bdy*cd2 - cdy*bd2) - ady*(bdx*cd2 - cdx*bd2) + ad2*(bdx*cdy - bdy*cdx)
    let m1 = bdy * cd2 - cdy * bd2;
    let m2 = bdx * cd2 - cdx * bd2;
    let m3 = bdx * cdy - bdy * cdx;
    let det = adx * m1 - ady * m2 + ad2 * m3;
    // Filtered error bound
    let perm = abs(adx) * (abs(bdy) * abs(cd2) + abs(cdy) * abs(bd2))
             + abs(ady) * (abs(bdx) * abs(cd2) + abs(cdx) * abs(bd2))
             + abs(ad2) * (abs(bdx) * abs(cdy) + abs(bdy) * abs(cdx));
    let error_bound = perm * 3.0517578125e-5;
    if (abs(det) <= error_bound) {{
        incircle_out[i] = {uncertain};
    }} else if (det > 0.0) {{
        incircle_out[i] = 1;
    }} else {{
        incircle_out[i] = -1;
    }}
}}
"#,
            workgroup_size = schedule.workgroup_size,
            uncertain = GPU_ORIENTATION_UNCERTAIN,
        )),

        GeometryGpuKernel::AabbOverlapF32 => Ok(format!(
            r#"// QualiaDB computational geometry: aabb_overlap_f32 v1
struct Params {{
    pair_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0)
var<storage, read> pairs: array<f32>;

@group(0) @binding(1)
var<storage, read_write> overlap_out: array<i32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn aabb_overlap_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.pair_count) {{
        return;
    }}
    let base = i * 12u;
    let aminx = pairs[base];      let aminy = pairs[base + 1u];  let aminz = pairs[base + 2u];
    let amaxx = pairs[base + 3u]; let amaxy = pairs[base + 4u];  let amaxz = pairs[base + 5u];
    let bminx = pairs[base + 6u]; let bminy = pairs[base + 7u];  let bminz = pairs[base + 8u];
    let bmaxx = pairs[base + 9u]; let bmaxy = pairs[base + 10u]; let bmaxz = pairs[base + 11u];
    // Overlap test: amin <= bmax && bmin <= amax on all axes.
    let dx1 = bmaxx - aminx;
    let dx2 = amaxx - bminx;
    let dy1 = bmaxy - aminy;
    let dy2 = amaxy - bminy;
    let dz1 = bmaxz - aminz;
    let dz2 = amaxz - bminz;
    // Filtered: if any gap is near zero, flag uncertain.
    let eps = 1.0e-5;
    let min_gap = min(min(min(dx1, dx2), min(dy1, dy2)), min(dz1, dz2));
    if (min_gap < 0.0) {{
        overlap_out[i] = {overlap_no};
    }} else if (min_gap < eps) {{
        overlap_out[i] = {overlap_uncertain};
    }} else {{
        overlap_out[i] = {overlap_yes};
    }}
}}
"#,
            workgroup_size = schedule.workgroup_size,
            overlap_no = GPU_OVERLAP_NO,
            overlap_uncertain = GPU_OVERLAP_UNCERTAIN,
            overlap_yes = GPU_OVERLAP_YES,
        )),

        GeometryGpuKernel::PointAabbDistSqF32 => Ok(format!(
            r#"// QualiaDB computational geometry: point_aabb_dist_sq_f32 v1
struct Params {{
    query_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0)
var<storage, read> data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dist_sq_out: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn point_aabb_dist_sq_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.query_count) {{
        return;
    }}
    let base = i * 6u;
    let px = data[base];     let py = data[base + 1u]; let pz = data[base + 2u];
    let minx = data[base + 3u]; let miny = data[base + 4u]; let minz = data[base + 5u];
    // Clamped distance: max(0, min - p) and max(0, p - max) per axis.
    // We only have min here; for a proper distance we need max too.
    // This kernel is a broad-phase filter: it computes the distance to
    // the AABB's min corner as a lower bound. The CPU merge does the
    // exact point-to-AABB distance.
    let dx = max(0.0, minx - px);
    let dy = max(0.0, miny - py);
    let dz = max(0.0, minz - pz);
    dist_sq_out[i] = dx * dx + dy * dy + dz * dz;
}}
"#,
            workgroup_size = schedule.workgroup_size,
        )),
    }
}
