//! Filtered packed-f32 tetrahedron shader with caller-selected result codes.
pub fn point_in_tetra_wgsl(workgroup_size: u32, codes: [i32; 3]) -> String {
    format!(
        r#"// QualiaDB computational geometry: point_in_tetra_f32 v1
struct Params {{
    query_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0)
var<storage, read> data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> result_out: array<i32>;

@group(0) @binding(2)
var<uniform> params: Params;

// Conservative f32 orient3d: +1 / -1 confident, 0 = uncertain (|det| within bound).
fn o3d(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> i32 {{
    let ba = b - a;
    let ca = c - a;
    let da = d - a;
    let m0 = ca.y * da.z - ca.z * da.y;
    let m1 = ca.x * da.z - ca.z * da.x;
    let m2 = ca.x * da.y - ca.y * da.x;
    let det = ba.x * m0 - ba.y * m1 + ba.z * m2;
    let perm = abs(ba.x) * (abs(ca.y) * abs(da.z) + abs(ca.z) * abs(da.y))
             + abs(ba.y) * (abs(ca.x) * abs(da.z) + abs(ca.z) * abs(da.x))
             + abs(ba.z) * (abs(ca.x) * abs(da.y) + abs(ca.y) * abs(da.x));
    let bound = perm * 1.0e-4;
    if (abs(det) <= bound) {{ return 0; }}
    if (det > 0.0) {{ return 1; }}
    return -1;
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn point_in_tetra_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.query_count) {{ return; }}
    let base = i * 15u;
    let p = vec3<f32>(data[base],       data[base + 1u],  data[base + 2u]);
    let a = vec3<f32>(data[base + 3u],  data[base + 4u],  data[base + 5u]);
    let b = vec3<f32>(data[base + 6u],  data[base + 7u],  data[base + 8u]);
    let c = vec3<f32>(data[base + 9u],  data[base + 10u], data[base + 11u]);
    let d = vec3<f32>(data[base + 12u], data[base + 13u], data[base + 14u]);
    let o = o3d(a, b, c, d);
    if (o == 0) {{ result_out[i] = {uncertain}; return; }}
    let s0 = o3d(p, b, c, d);
    let s1 = o3d(a, p, c, d);
    let s2 = o3d(a, b, p, d);
    let s3 = o3d(a, b, c, p);
    if (s0 == 0 || s1 == 0 || s2 == 0 || s3 == 0) {{ result_out[i] = {uncertain}; return; }}
    if (s0 != o || s1 != o || s2 != o || s3 != o) {{ result_out[i] = {outside}; return; }}
    result_out[i] = {inside};
}}
"#,
        workgroup_size = workgroup_size,
        uncertain = codes[0],
        outside = codes[1],
        inside = codes[2],
    )
}
