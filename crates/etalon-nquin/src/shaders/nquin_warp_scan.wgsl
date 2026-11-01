// 100% coalesced 128-byte cache-line loads + zero-bank-conflict shared memory transpose
// for 48-byte Super-Quins (6 x u64 = 12 x u32 = 3 x vec4<u32>).

struct SieveParams {
    n: u32,
    field: u32,
    needle_lo: u32,
    needle_hi: u32,
}

@group(0) @binding(0) var<storage, read> arena_uvec4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> matches_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> match_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: SieveParams;

// 12 x u32 per NQuin + 1 u32 bank-conflict pad = 13 u32 stride (gcd(13, 32) == 1)
var<workgroup> smem_quin: array<array<u32, 13>, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x; // 0..31
    smem_quin[t][12] = 0u;
    let base_quin = wid.x * 32u;
    let base_vec4 = base_quin * 3u;
    let total_vec4 = params.n * 3u;

    // 3 waves of 32 contiguous vec4<u32> loads (512B per wave = 4 exact 128B cache lines)
    for (var w: u32 = 0u; w < 3u; w = w + 1u) {
        let vec_idx = w * 32u + t;             // 0..95
        let quin_slot = vec_idx / 3u;          // 0..31
        let vec_in_quin = (vec_idx % 3u) * 4u; // 0, 4, or 8
        let global_vec_idx = base_vec4 + vec_idx;
        var v = vec4<u32>(0u, 0u, 0u, 0u);
        if (global_vec_idx < total_vec4) {
            v = arena_uvec4[global_vec_idx];
        }
        smem_quin[quin_slot][vec_in_quin + 0u] = v.x;
        smem_quin[quin_slot][vec_in_quin + 1u] = v.y;
        smem_quin[quin_slot][vec_in_quin + 2u] = v.z;
        smem_quin[quin_slot][vec_in_quin + 3u] = v.w;
    }
    workgroupBarrier();

    let quin_idx = base_quin + t;
    if (quin_idx >= params.n) {
        return;
    }

    // Bank-conflict-free per-thread NQuin equality evaluation:
    // field 0 = Subject (u32[0..1]), 1 = Predicate (u32[2..3]),
    // field 2 = Object  (u32[4..5]), 3 = Context   (u32[6..7]).
    let field_base = (params.field & 3u) * 2u;
    let lo = smem_quin[t][field_base];
    let hi = smem_quin[t][field_base + 1u];

    if (lo == params.needle_lo && hi == params.needle_hi) {
        let out_slot = atomicAdd(&match_count, 1u);
        matches_out[out_slot] = quin_idx;
    }
}
