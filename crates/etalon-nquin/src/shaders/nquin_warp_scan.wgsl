struct SieveParams {
    n: u32,
    field: u32,
    needle_lo: u32,
    needle_hi: u32,
}

@group(0) @binding(0) var<storage, read> quins_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> matches_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> match_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: SieveParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let base = i * 12u + params.field * 2u;
    let lo = quins_in[base];
    let hi = quins_in[base + 1u];
    if (lo == params.needle_lo && hi == params.needle_hi) {
        let slot = atomicAdd(&match_count, 1u);
        matches_out[slot] = i;
    }
}
