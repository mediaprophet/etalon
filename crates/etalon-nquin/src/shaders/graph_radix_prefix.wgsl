struct PrefixParams {
    total_bins: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> hist_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets_out: array<u32>;
@group(0) @binding(2) var<uniform> params: PrefixParams;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x != 0u) {
        return;
    }
    var sum = 0u;
    for (var i = 0u; i < params.total_bins; i = i + 1u) {
        offsets_out[i] = sum;
        sum = sum + hist_in[i];
    }
}
