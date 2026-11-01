struct PrefixParams {
    total_bins: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> hist_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets_out: array<u32>;
@group(0) @binding(2) var<uniform> params: PrefixParams;

var<workgroup> s_data: array<u32, 64>;
var<workgroup> s_carry: u32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    if (t == 0u) {
        s_carry = 0u;
    }
    workgroupBarrier();

    var base: u32 = 0u;
    while (base < params.total_bins) {
        let idx = base + t;
        var v = 0u;
        if (idx < params.total_bins) {
            v = hist_in[idx];
        }
        s_data[t] = v;
        workgroupBarrier();

        for (var offset: u32 = 1u; offset < 64u; offset = offset << 1u) {
            var addend = 0u;
            if (t >= offset) {
                addend = s_data[t - offset];
            }
            workgroupBarrier();
            s_data[t] = s_data[t] + addend;
            workgroupBarrier();
        }

        let chunk_excl = s_data[t] - v;
        let running_carry = s_carry;
        if (idx < params.total_bins) {
            offsets_out[idx] = running_carry + chunk_excl;
        }
        workgroupBarrier();

        if (t == 63u) {
            s_carry = running_carry + s_data[63u];
        }
        workgroupBarrier();

        base = base + 64u;
    }
}
