struct Params {
    scale: f32,
    big_endian: u32,
    count: u32,
    padding: u32,
};

@group(0) @binding(0) var<storage, read> raw_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> power_values: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn sign_extend_i16(value: u32) -> i32 {
    return bitcast<i32>((value & 0xffffu) << 16u) >> 16;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= params.count) {
        return;
    }

    var packed = raw_words[index];
    if (params.big_endian != 0u) {
        packed = ((packed & 0x00ff00ffu) << 8u) | ((packed & 0xff00ff00u) >> 8u);
    }
    let i = sign_extend_i16(packed);
    let q = sign_extend_i16(packed >> 16u);
    let scaled_i = f32(i) * params.scale;
    let scaled_q = f32(q) * params.scale;
    power_values[index] = scaled_i * scaled_i + scaled_q * scaled_q;
}
