use crate::{ForgeError, KernelSpec, Schedule, TargetBackend};

pub(super) fn emit_iq_power(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
    target: TargetBackend,
) -> Result<(), ForgeError> {
    if schedule.items_per_invocation != 1 || schedule.vector_width != 1 {
        return Err(ForgeError::InvalidSchedule(
            "IQ power requires one scalar pair per invocation".into(),
        ));
    }
    let workgroup = schedule.effective_workgroup_size();
    if !(1..=1024).contains(&workgroup) {
        return Err(ForgeError::InvalidSchedule(
            "IQ power workgroup size must be in 1..=1024".into(),
        ));
    }
    let entry = &kernel.entry_point;
    if kernel.id == "iq-power-i16" {
        return emit_iq_power_i16(source, entry, workgroup, target);
    }
    let body = match target {
        TargetBackend::Wgsl => format!(
            r"struct IqPowerParams {{ length: u32, pad0: u32, pad1: u32, pad2: u32 }}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: IqPowerParams;
@compute @workgroup_size({workgroup})
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if index < params.length && index < 2147483648u {{
        let i = input[2u * index];
        let q = input[2u * index + 1u];
        output[index] = i * i + q * q;
    }}
}}
"
        ),
        TargetBackend::CudaC => format!(
            r#"struct IqPowerParams {{ unsigned length, pad0, pad1, pad2; }};
extern "C" __global__ void {entry}(const float* input, float* output, IqPowerParams params) {{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < params.length && index < 2147483648u) {{
        float i = input[2u * index];
        float q = input[2u * index + 1u];
        output[index] = i * i + q * q;
    }}
}}
"#
        ),
        TargetBackend::Hlsl => format!(
            r"struct IqPowerParams {{ uint length, pad0, pad1, pad2; }};
StructuredBuffer<float> input : register(t0, space0);
RWStructuredBuffer<float> output : register(u1, space0);
ConstantBuffer<IqPowerParams> params : register(b2, space0);
[numthreads({workgroup}, 1, 1)]
void {entry}(uint3 gid : SV_DispatchThreadID) {{
    uint index = gid.x;
    if (index < params.length && index < 2147483648u) {{
        float i = input[2u * index];
        float q = input[2u * index + 1u];
        output[index] = i * i + q * q;
    }}
}}
"
        ),
        TargetBackend::Msl => format!(
            r"#include <metal_stdlib>
using namespace metal;
struct IqPowerParams {{ uint length, pad0, pad1, pad2; }};
kernel void {entry}(device const float* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   constant IqPowerParams& params [[buffer(2)]],
                   uint index [[thread_position_in_grid]]) {{
    if (index < params.length && index < 2147483648u) {{
        float i = input[2u * index];
        float q = input[2u * index + 1u];
        output[index] = i * i + q * q;
    }}
}}
"
        ),
        _ => {
            return Err(ForgeError::Emission(
                "Unsupported IQ power source target".into(),
            ))
        }
    };
    source.push_str(&body);
    Ok(())
}

fn emit_iq_power_i16(
    source: &mut String,
    entry: &str,
    workgroup: u32,
    target: TargetBackend,
) -> Result<(), ForgeError> {
    let body = match target {
        TargetBackend::Wgsl => format!(
            r"struct IqPowerI16Params {{ scale: f32, big_endian: u32, count: u32, padding: u32 }}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: IqPowerI16Params;
@compute @workgroup_size({workgroup})
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if index >= params.count {{ return; }}
    var packed = input[index];
    if params.big_endian != 0u {{
        packed = ((packed & 0x00ff00ffu) << 8u) | ((packed & 0xff00ff00u) >> 8u);
    }}
    let i = bitcast<i32>((packed & 0xffffu) << 16u) >> 16;
    let q = bitcast<i32>(packed & 0xffff0000u) >> 16;
    let scaled_i = f32(i) * params.scale;
    let scaled_q = f32(q) * params.scale;
    output[index] = scaled_i * scaled_i + scaled_q * scaled_q;
}}
"
        ),
        TargetBackend::CudaC => format!(
            r#"struct IqPowerI16Params {{ float scale; unsigned big_endian, count, padding; }};
extern "C" __global__ void {entry}(const unsigned* input, float* output, IqPowerI16Params params) {{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.count) return;
    unsigned packed = input[index];
    if (params.big_endian != 0u) {{
        packed = ((packed & 0x00ff00ffu) << 8u) | ((packed & 0xff00ff00u) >> 8u);
    }}
    int i = int(packed & 0xffffu);
    int q = int(packed >> 16u);
    i -= i >= 32768 ? 65536 : 0;
    q -= q >= 32768 ? 65536 : 0;
    float scaled_i = float(i) * params.scale;
    float scaled_q = float(q) * params.scale;
    output[index] = scaled_i * scaled_i + scaled_q * scaled_q;
}}
"#
        ),
        TargetBackend::Hlsl => format!(
            r"struct IqPowerI16Params {{ float scale; uint big_endian, count, padding; }};
StructuredBuffer<uint> input : register(t0, space0);
RWStructuredBuffer<float> output : register(u1, space0);
ConstantBuffer<IqPowerI16Params> params : register(b2, space0);
[numthreads({workgroup}, 1, 1)]
void {entry}(uint3 gid : SV_DispatchThreadID) {{
    uint index = gid.x;
    if (index >= params.count) return;
    uint packed = input[index];
    if (params.big_endian != 0u) {{
        packed = ((packed & 0x00ff00ffu) << 8u) | ((packed & 0xff00ff00u) >> 8u);
    }}
    int i = int(packed & 0xffffu);
    int q = int(packed >> 16u);
    i -= i >= 32768 ? 65536 : 0;
    q -= q >= 32768 ? 65536 : 0;
    float scaled_i = float(i) * params.scale;
    float scaled_q = float(q) * params.scale;
    output[index] = scaled_i * scaled_i + scaled_q * scaled_q;
}}
"
        ),
        TargetBackend::Msl => format!(
            r"#include <metal_stdlib>
using namespace metal;
struct IqPowerI16Params {{ float scale; uint big_endian, count, padding; }};
kernel void {entry}(device const uint* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   constant IqPowerI16Params& params [[buffer(2)]],
                   uint index [[thread_position_in_grid]]) {{
    if (index >= params.count) return;
    uint packed = input[index];
    if (params.big_endian != 0u) {{
        packed = ((packed & 0x00ff00ffu) << 8u) | ((packed & 0xff00ff00u) >> 8u);
    }}
    int i = int(packed & 0xffffu);
    int q = int(packed >> 16u);
    i -= i >= 32768 ? 65536 : 0;
    q -= q >= 32768 ? 65536 : 0;
    float scaled_i = float(i) * params.scale;
    float scaled_q = float(q) * params.scale;
    output[index] = scaled_i * scaled_i + scaled_q * scaled_q;
}}
"
        ),
        _ => {
            return Err(ForgeError::Emission(
                "unsupported signed-i16 IQ power target".into(),
            ))
        }
    };
    source.push_str(&body);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{emit_shader, BuiltinKernel, Schedule, TargetBackend};

    #[test]
    fn signed_iq_sources_preserve_packed_contract() {
        let spec = BuiltinKernel::IqPowerI16.spec();
        for target in [
            TargetBackend::Wgsl,
            TargetBackend::CudaC,
            TargetBackend::Hlsl,
            TargetBackend::Msl,
        ] {
            let shader = emit_shader(&spec, Schedule::default(), target).unwrap();
            assert!(shader.source.contains("big_endian"));
            assert!(shader.source.contains("0x00ff00ffu"));
            assert!(shader.source.contains("params.scale"));
        }
        let mut wrong_input = spec.clone();
        wrong_input.buffers[0].element = crate::BufferElement::Scalar(crate::ScalarType::F32);
        assert!(wrong_input.validate().is_err());
        let mut wrong_params = spec;
        wrong_params.buffers[2].element = crate::BufferElement::IqPowerParams;
        assert!(wrong_params.validate().is_err());
    }
}
