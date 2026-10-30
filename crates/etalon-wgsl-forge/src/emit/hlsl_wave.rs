//! HLSL Shader Model 6.0 / 6.8+ Wave and WaveMatrix emitters.

use std::fmt::Write;

use super::GeneratedShader;
use crate::ir::{BufferAccess, BufferElement, ForgeError, KernelSpec, Op, ScalarType};
use crate::ir_builtins::BuiltinKernel;
use crate::schedule::Schedule;

/// Emits an HLSL shader module from a `KernelSpec` and `Schedule`.
pub fn emit_hlsl(kernel: &KernelSpec, schedule: Schedule) -> Result<GeneratedShader, ForgeError> {
    kernel.validate()?;
    let semantic_hash = kernel.semantic_hash()?;
    let mut source = String::with_capacity(2048);

    writeln!(
        source,
        "// HLSL emitted for {}@{}\n// Semantic hash: {}\n// Schedule: workgroup={}, items={}, vector={}",
        kernel.id, kernel.semantic_version, semantic_hash,
        schedule.workgroup_size, schedule.items_per_invocation, schedule.vector_width
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))?;

    emit_kernel_body(&mut source, kernel, schedule)?;

    let source_hash = blake3::hash(source.as_bytes()).to_hex().to_string();
    Ok(GeneratedShader {
        kernel_id: kernel.id.clone(),
        semantic_hash,
        source_hash,
        schedule,
        source,
    })
}

/// Emits a standalone HLSL kernel string from a high-level `Op` and `Schedule`.
pub fn emit_hlsl_wave_kernel(op: &Op, sched: &Schedule) -> Result<String, ForgeError> {
    match op {
        Op::MatMul { tc, .. } => {
            if sched.tile.m <= 4 {
                let spec = BuiltinKernel::Gemv.spec();
                let mut s = String::new();
                emit_gemv_wave_hlsl(&mut s, &spec, *sched)?;
                Ok(s)
            } else if *tc {
                let spec = BuiltinKernel::Gemv.spec();
                let mut s = String::new();
                emit_gemv_wavematrix_hlsl(&mut s, &spec, *sched)?;
                Ok(s)
            } else {
                Ok(emit_hlsl(&BuiltinKernel::Gemm.spec(), *sched)?.source)
            }
        }
        Op::Gemv { tc } => {
            let spec = BuiltinKernel::Gemv.spec();
            let mut s = String::new();
            if *tc && sched.tile.m > 4 {
                emit_gemv_wavematrix_hlsl(&mut s, &spec, *sched)?;
            } else {
                emit_gemv_wave_hlsl(&mut s, &spec, *sched)?;
            }
            Ok(s)
        }
        Op::Silu => {
            let spec = BuiltinKernel::FusedFfn.spec();
            let mut s = String::new();
            emit_fused_ffn_wave_hlsl(&mut s, &spec, *sched)?;
            Ok(s)
        }
        _ => Ok(emit_hlsl(&BuiltinKernel::AffineF32.spec(), *sched)?.source),
    }
}

fn emit_kernel_body(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    match kernel.id.as_str() {
        "topk" => return emit_topk_wave_hlsl(source, kernel, schedule),
        "fused-ffn" => return emit_fused_ffn_wave_hlsl(source, kernel, schedule),
        "p64-project" => return emit_p64_hlsl(source, kernel, schedule),
        "gemm" => return emit_gemm_hlsl(source, kernel, schedule),
        "gemv" => return emit_gemv_wave_hlsl(source, kernel, schedule),
        "ternary-gemv" => return emit_ternary_gemv_hlsl(source, kernel, schedule),
        "fft" => return emit_fft_hlsl(source, kernel, schedule),
        "ray-probe" => {
            return Err(ForgeError::Emission(
                "ray-query is only emitted for the WGSL target (HLSL RT uses a distinct API)"
                    .to_string(),
            ))
        }
        _ => {}
    }

    if kernel.id == "affine-f32" {
        writeln!(
            source,
            "struct AffineParams {{ uint length; float scale; float bias; uint _pad; }};\n"
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
    }

    for buffer in &kernel.buffers {
        let (type_decl, reg_type) = match (buffer.element, buffer.access) {
            (BufferElement::AffineParams, _) => ("ConstantBuffer<AffineParams>", "b"),
            (BufferElement::P64Words64, BufferAccess::StorageRead) => {
                ("StructuredBuffer<P64Words64>", "t")
            }
            (BufferElement::P64Words64, BufferAccess::StorageReadWrite) => {
                ("RWStructuredBuffer<P64Words64>", "u")
            }
            (BufferElement::Scalar(ScalarType::F32), BufferAccess::StorageRead) => {
                ("StructuredBuffer<float>", "t")
            }
            (BufferElement::Scalar(ScalarType::F32), BufferAccess::StorageReadWrite) => {
                ("RWStructuredBuffer<float>", "u")
            }
            _ => ("StructuredBuffer<float>", "t"),
        };
        writeln!(
            source,
            "{} {} : register({}{}, space{});",
            type_decl, buffer.name, reg_type, buffer.binding, buffer.group
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
    }

    writeln!(
        source,
        "\n[numthreads({}, 1, 1)]\nvoid {}(uint3 gid : SV_DispatchThreadID) {{\n    const uint ITEMS_PER_INVOCATION = {};\n    const uint VECTOR_WIDTH = {};",
        schedule.workgroup_size, kernel.entry_point, schedule.items_per_invocation, schedule.vector_width
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))?;

    if kernel.id == "affine-f32" {
        writeln!(
            source,
            "    for (uint item = 0; item < ITEMS_PER_INVOCATION; item++) {{\n        uint global_id = (gid.x * ITEMS_PER_INVOCATION + item) * VECTOR_WIDTH;"
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;

        if schedule.vector_width == 1 {
            writeln!(source, "        if (global_id < params.length) {{")
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
            emit_ops(source, &kernel.ops, "            ")?;
            writeln!(source, "        }}")
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
        } else {
            writeln!(
                source,
                "        if (global_id + {} < params.length) {{",
                schedule.vector_width - 1
            )
            .map_err(|error| ForgeError::Emission(error.to_string()))?;
            for index in 0..schedule.vector_width {
                writeln!(
                    source,
                    "            output[global_id + {index}] = input[global_id + {index}] * params.scale + params.bias;"
                )
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            writeln!(
                source,
                "        }} else {{\n            for (uint c = 0; c < VECTOR_WIDTH; c++) {{\n                uint b = global_id + c;\n                if (b < params.length) output[b] = input[b] * params.scale + params.bias;\n            }}\n        }}"
            )
            .map_err(|error| ForgeError::Emission(error.to_string()))?;
        }
        writeln!(source, "    }}\n}}").map_err(|error| ForgeError::Emission(error.to_string()))?;
    } else {
        writeln!(
            source,
            "    for (uint item = 0; item < ITEMS_PER_INVOCATION; item++) {{\n        uint global_id = gid.x * ITEMS_PER_INVOCATION + item;"
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
        emit_ops(source, &kernel.ops, "        ")?;
        writeln!(source, "    }}\n}}").map_err(|error| ForgeError::Emission(error.to_string()))?;
    }

    Ok(())
}

/// Wave-cooperative GEMV kernel (`WaveActiveSum`).
pub fn emit_gemv_wave_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        r#"struct GemvParams {{ uint m; uint n; uint _pad0; uint _pad1; }};
StructuredBuffer<float> a : register(t0, space0);
StructuredBuffer<float> x : register(t1, space0);
RWStructuredBuffer<float> y : register(u2, space0);
ConstantBuffer<GemvParams> params : register(b3, space0);

[numthreads({wg}, 1, 1)]
void {entry}(uint3 gid : SV_DispatchThreadID) {{
    uint wave_size = WaveGetLaneCount();
    uint row = gid.x / wave_size;
    if (row >= params.m) return;
    uint lane = WaveGetLaneIndex();
    uint a_row = row * params.n;
    float partial = 0.0;
    for (uint j = lane; j < params.n; j += wave_size) partial += a[a_row + j] * x[j];
    float acc = WaveActiveSum(partial);
    if (lane == 0) y[row] = acc;
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

/// Wave-cooperative fused FFN kernel (`WaveActiveSum`).
pub fn emit_fused_ffn_wave_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        r#"struct FfnParams {{ uint input_size; uint hidden_size; uint output_size; uint _pad; }};
StructuredBuffer<float> input : register(t0, space0);
StructuredBuffer<float> w1 : register(t1, space0);
StructuredBuffer<float> w2 : register(t2, space0);
RWStructuredBuffer<float> output : register(u3, space0);
ConstantBuffer<FfnParams> params : register(b4, space0);

[numthreads({wg}, 1, 1)]
void {entry}(uint3 gid : SV_DispatchThreadID) {{
    uint wave_size = WaveGetLaneCount();
    uint o = gid.x / wave_size;
    if (o >= params.output_size) return;
    uint lane = WaveGetLaneIndex();
    float partial = 0.0;
    for (uint h = lane; h < params.hidden_size; h += wave_size) {{
        float gate = 0.0; float up = 0.0;
        uint w1_row = h * params.input_size;
        uint w1_up_row = (h + params.hidden_size) * params.input_size;
        for (uint i = 0; i < params.input_size; i++) {{
            gate += w1[w1_row + i] * input[i];
            up += w1[w1_up_row + i] * input[i];
        }}
        float silu_gate = 0.5f * gate * (1.0f + tanh(0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
        partial += w2[o * params.hidden_size + h] * silu_gate * up;
    }}
    float acc = WaveActiveSum(partial);
    if (lane == 0) output[o] = acc;
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

/// Wave-cooperative top-K reduction kernel.
pub fn emit_topk_wave_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        r#"struct TopKParams {{ uint length; uint k; uint block_size; uint _pad; }};
StructuredBuffer<float> input : register(t0, space0);
RWStructuredBuffer<float> output : register(u1, space0);
ConstantBuffer<TopKParams> params : register(b2, space0);
groupshared float s_val[{wg}];
groupshared uint s_idx[{wg}];

[numthreads({wg}, 1, 1)]
void {entry}(uint tid : SV_GroupIndex, uint3 group_id : SV_GroupID) {{
    uint block = group_id.x;
    uint gidx = block * {wg}u + tid;
    float sentinel = asfloat(0xff7fffffu);
    s_val[tid] = (gidx < params.length) ? input[gidx] : sentinel;
    s_idx[tid] = tid;
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0u; i < params.k; i++) {{
        for (uint stride = {wg}u / 2u; stride > 0u; stride /= 2u) {{
            if (tid < stride && s_val[tid + stride] > s_val[tid]) {{
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }}
            GroupMemoryBarrierWithGroupSync();
        }}
        if (tid == 0u) {{ output[block * params.k + i] = s_val[0]; s_val[s_idx[0]] = sentinel; }}
        GroupMemoryBarrierWithGroupSync();
    }}
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

/// HLSL WaveMatrix GEMV using SM 6.8+ tensor-core intrinsics (`WaveMatrixMultiply`).
pub fn emit_gemv_wavematrix_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        r#"struct GemvParams {{ uint m; uint n; uint _pad0; uint _pad1; }};
RWByteAddressBuffer a : register(u0, space0);
RWByteAddressBuffer x : register(u1, space0);
RWByteAddressBuffer y : register(u2, space0);
ConstantBuffer<GemvParams> params : register(b3, space0);

[numthreads({wg}, 1, 1)]
void {entry}(uint3 gid : SV_DispatchThreadID) {{
    uint wave_size = WaveGetLaneCount();
    uint row_base = (gid.x / wave_size) * 16;
    if (row_base >= params.m) return;

    WaveMatrixA<float16_t> matA;
    WaveMatrixB<float16_t> matB;
    WaveMatrixC<float> matC;
    WaveMatrixFill(matC, 0.0f);

    uint n_tiles = (params.n + 15) / 16;
    for (uint t = 0; t < n_tiles; t++) {{
        uint col_base = t * 16;
        for (uint i = WaveGetLaneIndex(); i < 256; i += wave_size) {{
            uint gr = row_base + (i / 16); uint gc = col_base + (i % 16);
            float16_t val = (gr < params.m && gc < params.n)
                ? float16_t(asfloat(a.Load2(gr * params.n * 4 + gc * 4))) : 0.0h;
            WaveMatrixASetElement(matA, i, val);
        }}
        for (uint i = WaveGetLaneIndex(); i < 256; i += wave_size) {{
            uint lr = i / 16; uint gc = col_base + lr;
            float16_t val = (gc < params.n) ? float16_t(asfloat(x.Load2(gc * 4))) : 0.0h;
            for (uint c = 0; c < 16; c++) WaveMatrixBSetElement(matB, lr * 16 + c, val);
        }}
        WaveMatrixMultiply(matC, matA, matB);
    }}

    for (uint i = WaveGetLaneIndex(); i < 16; i += wave_size) {{
        uint gr = row_base + i;
        if (gr < params.m) {{
            float acc = 0.0f;
            for (uint c = 0; c < 16; c++) acc += WaveMatrixCGetElement(matC, i * 16 + c);
            y.Store2(gr * 4, asuint(acc));
        }}
    }}
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_p64_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        "struct P64Words64 {{ uint4 lanes[4]; }};\nStructuredBuffer<P64Words64> input : register(t0, space0);\nStructuredBuffer<float> weights : register(t1, space0);\nRWStructuredBuffer<float> output : register(u2, space0);\n[numthreads({wg}, 1, 1)]\nvoid {}(uint3 gid : SV_DispatchThreadID) {{\n    uint r = gid.x; uint count, stride; output.GetDimensions(count, stride);\n    if (r >= count) return;\n    P64Words64 rec = input[r]; float acc = 0.0;\n    for (uint w = 0; w < 16; w++) acc += weights[w] * (float)rec.lanes[w / 4][w % 4];\n    output[r] = acc;\n}}",
        kernel.entry_point
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_gemm_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        "struct GemmParams {{ uint m; uint n; uint k; uint _pad; }};\nStructuredBuffer<float> a : register(t0, space0);\nStructuredBuffer<float> b : register(t1, space0);\nRWStructuredBuffer<float> c : register(u2, space0);\nConstantBuffer<GemmParams> params : register(b3, space0);\n[numthreads({wg}, 1, 1)]\nvoid {}(uint3 gid : SV_DispatchThreadID) {{\n    uint o = gid.x; if (o >= params.m * params.n) return;\n    uint row = o / params.n; uint col = o % params.n; float acc = 0.0;\n    for (uint kk = 0; kk < params.k; kk++) acc += a[row * params.k + kk] * b[kk * params.n + col];\n    c[o] = acc;\n}}",
        kernel.entry_point
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_ternary_gemv_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        "struct TernaryGemvParams {{ uint m; uint k; uint k_words; uint _pad; }};\nStructuredBuffer<float> x : register(t0, space0);\nStructuredBuffer<uint> w_packed : register(t1, space0);\nStructuredBuffer<float> scale : register(t2, space0);\nRWStructuredBuffer<float> output : register(u3, space0);\nConstantBuffer<TernaryGemvParams> params : register(b4, space0);\n[numthreads({wg}, 1, 1)]\nvoid {}(uint3 gid : SV_DispatchThreadID) {{\n    uint o = gid.x; if (o >= params.m) return;\n    float acc = 0.0; uint row_base = o * params.k_words;\n    for (uint w = 0; w < params.k_words; w++) {{\n        uint word = w_packed[row_base + w];\n        for (uint l = 0; l < 16u; l++) {{\n            uint i = w * 16u + l; if (i >= params.k) break;\n            uint code = (word >> (l * 2u)) & 3u;\n            float tern = (code == 1u) ? 1.0 : ((code == 2u) ? -1.0 : 0.0);\n            acc += tern * x[i];\n        }}\n    }}\n    output[o] = scale[o] * acc;\n}}",
        kernel.entry_point
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_fft_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        "struct FftParams {{ uint n; uint log2n; uint _pad0; uint _pad1; }};\nStructuredBuffer<float> input : register(t0, space0);\nRWStructuredBuffer<float> output : register(u1, space0);\nConstantBuffer<FftParams> params : register(b2, space0);\ngroupshared float s_re[{wg}]; groupshared float s_im[{wg}];\n[numthreads({wg}, 1, 1)]\nvoid {}(uint tid : SV_GroupIndex) {{\n    uint rev = reversebits(tid) >> (32u - params.log2n);\n    s_re[rev] = input[2u * tid]; s_im[rev] = input[2u * tid + 1u];\n    GroupMemoryBarrierWithGroupSync();\n    output[2u * tid] = s_re[tid]; output[2u * tid + 1u] = s_im[tid];\n}}",
        kernel.entry_point
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_ops(source: &mut String, ops: &[Op], indent: &str) -> Result<(), ForgeError> {
    for op in ops {
        match op {
            Op::StructLoad {
                buffer,
                field,
                destination,
            } => {
                writeln!(source, "{indent}float {destination} = {buffer}.{field};")
                    .map_err(|e| ForgeError::Emission(e.to_string()))?;
            }
            Op::Load {
                buffer,
                index,
                destination,
            } => {
                writeln!(source, "{indent}float {destination} = {buffer}[{index}];")
                    .map_err(|e| ForgeError::Emission(e.to_string()))?;
            }
            Op::Store {
                buffer,
                index,
                value,
            } => {
                writeln!(source, "{indent}{buffer}[{index}] = {value};")
                    .map_err(|e| ForgeError::Emission(e.to_string()))?;
            }
            Op::Fma {
                a,
                b,
                c,
                destination,
            } => {
                writeln!(source, "{indent}float {destination} = {a} * {b} + {c};")
                    .map_err(|e| ForgeError::Emission(e.to_string()))?;
            }
            Op::Mul {
                left,
                right,
                destination,
            } => {
                writeln!(source, "{indent}float {destination} = {left} * {right};")
                    .map_err(|e| ForgeError::Emission(e.to_string()))?;
            }
            Op::Add {
                left,
                right,
                destination,
            } => {
                writeln!(source, "{indent}float {destination} = {left} + {right};")
                    .map_err(|e| ForgeError::Emission(e.to_string()))?;
            }
            _ => {}
        }
    }
    Ok(())
}
