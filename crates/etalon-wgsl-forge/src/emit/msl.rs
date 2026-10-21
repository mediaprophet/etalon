//! Metal Shading Language (MSL) emitter.

use std::fmt::Write;

use super::msl_kernels::{
    emit_fft_msl, emit_fused_qkv_rope_msl, emit_gemm_msl, emit_gemv_msl, emit_gemv_simd_msl,
    emit_gemv_simdgroup_matrix_msl, emit_rmsnorm_msl, emit_sdpa_decode_msl, emit_ternary_gemv_msl,
};
use super::GeneratedShader;
use crate::ir::{BufferAccess, BufferElement, ForgeError, KernelSpec, Op, ScalarType};
use crate::ir_builtins::BuiltinKernel;
use crate::schedule::Schedule;

/// Emits an MSL shader module from a `KernelSpec` and `Schedule`.
pub fn emit_msl(kernel: &KernelSpec, schedule: Schedule) -> Result<GeneratedShader, ForgeError> {
    kernel.validate()?;
    let semantic_hash = kernel.semantic_hash()?;
    let mut source = String::with_capacity(2048);

    writeln!(
        source,
        "// MSL emitted for {}@{}\n// Semantic hash: {}\n// Schedule: workgroup={}, items={}, vector={}",
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

/// Emits a standalone MSL kernel string from a high-level `Op` and `Schedule`.
pub fn emit_msl_kernel(op: &Op, sched: &Schedule) -> Result<String, ForgeError> {
    match op {
        Op::MatMul { tc, .. } => {
            if sched.tile.m <= 4 {
                let spec = BuiltinKernel::Gemv.spec();
                let mut s = String::new();
                emit_gemv_simd_msl(&mut s, &spec, *sched)?;
                Ok(s)
            } else if *tc {
                let spec = BuiltinKernel::Gemv.spec();
                let mut s = String::new();
                emit_gemv_simdgroup_matrix_msl(&mut s, &spec, *sched)?;
                Ok(s)
            } else {
                Ok(emit_msl(&BuiltinKernel::Gemm.spec(), *sched)?.source)
            }
        }
        Op::Gemv { tc } => {
            let spec = BuiltinKernel::Gemv.spec();
            let mut s = String::new();
            if *tc && sched.tile.m > 4 {
                emit_gemv_simdgroup_matrix_msl(&mut s, &spec, *sched)?;
            } else {
                emit_gemv_simd_msl(&mut s, &spec, *sched)?;
            }
            Ok(s)
        }
        Op::RoPE { .. } => {
            let mut spec = BuiltinKernel::Gemv.spec();
            spec.id = "fused-qkv-rope".to_string();
            spec.entry_point = "fused_qkv_rope_main".to_string();
            let mut s = String::new();
            emit_fused_qkv_rope_msl(&mut s, &spec, *sched)?;
            Ok(s)
        }
        Op::RmsNorm { .. } => {
            let mut spec = BuiltinKernel::AffineF32.spec();
            spec.id = "rmsnorm".to_string();
            spec.entry_point = "rmsnorm_main".to_string();
            let mut s = String::new();
            emit_rmsnorm_msl(&mut s, &spec, *sched)?;
            Ok(s)
        }
        Op::Silu => Ok(emit_msl(&BuiltinKernel::FusedFfn.spec(), *sched)?.source),
        _ => Ok(emit_msl(&BuiltinKernel::AffineF32.spec(), *sched)?.source),
    }
}

fn emit_kernel_body(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    match kernel.id.as_str() {
        "topk" => return emit_topk_msl(source, kernel, schedule),
        "fused-ffn" => return emit_ffn_msl(source, kernel, schedule),
        "p64-project" => return emit_p64_msl(source, kernel, schedule),
        "gemm" => return emit_gemm_msl(source, kernel, schedule),
        "gemv" => {
            if schedule.workgroup_size >= 32 {
                return emit_gemv_simd_msl(source, kernel, schedule);
            }
            return emit_gemv_msl(source, kernel, schedule);
        }
        "gemv-simd-matrix" => return emit_gemv_simdgroup_matrix_msl(source, kernel, schedule),
        "fused-qkv-rope" => return emit_fused_qkv_rope_msl(source, kernel, schedule),
        "rmsnorm" => return emit_rmsnorm_msl(source, kernel, schedule),
        "sdpa-decode" => return emit_sdpa_decode_msl(source, kernel, schedule),
        "ternary-gemv" => return emit_ternary_gemv_msl(source, kernel, schedule),
        "fft" => return emit_fft_msl(source, kernel, schedule),
        "ray-probe" => {
            return Err(ForgeError::Emission(
                "ray-query is only emitted for the WGSL target (Metal RT uses a distinct API)"
                    .to_string(),
            ))
        }
        _ => {}
    }

    writeln!(source, "#include <metal_stdlib>\nusing namespace metal;\n")
        .map_err(|error| ForgeError::Emission(error.to_string()))?;

    if kernel.id == "affine-f32" {
        writeln!(
            source,
            "struct AffineParams {{ uint length; float scale; float bias; uint _pad; }};\n"
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
    }

    writeln!(source, "kernel void {}(", kernel.entry_point)
        .map_err(|error| ForgeError::Emission(error.to_string()))?;

    for (i, buffer) in kernel.buffers.iter().enumerate() {
        let type_decl = match (buffer.element, buffer.access) {
            (BufferElement::AffineParams, _) => "constant AffineParams&",
            (BufferElement::P64Words64, BufferAccess::StorageRead) => "device const P64Words64*",
            (BufferElement::P64Words64, BufferAccess::StorageReadWrite) => "device P64Words64*",
            (BufferElement::Scalar(ScalarType::F32), BufferAccess::StorageRead) => {
                "device const float*"
            }
            (BufferElement::Scalar(ScalarType::F32), BufferAccess::StorageReadWrite) => {
                "device float*"
            }
            _ => "device float*",
        };
        let sep = if i < kernel.buffers.len() - 1 { "," } else { "" };
        writeln!(source, "    {} {} [[buffer({})]]{}", type_decl, buffer.name, buffer.binding, sep)
            .map_err(|error| ForgeError::Emission(error.to_string()))?;
    }
    writeln!(
        source,
        "    , uint3 gid [[thread_position_in_grid]]\n) {{\n    const uint ITEMS_PER_INVOCATION = {}u;\n    const uint VECTOR_WIDTH = {}u;",
        schedule.items_per_invocation, schedule.vector_width
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
            writeln!(source, "        if (global_id + {}u < params.length) {{", schedule.vector_width - 1)
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
            for index in 0..schedule.vector_width {
                writeln!(
                    source,
                    "            output[global_id + {index}u] = input[global_id + {index}u] * params.scale + params.bias;"
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

fn emit_topk_msl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        "#include <metal_stdlib>\nusing namespace metal;\nstruct TopKParams {{ uint length; uint k; uint block_size; uint _pad; }};\nkernel void {}(\n    device const float* input [[buffer(0)]],\n    device float* output [[buffer(1)]],\n    constant TopKParams& params [[buffer(2)]],\n    uint tid [[thread_position_in_threadgroup]],\n    uint block [[threadgroup_position_in_grid]]\n) {{",
        kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))?;

    for shared in &kernel.shared_memory {
        writeln!(
            source,
            "    threadgroup {} {}[{}];",
            msl_scalar(shared.element),
            shared.name,
            shared.length.resolve(wg)
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
    }

    writeln!(
        source,
        r#"    uint gidx = block * {wg}u + tid;
    float sentinel = as_type<float>(0xff7fffffu);
    s_val[tid] = (gidx < params.length) ? input[gidx] : sentinel;
    s_idx[tid] = tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < params.k; i++) {{
        r_val[tid] = s_val[tid]; r_idx[tid] = s_idx[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = {wg}u / 2u; stride > 0u; stride /= 2u) {{
            if (tid < stride && r_val[tid + stride] > r_val[tid]) {{
                r_val[tid] = r_val[tid + stride]; r_idx[tid] = r_idx[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0u) {{ output[block * params.k + i] = r_val[0]; s_val[r_idx[0]] = sentinel; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}"#,
        wg = wg
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

fn emit_ffn_msl(
    source: &mut String,
    kernel: &KernelSpec,
    _schedule: Schedule,
) -> Result<(), ForgeError> {
    writeln!(
        source,
        "#include <metal_stdlib>\nusing namespace metal;\nstruct FfnParams {{ uint input_size; uint hidden_size; uint output_size; uint _pad; }};\nkernel void {}(\n    device const float* input [[buffer(0)]], device const float* w1 [[buffer(1)]],\n    device const float* w2 [[buffer(2)]], device float* output [[buffer(3)]],\n    constant FfnParams& params [[buffer(4)]], uint3 gid [[thread_position_in_grid]]\n) {{\n    uint o = gid.x; if (o >= params.output_size) return;\n    float acc = 0.0f;\n    for (uint h = 0; h < params.hidden_size; h++) {{\n        float hv = 0.0f; uint w1_row = h * params.input_size;\n        for (uint i = 0; i < params.input_size; i++) hv += w1[w1_row + i] * input[i];\n        float g = 0.5f * hv * (1.0f + tanh(0.7978845608f * (hv + 0.044715f * hv * hv * hv)));\n        acc += w2[o * params.hidden_size + h] * g;\n    }}\n    output[o] = acc;\n}}",
        kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

fn emit_p64_msl(
    source: &mut String,
    kernel: &KernelSpec,
    _schedule: Schedule,
) -> Result<(), ForgeError> {
    writeln!(
        source,
        "#include <metal_stdlib>\nusing namespace metal;\nstruct P64Words64 {{ uint4 lanes[4]; }};\nstruct P64Params {{ uint record_count; uint _pad0; uint _pad1; uint _pad2; }};\nkernel void {}(\n    device const P64Words64* input [[buffer(0)]], device const float* weights [[buffer(1)]],\n    device float* output [[buffer(2)]], constant P64Params& params [[buffer(3)]],\n    uint3 gid [[thread_position_in_grid]]\n) {{\n    uint r = gid.x; if (r >= params.record_count) return;\n    P64Words64 rec = input[r]; float acc = 0.0f;\n    for (uint w = 0; w < 16u; w++) acc += weights[w] * (float)rec.lanes[w / 4u][w % 4u];\n    output[r] = acc;\n}}",
        kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

pub(crate) fn msl_scalar(element: ScalarType) -> &'static str {
    match element {
        ScalarType::F32 => "float",
        ScalarType::U32 => "uint",
        ScalarType::I32 => "int",
        ScalarType::U64Words => "uint2",
    }
}

fn emit_ops(source: &mut String, ops: &[Op], indent: &str) -> Result<(), ForgeError> {
    for op in ops {
        match op {
            Op::StructLoad { buffer, field, destination } => {
                writeln!(source, "{indent}float {destination} = {buffer}.{field};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Load { buffer, index, destination } => {
                writeln!(source, "{indent}float {destination} = {buffer}[{index}];")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Store { buffer, index, value } => {
                writeln!(source, "{indent}{buffer}[{index}] = {value};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Fma { a, b, c, destination } => {
                writeln!(source, "{indent}float {destination} = {a} * {b} + {c};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Mul { left, right, destination } => {
                writeln!(source, "{indent}float {destination} = {left} * {right};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Add { left, right, destination } => {
                writeln!(source, "{indent}float {destination} = {left} + {right};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::DotProduct { left_buffer, left_base, right_buffer, right_base, len, destination } => {
                writeln!(
                    source,
                    "{indent}float {destination} = 0.0;\n{indent}for (uint i = 0; i < {len}; i++) {destination} += {left_buffer}[{left_base} + i] * {right_buffer}[{right_base} + i];"
                )
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Loop { induction_var, start, end, step, body } => {
                writeln!(source, "{indent}for (uint {induction_var} = {start}; {induction_var} < {end}; {induction_var} += {step}) {{")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
                emit_ops(source, body, &format!("{indent}    "))?;
                writeln!(source, "{indent}}}")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Relu { operand, destination } => {
                writeln!(source, "{indent}float {destination} = max(0.0f, {operand});")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Gelu { operand, destination } => {
                writeln!(
                    source,
                    "{indent}float {destination} = 0.5f * {operand} * (1.0f + tanh(0.7978845608f * ({operand} + 0.044715f * {operand} * {operand} * {operand})));"
                )
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Barrier => {
                writeln!(source, "{indent}threadgroup_barrier(mem_flags::mem_threadgroup);")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            other => {
                return Err(ForgeError::Emission(format!(
                    "Op {other:?} requires dedicated MSL kernel template"
                )));
            }
        }
    }
    Ok(())
}
