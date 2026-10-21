//! CUDA-C and PTX emission routing `M <= 4` to DP4A/scalar GEMV and `M >= 8` to WMMA.

use std::fmt::Write;

use crate::emit::coopmat::{CoopMatShape, should_use_coopmat};
use crate::emit::GeneratedShader;
use crate::ir::{DType, ForgeError, KernelSpec, Op};
use crate::schedule::Schedule;

pub use crate::emit::cuda_kernels::{
    ARGMAX_F32_ENTRY, ARGMAX_F32_SRC, GEMM_F32_ENTRY, GEMM_F32_SRC, GEMM_F64_ENTRY, GEMM_F64_SRC,
    GEMV_F32_ENTRY, GEMV_F32_SRC, GEMV_F64_ENTRY, GEMV_F64_SRC, KV_SLOT_WRITE_ENTRY,
    KV_SLOT_WRITE_SRC, Q4K_SOA_DP4A_GEMV_ENTRY, Q4K_SOA_DP4A_GEMV_SRC, Q4K_SOA_FUSED_SWIGLU_ENTRY,
    Q4K_SOA_FUSED_SWIGLU_SRC, Q4K_SOA_GEMV_ENTRY, Q4K_SOA_GEMV_RESID_ENTRY, Q4K_SOA_GEMV_RESID_SRC,
    Q4K_SOA_GEMV_ROWS, Q4K_SOA_GEMV_SRC, Q4K_SOA_QKV_ENTRY, Q4K_SOA_QKV_SRC,
    Q4K_SOA_WMMA_GEMV_ENTRY, Q4K_SOA_WMMA_GEMV_ROWS, Q4K_SOA_WMMA_GEMV_SRC, RESIDUAL_ADD_ENTRY,
    RESIDUAL_ADD_SRC, RMSNORM_F32_ENTRY, RMSNORM_F32_SRC, ROPE_INTERLEAVED_ENTRY,
    ROPE_INTERLEAVED_SRC, SDPA_DECODE_ENTRY, SDPA_DECODE_SRC, WMMA_GEMM_16X16_ENTRY,
    WMMA_GEMM_16X16_SRC, WMMA_GEMM_TILED_ENTRY, WMMA_GEMM_TILED_SRC,
};

/// Emits a complete CUDA-C module from a `KernelSpec` and `Schedule`.
pub fn emit_cuda_c(kernel: &KernelSpec, schedule: Schedule) -> Result<GeneratedShader, ForgeError> {
    kernel.validate()?;
    let semantic_hash = kernel.semantic_hash()?;
    let mut source = String::with_capacity(2048);
    writeln!(
        source,
        "// CUDA-C emitted for {}@{}",
        kernel.id, kernel.semantic_version
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))?;

    let wg = schedule.effective_workgroup_size();
    match kernel.id.as_str() {
        "affine-f32" => emit_affine(&mut source)?,
        "fused-ffn" => emit_ffn(&mut source)?,
        "topk" => emit_topk(&mut source, wg)?,
        "gemm" => {
            source.push_str(GEMM_F32_SRC);
            source.push('\n');
        }
        "gemv" => {
            source.push_str(GEMV_F32_SRC);
            source.push('\n');
        }
        "ternary-gemv" => emit_ternary_gemv(&mut source)?,
        "p64-project" => emit_p64_project(&mut source)?,
        "fft" => emit_fft(&mut source, wg)?,
        other => {
            return Err(ForgeError::Emission(format!(
                "CUDA-C emission not implemented for kernel {other}"
            )))
        }
    }

    let source_hash = blake3::hash(source.as_bytes()).to_hex().to_string();
    Ok(GeneratedShader {
        kernel_id: kernel.id.clone(),
        semantic_hash,
        source_hash,
        schedule,
        source,
    })
}

/// Emits a CUDA-C kernel string from a high-level `Op` and `Schedule`.
///
/// Enforces Runbook 01 Bug 1.5: single-token / small-batch decode (`M <= 4`) bypasses
/// WMMA and routes to DP4A / scalar cooperative GEMV (`Q4K_SOA_DP4A_GEMV_SRC`), while
/// batched prefill/verification (`M >= 8`) routes to `WMMA_GEMM_TILED_SRC`.
pub fn emit_cuda_kernel(op: &Op, sched: &Schedule) -> Result<String, ForgeError> {
    match op {
        Op::Gemv { tc } => {
            let shape = CoopMatShape::f16_16x16x16(1, sched.tile.n.max(16), sched.tile.k.max(16));
            if *tc && should_use_coopmat(&shape) {
                Ok(Q4K_SOA_WMMA_GEMV_SRC.to_string())
            } else {
                Ok(Q4K_SOA_DP4A_GEMV_SRC.to_string())
            }
        }
        Op::MatMul {
            tc,
            transpose_b: false,
        } => {
            let shape = CoopMatShape {
                m: sched.tile.m,
                n: sched.tile.n.max(16),
                k: sched.tile.k.max(16),
                tile_m: 16,
                tile_n: 16,
                tile_k: 16,
                a_type: DType::F16,
                b_type: DType::F16,
                c_type: DType::F32,
            };
            if *tc && should_use_coopmat(&shape) {
                Ok(WMMA_GEMM_TILED_SRC.to_string())
            } else if *tc && sched.tile.m <= 4 {
                Ok(Q4K_SOA_DP4A_GEMV_SRC.to_string())
            } else {
                Ok(GEMM_F32_SRC.to_string())
            }
        }
        Op::RmsNorm { .. } => Ok(RMSNORM_F32_SRC.to_string()),
        Op::RoPE { .. } => Ok(ROPE_INTERLEAVED_SRC.to_string()),
        Op::ElementwiseAdd => Ok(RESIDUAL_ADD_SRC.to_string()),
        Op::Silu => Ok(Q4K_SOA_FUSED_SWIGLU_SRC.to_string()),
        _ => Err(ForgeError::Emission(format!(
            "Op {op:?} not yet lowered in CUDA generator"
        ))),
    }
}

fn emit_affine(source: &mut String) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"struct AffineParams {{ unsigned length; float scale; float bias; unsigned _pad; }};
extern "C" __global__ void affine_f32(const float* input, float* output, AffineParams params) {{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < params.length) {{ output[gid] = input[gid] * params.scale + params.bias; }}
}}"#
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_ffn(source: &mut String) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"struct FfnParams {{ unsigned input_size; unsigned hidden_size; unsigned output_size; unsigned _pad; }};
extern "C" __global__ void fused_ffn(const float* input, const float* w1, const float* w2, float* output, FfnParams params) {{
    unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= params.output_size) return;
    float acc = 0.0f;
    for (unsigned h = 0; h < params.hidden_size; h++) {{
        float hv = 0.0f;
        unsigned w1_row = h * params.input_size;
        for (unsigned i = 0; i < params.input_size; i++) hv += w1[w1_row + i] * input[i];
        float g = 0.5f * hv * (1.0f + tanhf(0.7978845608f * (hv + 0.044715f * hv * hv * hv)));
        acc += w2[o * params.hidden_size + h] * g;
    }}
    output[o] = acc;
}}"#
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_topk(source: &mut String, wg: u32) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"struct TopKParams {{ unsigned length; unsigned k; unsigned block_size; unsigned _pad; }};
extern "C" __global__ void topk(const float* input, float* output, TopKParams params) {{
    const unsigned WG = {wg}u;
    __shared__ float s_val[{wg}];
    __shared__ unsigned s_idx[{wg}];
    __shared__ float r_val[{wg}];
    __shared__ unsigned r_idx[{wg}];
    unsigned tid = threadIdx.x;
    unsigned block = blockIdx.x;
    unsigned gidx = block * WG + tid;
    float sentinel = __int_as_float(0xff7fffff);
    float v = sentinel;
    if (gidx < params.length) v = input[gidx];
    s_val[tid] = v;
    s_idx[tid] = tid;
    __syncthreads();
    for (unsigned i = 0; i < params.k; i++) {{
        r_val[tid] = s_val[tid];
        r_idx[tid] = s_idx[tid];
        __syncthreads();
        for (unsigned stride = WG / 2u; stride > 0u; stride /= 2u) {{
            if (tid < stride) {{
                if (r_val[tid + stride] > r_val[tid]) {{
                    r_val[tid] = r_val[tid + stride];
                    r_idx[tid] = r_idx[tid + stride];
                }}
            }}
            __syncthreads();
        }}
        if (tid == 0u) {{
            output[block * params.k + i] = r_val[0];
            s_val[r_idx[0]] = sentinel;
        }}
        __syncthreads();
    }}
}}"#
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_ternary_gemv(source: &mut String) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"struct TernaryGemvParams {{ unsigned m; unsigned k; unsigned k_words; unsigned _pad; }};
extern "C" __global__ void ternary_gemv(const float* x,
                                        const unsigned* w_packed,
                                        const float* scale,
                                        float* output,
                                        TernaryGemvParams params) {{
    unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= params.m) return;
    float acc = 0.0f;
    unsigned row_base = o * params.k_words;
    for (unsigned word_idx = 0u; word_idx < params.k_words; word_idx++) {{
        unsigned word = w_packed[row_base + word_idx];
        unsigned lane_base = word_idx * 16u;
        for (unsigned lane = 0u; lane < 16u; lane++) {{
            unsigned i = lane_base + lane;
            if (i >= params.k) break;
            unsigned code = (word >> (lane * 2u)) & 3u;
            float tern = 0.0f;
            if (code == 1u) {{ tern = 1.0f; }} else if (code == 2u) {{ tern = -1.0f; }}
            acc += tern * x[i];
        }}
    }}
    output[o] = scale[o] * acc;
}}"#
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_p64_project(source: &mut String) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"struct P64Words64 {{ unsigned words[16]; }};
extern "C" __global__ void p64_project(const P64Words64* input,
                                       const float* weights,
                                       float* output,
                                       unsigned record_count) {{
    unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= record_count) return;
    P64Words64 rec = input[r];
    float acc = 0.0f;
    for (unsigned w = 0u; w < 16u; w++) {{
        acc += weights[w] * (float)rec.words[w];
    }}
    output[r] = acc;
}}"#
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

fn emit_fft(source: &mut String, wg: u32) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"struct FftParams {{ unsigned n; unsigned log2n; unsigned _pad0; unsigned _pad1; }};
extern "C" __global__ void fft(const float* input,
                               float* output,
                               FftParams params) {{
    __shared__ float s_re[{wg}];
    __shared__ float s_im[{wg}];
    unsigned t = threadIdx.x;
    unsigned n = params.n;
    unsigned logn = params.log2n;
    unsigned rev = __brev(t) >> (32u - logn);
    s_re[rev] = input[2u * t];
    s_im[rev] = input[2u * t + 1u];
    __syncthreads();
    for (unsigned s = 0u; s < logn; s++) {{
        unsigned span = 1u << s;
        unsigned m = span << 1u;
        if (t < (n >> 1u)) {{
            unsigned k = t & (span - 1u);
            unsigned j = ((t >> s) << (s + 1u)) + k;
            unsigned jp = j + span;
            float ang = -6.28318548f * (float)k / (float)m;
            float wr = cosf(ang);
            float wi = sinf(ang);
            float ur = s_re[j];
            float ui = s_im[j];
            float vr = s_re[jp];
            float vi = s_im[jp];
            float tr = vr * wr - vi * wi;
            float ti = vr * wi + vi * wr;
            s_re[j] = ur + tr;
            s_im[j] = ui + ti;
            s_re[jp] = ur - tr;
            s_im[jp] = ui - ti;
        }}
        __syncthreads();
    }}
    output[2u * t] = s_re[t];
    output[2u * t + 1u] = s_im[t];
}}"#
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}
