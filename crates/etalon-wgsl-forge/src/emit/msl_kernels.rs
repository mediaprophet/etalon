//! Specialized Metal Shading Language (MSL) kernel templates.

use std::fmt::Write;

use crate::ir::{ForgeError, KernelSpec};
use crate::schedule::Schedule;

/// Dense row-major GEMM in Metal.
pub fn emit_gemm_msl(
    source: &mut String,
    kernel: &KernelSpec,
    _schedule: Schedule,
) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
struct GemmParams {{ uint m; uint n; uint k; uint _pad; }};
kernel void {entry}(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]], constant GemmParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint o = gid.x;
    if (o >= params.m * params.n) return;
    uint row = o / params.n; uint col = o % params.n;
    float acc = 0.0f; uint a_row = row * params.k;
    for (uint kk = 0; kk < params.k; kk++) acc += a[a_row + kk] * b[kk * params.n + col];
    c[o] = acc;
}}"#,
        entry = kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

/// Dense row-major GEMV in Metal (one thread per output row).
pub fn emit_gemv_msl(
    source: &mut String,
    kernel: &KernelSpec,
    _schedule: Schedule,
) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
struct GemvParams {{ uint m; uint n; uint _pad0; uint _pad1; }};
kernel void {entry}(
    device const float* a [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant GemvParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint i = gid.x;
    if (i >= params.m) return;
    float acc = 0.0f; uint a_row = i * params.n;
    for (uint j = 0; j < params.n; j++) acc += a[a_row + j] * x[j];
    y[i] = acc;
}}"#,
        entry = kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

/// SIMD-group cooperative GEMV in Metal (`simd_sum`).
pub fn emit_gemv_simd_msl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    let rows_per_block = (wg / 32).max(1);
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
struct GemvParams {{ uint m; uint n; uint _pad0; uint _pad1; }};
kernel void {entry}(
    device const float* a [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant GemvParams& params [[buffer(3)]],
    uint3 gtid [[thread_position_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]]
) {{
    uint simd_lane = gtid.x % 32u;
    uint row = gid.x * {rpb}u + (gtid.x / 32u);
    if (row >= params.m) return;
    float acc = 0.0f; uint a_row = row * params.n;
    for (uint j = simd_lane; j < params.n; j += 32u) acc += a[a_row + j] * x[j];
    acc = simd_sum(acc);
    if (simd_lane == 0u) y[row] = acc;
}}"#,
        entry = kernel.entry_point,
        rpb = rows_per_block,
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

/// BitNet-style ternary GEMV in Metal.
pub fn emit_ternary_gemv_msl(
    source: &mut String,
    kernel: &KernelSpec,
    _schedule: Schedule,
) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
struct TernaryGemvParams {{ uint m; uint k; uint k_words; uint _pad; }};
kernel void {entry}(
    device const float* x [[buffer(0)]], device const uint* w_packed [[buffer(1)]],
    device const float* scale [[buffer(2)]], device float* output [[buffer(3)]],
    constant TernaryGemvParams& params [[buffer(4)]], uint3 gid [[thread_position_in_grid]]
) {{
    uint o = gid.x; if (o >= params.m) return;
    float acc = 0.0f; uint row_base = o * params.k_words;
    for (uint word_idx = 0; word_idx < params.k_words; word_idx++) {{
        uint word = w_packed[row_base + word_idx];
        for (uint lane = 0; lane < 16u; lane++) {{
            uint i = word_idx * 16u + lane;
            if (i >= params.k) break;
            uint code = (word >> (lane * 2u)) & 3u;
            float tern = (code == 1u) ? 1.0f : ((code == 2u) ? -1.0f : 0.0f);
            acc += tern * x[i];
        }}
    }}
    output[o] = scale[o] * acc;
}}"#,
        entry = kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

/// Forward radix-2 DIT FFT in Metal.
pub fn emit_fft_msl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
struct FftParams {{ uint n; uint log2n; uint _pad0; uint _pad1; }};
kernel void {entry}(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant FftParams& params [[buffer(2)]], uint tid [[thread_position_in_threadgroup]]
) {{
    threadgroup float s_re[{wg}]; threadgroup float s_im[{wg}];
    uint t = tid; uint n = params.n; uint logn = params.log2n;
    uint rev = reverse_bits(t) >> (32u - logn);
    s_re[rev] = input[2u * t]; s_im[rev] = input[2u * t + 1u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0u; s < logn; s++) {{
        uint span = 1u << s; uint m = span << 1u;
        if (t < (n >> 1u)) {{
            uint k = t & (span - 1u); uint j = ((t >> s) << (s + 1u)) + k; uint jp = j + span;
            float ang = -6.28318548f * (float)k / (float)m;
            float wr = cos(ang); float wi = sin(ang);
            float ur = s_re[j]; float ui = s_im[j]; float vr = s_re[jp]; float vi = s_im[jp];
            float tr = vr * wr - vi * wi; float ti = vr * wi + vi * wr;
            s_re[j] = ur + tr; s_im[j] = ui + ti; s_re[jp] = ur - tr; s_im[jp] = ui - ti;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    output[2u * t] = s_re[t]; output[2u * t + 1u] = s_im[t];
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

/// MSL fused QKV + RoPE kernel.
pub fn emit_fused_qkv_rope_msl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size.max(32);
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
#define ROWS_PER_BLOCK 16u
kernel void {entry}(
    device const float* x [[buffer(0)]], device const float* Wq [[buffer(1)]],
    device const float* Wk [[buffer(2)]], device const float* Wv [[buffer(3)]],
    device float* yq [[buffer(4)]], device float* yk [[buffer(5)]],
    device float* yv [[buffer(6)]], device const uint* dims [[buffer(7)]],
    device const uint* rope_params [[buffer(8)]],
    uint3 gtid [[thread_position_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]
) {{
    uint n_in = dims[0]; uint n_q = dims[1]; uint n_kv = dims[2];
    uint head_dim = dims[4]; uint pos = dims[5];
    uint row0 = gid.x * ROWS_PER_BLOCK; uint t = gtid.x;
    if (row0 >= n_q) return;
    float base = as_type<float>(rope_params[0]); float scale = as_type<float>(rope_params[1]);
    float inv_scale = 1.0f / scale; float inv_head_dim = 1.0f / (float)head_dim;
    for (uint r = 0u; r < ROWS_PER_BLOCK; r++) {{
        uint row = row0 + r;
        if (row < n_q) {{
            float acc = 0.0f;
            for (uint j = t; j < n_in; j += {wg}) acc += Wq[row * n_in + j] * x[j];
            acc = simd_sum(acc);
            if (t == 0u) {{
                float theta = (float)pos * inv_scale * pow(base, -2.0f * (float)((row % head_dim) / 2u) * inv_head_dim);
                yq[row] = acc * cos(theta);
            }}
        }}
        if (row < n_kv) {{
            float acc_k = 0.0f; float acc_v = 0.0f;
            for (uint j = t; j < n_in; j += {wg}) {{ acc_k += Wk[row * n_in + j] * x[j]; acc_v += Wv[row * n_in + j] * x[j]; }}
            acc_k = simd_sum(acc_k); acc_v = simd_sum(acc_v);
            if (t == 0u) {{
                float theta = (float)pos * inv_scale * pow(base, -2.0f * (float)((row % head_dim) / 2u) * inv_head_dim);
                yk[row] = acc_k * cos(theta); yv[row] = acc_v;
            }}
        }}
    }}
}}"#,
        wg = wg,
        entry = kernel.entry_point,
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

/// MSL RMSNorm kernel.
pub fn emit_rmsnorm_msl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size.max(32);
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {entry}(
    device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]],
    device float* y [[buffer(2)]], device const uint* params [[buffer(3)]],
    uint3 gtid [[thread_position_in_threadgroup]]
) {{
    uint n = params[0]; float eps = as_type<float>(params[1]); uint t = gtid.x;
    threadgroup float s_sum[{wg}];
    s_sum[t] = (t < n) ? (x[t] * x[t]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = {wg} / 2u; s > 0u; s >>= 1u) {{
        if (t < s) s_sum[t] += s_sum[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float inv_rms = rsqrt(s_sum[0] / (float)n + eps);
    if (t < n) y[t] = x[t] * inv_rms * weight[t];
}}"#,
        wg = wg,
        entry = kernel.entry_point,
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

/// MSL SDPA decode kernel.
pub fn emit_sdpa_decode_msl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size.max(32);
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {entry}(
    device const float* q [[buffer(0)]], device const float* kv [[buffer(1)]],
    device float* out [[buffer(2)]], device const uint* params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]], uint3 gtid [[thread_position_in_threadgroup]]
) {{
    uint n_head = params[0]; uint head_dim = params[2]; uint layer = params[3];
    uint pos = params[4]; uint max_context = params[5]; uint layer_stride = params[6];
    uint slot_kv_elems = params[7]; uint q_per_kv = params[8];
    float scale = 1.0f / sqrt((float)head_dim);
    uint head = gid.x; if (head >= n_head) return;
    uint t = gtid.x; uint kv_head = head / q_per_kv;
    threadgroup float s_max[1];
    float max_score = -1e30f;
    for (uint kv_idx = 0u; kv_idx <= pos && kv_idx < max_context; kv_idx++) {{
        float dot = 0.0f;
        for (uint d = t; d < head_dim; d += {wg}) {{
            dot += q[head * head_dim + d] * kv[kv_idx * layer_stride + layer * 2u * slot_kv_elems + kv_head * head_dim + d];
        }}
        dot = simd_sum(dot);
        max_score = max(max_score, scale * dot);
    }}
    if (t == 0u) s_max[0] = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = s_max[0];
    float sum_weights = 0.0f;
    for (uint kv_idx = 0u; kv_idx <= pos && kv_idx < max_context; kv_idx++) {{
        float dot = 0.0f;
        for (uint d = t; d < head_dim; d += {wg}) {{
            dot += q[head * head_dim + d] * kv[kv_idx * layer_stride + layer * 2u * slot_kv_elems + kv_head * head_dim + d];
        }}
        dot = simd_sum(dot);
        sum_weights += exp2(scale * dot - max_score);
    }}
    float inv_sum = 1.0f / sum_weights;
    for (uint d = t; d < head_dim; d += {wg}) {{
        float acc = 0.0f;
        for (uint kv_idx = 0u; kv_idx <= pos && kv_idx < max_context; kv_idx++) {{
            float dot = 0.0f;
            for (uint dd = 0u; dd < head_dim; dd++) {{
                dot += q[head * head_dim + dd] * kv[kv_idx * layer_stride + layer * 2u * slot_kv_elems + kv_head * head_dim + dd];
            }}
            float weight = exp2(scale * dot - max_score) * inv_sum;
            acc += weight * kv[kv_idx * layer_stride + layer * 2u * slot_kv_elems + slot_kv_elems + kv_head * head_dim + d];
        }}
        out[head * head_dim + d] = acc;
    }}
}}"#,
        wg = wg,
        entry = kernel.entry_point,
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}

/// MSL SIMD-group matrix GEMV using `metal::simdgroup_matrix` (8x8 tiles).
pub fn emit_gemv_simdgroup_matrix_msl(
    source: &mut String,
    kernel: &KernelSpec,
    _schedule: Schedule,
) -> Result<(), ForgeError> {
    writeln!(
        source,
        r#"#include <metal_stdlib>
using namespace metal;
struct GemvParams {{ uint m; uint n; uint _pad0; uint _pad1; }};
kernel void {entry}(
    device const float* a [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant GemvParams& params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]], uint3 gtid [[thread_position_in_threadgroup]]
) {{
    uint row = gid.x * 8u; if (row >= params.m) return;
    simdgroup_matrix<float, 8, 8> sm_a;
    simdgroup_matrix<float, 8, 8> sm_x;
    simdgroup_matrix<float, 8, 8> sm_acc = make_simdgroup_matrix<float, 8, 8>(0.0f);
    uint n_tiles = params.n / 8u;
    for (uint t = 0u; t < n_tiles; t++) {{
        for (uint i = 0u; i < 8u; i++) {{
            for (uint j = 0u; j < 8u; j++) {{
                uint r = row + i; uint c = t * 8u + j;
                sm_a[i * 8u + j] = (r < params.m && c < params.n) ? a[r * params.n + c] : 0.0f;
            }}
        }}
        for (uint j = 0u; j < 8u; j++) {{
            uint c = t * 8u + j;
            sm_x[j] = (c < params.n) ? x[c] : 0.0f;
            for (uint i = 1u; i < 8u; i++) sm_x[i * 8u + j] = 0.0f;
        }}
        simdgroup_multiply_accumulate(sm_acc, sm_a, sm_x, sm_acc);
    }}
    uint lane = gtid.x;
    if (lane < 8u && (row + lane) < params.m) y[row + lane] = sm_acc[lane * 8u];
}}"#,
        entry = kernel.entry_point,
    )
    .map_err(|e| ForgeError::Emission(e.to_string()))
}
