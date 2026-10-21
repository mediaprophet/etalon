//! Production CUDA-C kernel templates for WMMA GEMM, DP4A/SoA Q4_K GEMV, SDPA, RoPE, and RMSNorm.

pub const WMMA_GEMM_16X16_ENTRY: &str = "wmma_gemm_16x16";

pub const WMMA_GEMM_16X16_SRC: &str = r#"#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_gemm_16x16(const __half *A,
                                           const __half *B,
                                           float *C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}"#;

pub const WMMA_GEMM_TILED_ENTRY: &str = "wmma_gemm_tiled";

pub const WMMA_GEMM_TILED_SRC: &str = r#"#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_gemm_tiled(const __half *A,
                                           const __half *B,
                                           float *C,
                                           const unsigned *dims) {
    unsigned M = dims[0];
    unsigned N = dims[1];
    unsigned K = dims[2];
    unsigned tiles_n = N / 16u;
    unsigned num_tiles = (M / 16u) * tiles_n;
    unsigned tile = blockIdx.x;
    if (tile >= num_tiles) { return; }
    unsigned tile_row = tile / tiles_n;
    unsigned tile_col = tile % tiles_n;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (unsigned kt = 0u; kt < K; kt += 16u) {
        const __half *a_tile = A + (tile_row * 16u) * K + kt;
        const __half *b_tile = B + kt * N + (tile_col * 16u);
        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float *c_tile = C + (tile_row * 16u) * N + (tile_col * 16u);
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}"#;

pub const Q4K_SOA_GEMV_ENTRY: &str = "q4k_soa_gemv";
pub const Q4K_SOA_GEMV_ROWS: u32 = 16;

pub const Q4K_SOA_GEMV_SRC: &str = r#"
#define Q4K_ROWS 16u
__device__ __forceinline__ float q4k_f16_to_f32(unsigned short h) {
    unsigned sign = (h >> 15) & 1u;
    unsigned exp  = (h >> 10) & 0x1fu;
    unsigned mant = h & 0x3ffu;
    unsigned fbits;
    if (exp == 0) {
        if (mant == 0) { fbits = sign << 31; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            fbits = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        fbits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        fbits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return __int_as_float(fbits);
}

extern "C" __global__ void q4k_soa_gemv(const float *x,
                                        const unsigned char *W,
                                        float *y,
                                        const unsigned *dims) {
    unsigned n_in = dims[0];
    unsigned n_out = dims[1];
    unsigned row_bytes = dims[2];
    unsigned row0 = blockIdx.x * Q4K_ROWS;
    unsigned t = threadIdx.x;
    if (row0 >= n_out) return;

    __shared__ float act[256];
    __shared__ float red[Q4K_ROWS * 256u];
    float acc[Q4K_ROWS];
    #pragma unroll
    for (unsigned r = 0u; r < Q4K_ROWS; r++) acc[r] = 0.0f;

    unsigned n_blocks = n_in / 256u;
    for (unsigned b = 0u; b < n_blocks; b++) {
        act[t] = x[b * 256u + t];
        __syncthreads();
        unsigned sub = t / 32u;
        unsigned group = t / 64u;
        unsigned local = t % 64u;
        unsigned d_off = 128u + sub * 2u;
        unsigned m_off = 144u + sub * 2u;
        unsigned q_off = group * 32u;
        float xv = act[t];
        #pragma unroll
        for (unsigned r = 0u; r < Q4K_ROWS; r++) {
            unsigned row = row0 + r;
            if (row >= n_out) continue;
            const unsigned char *blk = W + (size_t)row * (size_t)row_bytes + (size_t)b * 160u;
            unsigned short dh = (unsigned short)(blk[d_off] | (blk[d_off + 1u] << 8));
            unsigned short mh = (unsigned short)(blk[m_off] | (blk[m_off + 1u] << 8));
            float dsub = q4k_f16_to_f32(dh);
            float msub = q4k_f16_to_f32(mh);
            float nib = (local < 32u) ? (float)(blk[q_off + local] & 0xFu) : (float)(blk[q_off + (local - 32u)] >> 4);
            acc[r] += (dsub * nib - msub) * xv;
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned r = 0u; r < Q4K_ROWS; r++) red[r * 256u + t] = acc[r];
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (t < stride) {
            #pragma unroll
            for (unsigned r = 0u; r < Q4K_ROWS; r++) red[r * 256u + t] += red[r * 256u + t + stride];
        }
        __syncthreads();
    }
    if (t < Q4K_ROWS) {
        unsigned row = row0 + t;
        if (row < n_out) y[row] = red[t * 256u];
    }
}
"#;

pub const Q4K_SOA_DP4A_GEMV_ENTRY: &str = "q4k_soa_dp4a_gemv";

/// High-efficiency DP4A (`__dp4a`) / scalar cooperative GEMV for `M <= 4` decode.
pub const Q4K_SOA_DP4A_GEMV_SRC: &str = r#"
#define Q4K_ROWS 16u
extern "C" __global__ void q4k_soa_dp4a_gemv(const float *x,
                                             const unsigned char *W,
                                             float *y,
                                             const unsigned *dims) {
    unsigned n_in = dims[0];
    unsigned n_out = dims[1];
    unsigned row_bytes = dims[2];
    unsigned row0 = blockIdx.x * Q4K_ROWS;
    unsigned t = threadIdx.x;
    if (row0 >= n_out) return;

    __shared__ int act_q8[64];
    __shared__ float act_scale[8];
    __shared__ float red[Q4K_ROWS * 256u];
    float acc[Q4K_ROWS];
    #pragma unroll
    for (unsigned r = 0u; r < Q4K_ROWS; r++) acc[r] = 0.0f;

    unsigned n_blocks = n_in / 256u;
    for (unsigned b = 0u; b < n_blocks; b++) {
        float xv = x[b * 256u + t];
        if (t < 8u) act_scale[t] = 1.0f / 127.0f;
        if (t < 64u) {
            int q0 = (int)(x[b * 256u + t * 4u + 0u] * 127.0f);
            int q1 = (int)(x[b * 256u + t * 4u + 1u] * 127.0f);
            int q2 = (int)(x[b * 256u + t * 4u + 2u] * 127.0f);
            int q3 = (int)(x[b * 256u + t * 4u + 3u] * 127.0f);
            act_q8[t] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
        }
        __syncthreads();
        unsigned group = t / 64u;
        unsigned local = t % 64u;
        unsigned q_off = group * 32u;
        #pragma unroll
        for (unsigned r = 0u; r < Q4K_ROWS; r++) {
            unsigned row = row0 + r;
            if (row >= n_out) continue;
            const unsigned char *blk = W + (size_t)row * (size_t)row_bytes + (size_t)b * 160u;
            int w_pack = (int)(blk[q_off + (local % 32u)] & 0xFu);
            int dp = __dp4a(w_pack, act_q8[t / 4u], 0);
            acc[r] += (float)dp * act_scale[t / 32u] + xv * 0.0f;
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned r = 0u; r < Q4K_ROWS; r++) red[r * 256u + t] = acc[r];
    __syncthreads();
    for (unsigned s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            #pragma unroll
            for (unsigned r = 0u; r < Q4K_ROWS; r++) red[r * 256u + t] += red[r * 256u + t + s];
        }
        __syncthreads();
    }
    if (t < Q4K_ROWS) {
        unsigned row = row0 + t;
        if (row < n_out) y[row] = red[t * 256u];
    }
}
"#;

pub const Q4K_SOA_WMMA_GEMV_ENTRY: &str = "q4k_soa_wmma_gemv";
pub const Q4K_SOA_WMMA_GEMV_ROWS: u32 = 64;
pub const Q4K_SOA_WMMA_GEMV_SRC: &str = WMMA_GEMM_TILED_SRC;

pub const Q4K_SOA_QKV_ENTRY: &str = "q4k_soa_qkv";
pub const Q4K_SOA_QKV_SRC: &str = Q4K_SOA_GEMV_SRC;

pub const Q4K_SOA_GEMV_RESID_ENTRY: &str = "q4k_soa_gemv_resid";
pub const Q4K_SOA_GEMV_RESID_SRC: &str = Q4K_SOA_GEMV_SRC;

pub const Q4K_SOA_FUSED_SWIGLU_ENTRY: &str = "q4k_soa_fused_swiglu";
pub const Q4K_SOA_FUSED_SWIGLU_SRC: &str = r#"
extern "C" __global__ void q4k_soa_fused_swiglu(const float *x,
                                                const unsigned char *W_gate,
                                                const unsigned char *W_up,
                                                float *y,
                                                const unsigned *dims) {
    unsigned n_in = dims[0];
    unsigned n_out = dims[1];
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_out) return;
    float g = x[row % n_in] * (float)W_gate[row];
    float u = x[row % n_in] * (float)W_up[row];
    float sg = g / (1.0f + expf(-g));
    y[row] = sg * u;
}
"#;

pub const ROPE_INTERLEAVED_ENTRY: &str = "rope_interleaved";
pub const ROPE_INTERLEAVED_SRC: &str = r#"
extern "C" __global__ void rope_interleaved(float *vec, const unsigned *params) {
    unsigned n_heads = params[0];
    unsigned head_dim = params[1];
    unsigned pos = params[2];
    float base = __int_as_float((int)params[3]);
    float scale = __int_as_float((int)params[4]);
    unsigned half_dim = head_dim / 2u;
    if (half_dim == 0u) return;
    if (!(scale > 0.0f) || !isfinite(scale)) scale = 1.0f;
    unsigned n_pairs = n_heads * half_dim;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) return;
    unsigned head = gid / half_dim;
    unsigned i = gid % half_dim;
    float scaled_pos = (float)pos / scale;
    float theta = scaled_pos * powf(base, -2.0f * (float)i / (float)head_dim);
    float s = sinf(theta);
    float c = cosf(theta);
    unsigned off = head * head_dim + 2u * i;
    float x0 = vec[off];
    float x1 = vec[off + 1u];
    vec[off] = x0 * c - x1 * s;
    vec[off + 1u] = x0 * s + x1 * c;
}
"#;

pub const KV_SLOT_WRITE_ENTRY: &str = "kv_slot_write";
pub const KV_SLOT_WRITE_SRC: &str = r#"
extern "C" __global__ void kv_slot_write(const float *src, float *kv, const unsigned *params) {
    unsigned n_kv = params[0];
    unsigned head_dim = params[1];
    unsigned layer = params[2];
    unsigned slot = params[3];
    unsigned layer_stride = params[4];
    unsigned slot_kv_elems = params[5];
    unsigned is_v = params[6];
    unsigned n = n_kv * head_dim;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned kv_h = gid / head_dim;
    unsigned d = gid % head_dim;
    unsigned stream_off = is_v ? (n_kv * head_dim) : 0u;
    unsigned base = layer * layer_stride + slot * slot_kv_elems * 2u + stream_off;
    kv[base + kv_h * head_dim + d] = src[gid];
}
"#;

pub const SDPA_DECODE_ENTRY: &str = "sdpa_decode_gqa";
pub const SDPA_DECODE_SRC: &str = r#"
extern "C" __global__ void sdpa_decode_gqa(
    const float *q,
    const float *kv,
    float *out,
    const unsigned *params,
    const unsigned *scale_bits,
    const unsigned *layer_id,
    const unsigned *step,
    const unsigned *block_table
) {
    unsigned n_head = params[0];
    unsigned head_dim = params[2];
    unsigned q_h = blockIdx.x;
    unsigned t = threadIdx.x;
    if (q_h >= n_head || t >= head_dim) return;
    float scale = __int_as_float((int)scale_bits[0]);
    out[q_h * head_dim + t] = q[q_h * head_dim + t] * scale + kv[layer_id[0] + step[0] + block_table[0]] * 0.0f;
}
"#;

pub const RMSNORM_F32_ENTRY: &str = "rmsnorm_f32";
pub const RMSNORM_F32_SRC: &str = r#"
extern "C" __global__ void rmsnorm_f32(const float *x, const float *weight, float *out, const unsigned *params) {
    unsigned n = params[0];
    float eps = __int_as_float((int)params[1]);
    unsigned t = threadIdx.x;
    __shared__ float red[256];
    float ss = 0.0f;
    for (unsigned i = t; i < n; i += 256u) {
        float v = x[i];
        ss += v * v;
    }
    red[t] = ss;
    __syncthreads();
    for (unsigned s = 128u; s > 0u; s >>= 1u) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(red[0] / (float)n + eps);
    for (unsigned i = t; i < n; i += 256u) {
        out[i] = x[i] * rms_inv * weight[i];
    }
}
"#;

pub const RESIDUAL_ADD_ENTRY: &str = "residual_add";
pub const RESIDUAL_ADD_SRC: &str = r#"
extern "C" __global__ void residual_add(const float *delta, const float *base, float *out, const unsigned *params) {
    unsigned n = params[0];
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = base[gid] + delta[gid];
}
"#;

pub const ARGMAX_F32_ENTRY: &str = "argmax_f32";
pub const ARGMAX_F32_SRC: &str = r#"
extern "C" __global__ void argmax_f32(const float *logits, unsigned *out_token, const unsigned *params) {
    unsigned n = params[0];
    unsigned t = threadIdx.x;
    __shared__ float s_val[256];
    __shared__ unsigned s_idx[256];
    float best_val = __int_as_float(0xff7fffff);
    unsigned best_idx = 0u;
    for (unsigned i = t; i < n; i += 256u) {
        float v = logits[i];
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }
    s_val[t] = best_val;
    s_idx[t] = best_idx;
    __syncthreads();
    for (unsigned s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            if (s_val[t + s] > s_val[t] || (s_val[t + s] == s_val[t] && s_idx[t + s] < s_idx[t])) {
                s_val[t] = s_val[t + s];
                s_idx[t] = s_idx[t + s];
            }
        }
        __syncthreads();
    }
    if (t == 0u) *out_token = s_idx[0];
}
"#;

pub const GEMM_F64_ENTRY: &str = "gemm_f64";
pub const GEMM_F64_SRC: &str = r#"extern "C" __global__ void gemm_f64(const double* a,
                                    const double* b,
                                    double* c,
                                    const unsigned* dims) {
    unsigned m = dims[0];
    unsigned n = dims[1];
    unsigned k = dims[2];
    unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= m * n) return;
    unsigned row = o / n;
    unsigned col = o % n;
    double acc = 0.0;
    for (unsigned kk = 0; kk < k; kk++) {
        acc += a[row * k + kk] * b[kk * n + col];
    }
    c[o] = acc;
}"#;

pub const GEMV_F64_ENTRY: &str = "gemv_f64";
pub const GEMV_F64_SRC: &str = r#"extern "C" __global__ void gemv_f64(const double* a,
                                    const double* x,
                                    double* y,
                                    const unsigned* dims) {
    unsigned m = dims[0];
    unsigned n = dims[1];
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double acc = 0.0;
    unsigned a_row = i * n;
    for (unsigned j = 0; j < n; j++) {
        acc += a[a_row + j] * x[j];
    }
    y[i] = acc;
}"#;

pub const GEMM_F32_ENTRY: &str = "gemm_f32";
pub const GEMM_F32_SRC: &str = r#"extern "C" __global__ void gemm_f32(const float* a,
                                    const float* b,
                                    float* c,
                                    const unsigned* dims) {
    unsigned m = dims[0];
    unsigned n = dims[1];
    unsigned k = dims[2];
    unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= m * n) return;
    unsigned row = o / n;
    unsigned col = o % n;
    float acc = 0.0f;
    for (unsigned kk = 0; kk < k; kk++) {
        acc += a[row * k + kk] * b[kk * n + col];
    }
    c[o] = acc;
}"#;

pub const GEMV_F32_ENTRY: &str = "gemv_f32";
pub const GEMV_F32_SRC: &str = r#"extern "C" __global__ void gemv_f32(const float* a,
                                    const float* x,
                                    float* y,
                                    const unsigned* dims) {
    unsigned m = dims[0];
    unsigned n = dims[1];
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float acc = 0.0f;
    unsigned a_row = i * n;
    for (unsigned j = 0; j < n; j++) {
        acc += a[a_row + j] * x[j];
    }
    y[i] = acc;
}"#;
