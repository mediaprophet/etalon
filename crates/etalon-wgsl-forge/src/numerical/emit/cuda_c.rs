pub use crate::emit::cuda_c::*;
pub const Q4K_SOA_WMMA_GEMV_RESID_ENTRY: &str = "q4k_soa_wmma_gemv_resid";
pub const Q4K_SOA_WMMA_GEMV_RESID_SRC: &str = r#"#include <mma.h>
using namespace nvcuda;
#define WMMA_M 16u
#define WMMA_N 16u
#define WMMA_K 16u
#define WARPS_PER_BLOCK 4u
#define ROWS_PER_BLOCK (WMMA_M * WARPS_PER_BLOCK)
#define K_BLOCK 256u
#define K_TILES_PER_BLOCK (K_BLOCK / WMMA_K)

extern "C" __global__ void q4k_soa_wmma_gemv_resid(const float *x,
                                                    const unsigned char *W,
                                                    float *y,
                                                    const unsigned *dims,
                                                    const float *residual) {
    unsigned n_in = dims[0];
    unsigned n_out = dims[1];
    unsigned row_bytes = dims[2];
    unsigned row0 = blockIdx.x * ROWS_PER_BLOCK;
    unsigned warp_id = threadIdx.x / 32u;
    unsigned lane = threadIdx.x % 32u;
    unsigned warp_row0 = row0 + warp_id * WMMA_M;
    if (row0 >= n_out) return;

    __shared__ __half x_tile[K_TILES_PER_BLOCK][WMMA_K * WMMA_N];
    __shared__ __half w_tile[WARPS_PER_BLOCK][WMMA_M * WMMA_K];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    unsigned n_k_blocks = n_in / K_BLOCK;
    for (unsigned kb = 0u; kb < n_k_blocks; kb++) {
        for (unsigned i = threadIdx.x; i < K_BLOCK; i += 128u) {
            unsigned tile = i / WMMA_K;
            unsigned elem = i % WMMA_K;
            x_tile[tile][elem] = __float2half(x[kb * K_BLOCK + i]);
        }
        for (unsigned i = threadIdx.x; i < K_TILES_PER_BLOCK * WMMA_K * (WMMA_N - 1u); i += 128u) {
            unsigned tile = i / (WMMA_K * (WMMA_N - 1u));
            unsigned rest = i % (WMMA_K * (WMMA_N - 1u));
            unsigned elem = rest / (WMMA_N - 1u);
            unsigned col = rest % (WMMA_N - 1u) + 1u;
            x_tile[tile][elem * WMMA_N + col] = __float2half(0.0f);
        }
        __syncthreads();

        for (unsigned kt = 0u; kt < K_TILES_PER_BLOCK; kt++) {
            unsigned k_base = kt * WMMA_K;
            unsigned group = k_base / 32u;
            unsigned sub_in_group = (k_base % 32u) / 16u;

            for (unsigned i = lane; i < WMMA_M * WMMA_K; i += 32u) {
                unsigned r = i / WMMA_K;
                unsigned k = i % WMMA_K;
                unsigned row = warp_row0 + r;
                if (row >= n_out) {
                    w_tile[warp_id][i] = __float2half(0.0f);
                    continue;
                }
                const unsigned char *blk =
                    W + (size_t)row * (size_t)row_bytes + (size_t)kb * 160u;
                unsigned d_off = 128u + group * 2u;
                unsigned m_off = 144u + group * 2u;
                unsigned short dh = (unsigned short)(blk[d_off] | (blk[d_off + 1u] << 8));
                unsigned short mh = (unsigned short)(blk[m_off] | (blk[m_off + 1u] << 8));
                float dsub = __half2float(__ushort_as_half(dh));
                float msub = __half2float(__ushort_as_half(mh));
                unsigned nib_idx = sub_in_group * 16u + k;
                unsigned byte_idx = group * 32u + nib_idx;
                unsigned char byte_val = blk[byte_idx / 2u];
                float nib = (nib_idx % 2u == 0u)
                    ? (float)(byte_val & 0xFu)
                    : (float)(byte_val >> 4);
                float deq = dsub * nib - msub;
                w_tile[warp_id][i] = __float2half(deq);
            }
            __syncwarp();

            wmma::load_matrix_sync(a_frag, w_tile[warp_id], WMMA_K);
            wmma::load_matrix_sync(b_frag, x_tile[kt], WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    __shared__ float c_store[WARPS_PER_BLOCK][WMMA_M * WMMA_N];
    wmma::store_matrix_sync(c_store[warp_id], c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    if (lane < WMMA_M) {
        unsigned row = warp_row0 + lane;
        if (row < n_out) {
            y[row] = residual[row] + c_store[warp_id][lane * WMMA_N];
        }
    }
}
"#;
pub const Q6K_SOA_GEMV_ENTRY: &str = "q6k_soa_gemv";
pub const Q6K_SOA_GEMV_ROWS: u32 = 16;
pub const Q6K_SOA_GEMV_SRC: &str = r#"
#define Q6K_ROWS 16u
__device__ __forceinline__ float q6k_f16_to_f32(unsigned short h) {
    unsigned sign = (h >> 15) & 1u;
    unsigned exp  = (h >> 10) & 0x1fu;
    unsigned mant = h & 0x3ffu;
    unsigned fbits;
    if (exp == 0) {
        if (mant == 0) fbits = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            fbits = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) fbits = (sign << 31) | 0x7f800000u | (mant << 13);
    else fbits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    return __int_as_float(fbits);
}

extern "C" __global__ void q6k_soa_gemv(const float *x,
                                        const unsigned char *W,
                                        float *y,
                                        const unsigned *dims) {
    unsigned n_in = dims[0];
    unsigned n_out = dims[1];
    unsigned row_bytes = dims[2];
    unsigned row0 = blockIdx.x * Q6K_ROWS;
    unsigned t = threadIdx.x;
    if (row0 >= n_out) return;

    float acc[Q6K_ROWS];
    #pragma unroll
    for (unsigned r = 0u; r < Q6K_ROWS; r++) acc[r] = 0.0f;

    unsigned n_blocks = n_in / 256u;
    for (unsigned b = 0u; b < n_blocks; b++) {
        float xv = x[b * 256u + t];
        #pragma unroll
        for (unsigned r = 0u; r < Q6K_ROWS; r++) {
            unsigned row = row0 + r;
            if (row >= n_out) continue;
            const unsigned char *blk = W + (size_t)row * (size_t)row_bytes + (size_t)b * 210u;
            unsigned y_in_block = t;
            unsigned chunk = y_in_block / 128u;
            unsigned ql_idx = y_in_block % 128u;
            unsigned short dh = (unsigned short)(blk[208] | (blk[209] << 8));
            float d = q6k_f16_to_f32(dh);
            unsigned ql_byte = blk[ql_idx / 2u];
            unsigned qh_idx = (y_in_block / 4u) & 63u;
            unsigned qh_byte = blk[128u + qh_idx];
            unsigned q = (ql_byte & 0xFu) | ((qh_byte >> 4u) & 0x3u) << 4u;
            unsigned scale_idx = chunk * 8u + (ql_idx / 16u);
            int scale = (signed char)blk[192u + scale_idx];
            float scale_f = (float)scale;
            float deq = d * scale_f * ((float)q - 32.0f);
            acc[r] += deq * xv;
        }
    }

    __shared__ float red[Q6K_ROWS * 256u];
    #pragma unroll
    for (unsigned r = 0u; r < Q6K_ROWS; r++) red[r * 256u + t] = acc[r];
    __syncthreads();
    for (unsigned s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            #pragma unroll
            for (unsigned r = 0u; r < Q6K_ROWS; r++)
                red[r * 256u + t] += red[r * 256u + t + s];
        }
        __syncthreads();
    }
    if (t < Q6K_ROWS) {
        unsigned row = row0 + t;
        if (row < n_out) y[row] = red[t * 256u];
    }
}
"#;

pub fn emit_cuda_c(
    kernel: &crate::numerical::KernelSpec,
    schedule: crate::numerical::Schedule,
) -> Result<super::GeneratedShader, crate::numerical::ForgeError> {
    super::emit_canonical(kernel, schedule, super::TargetBackend::CudaC)
}
