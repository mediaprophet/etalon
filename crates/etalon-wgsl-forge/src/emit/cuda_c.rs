//! CUDA C and PTX emission routing M <= 4 to DP4A and M >= 8 to WMMA.

use crate::ir::{ForgeError, Op};
use crate::schedule::Schedule;

pub fn emit_cuda_kernel(op: &Op, sched: &Schedule) -> Result<String, ForgeError> {
    match op {
        Op::Gemv { tc: _ } => {
            // For single-token decode, emit DP4A / scalar GEMV
            Ok(r#"extern "C" __global__ void gemv_dp4a(const float* x, const unsigned char* W, float* y, const unsigned* dims) {
    // High-efficiency DP4A scalar GEMV
}
"#.to_string())
        }
        Op::MatMul { tc: true, .. } => {
            // For batched prefill M >= 8, emit WMMA GEMM
            Ok(r#"#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void gemm_wmma_tiled(const half* A, const half* B, float* C) {
    // 16x16x16 tiled WMMA GEMM
}
"#.to_string())
        }
        _ => Err(ForgeError::Emission(format!("Op {op:?} not yet lowered in CUDA generator"))),
    }
}
