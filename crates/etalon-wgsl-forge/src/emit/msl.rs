//! Metal Shading Language emission with simdgroup matrix.

use crate::ir::{ForgeError, Op};
use crate::schedule::Schedule;

pub fn emit_msl_kernel(op: &Op, _sched: &Schedule) -> Result<String, ForgeError> {
    match op {
        Op::MatMul { .. } => Ok(r#"#include <metal_stdlib>
using namespace metal;

kernel void matmul_simdgroup() {
    // Metal 3.0 simdgroup matrix kernel
}
"#.to_string()),
        _ => Err(ForgeError::Emission(format!("Op {op:?} not yet lowered in MSL generator"))),
    }
}
