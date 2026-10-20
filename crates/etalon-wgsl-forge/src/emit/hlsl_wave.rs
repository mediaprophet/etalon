//! HLSL Shader Model 6.0-6.9 WaveOps emission without external subprocesses.

use crate::ir::{ForgeError, Op};
use crate::schedule::Schedule;

pub fn emit_hlsl_wave_kernel(op: &Op, sched: &Schedule) -> Result<String, ForgeError> {
    match op {
        Op::PrefixScan => Ok(format!(
            r#"[numthreads({threads}, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    // SM 6.0 WavePrefixSum intrinsic
}}
"#,
            threads = sched.workgroup.x
        )),
        _ => Err(ForgeError::Emission(format!("Op {op:?} not yet lowered in HLSL generator"))),
    }
}
