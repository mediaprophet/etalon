//! Portable WGSL emission including subgroups and cooperative matrix.

use crate::ir::{ForgeError, Op};
use crate::schedule::Schedule;

pub fn emit_wgsl_kernel(op: &Op, sched: &Schedule) -> Result<String, ForgeError> {
    match op {
        Op::RadixHistogram { shift } => Ok(format!(
            r#"struct Params {{ n: u32, shift: u32, num_wgs: u32, _pad1: u32 }}
@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> wg_hist: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size({threads})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    atomicStore(&local_hist[t], 0u);
    workgroupBarrier();
    let i = gid.x;
    if (i < params.n) {{
        let lo = keys_in[i * 2u];
        let hi = keys_in[i * 2u + 1u];
        let d = select((hi >> ({shift}u - 32u)) & 255u, (lo >> {shift}u) & 255u, {shift}u < 32u);
        atomicAdd(&local_hist[d], 1u);
    }}
    workgroupBarrier();
    wg_hist[t * params.num_wgs + wid.x] = atomicLoad(&local_hist[t]);
}}"#,
            threads = sched.workgroup.x,
            shift = shift,
        )),
        _ => Err(ForgeError::Emission(format!("Op {op:?} not yet lowered in wgsl generator"))),
    }
}
