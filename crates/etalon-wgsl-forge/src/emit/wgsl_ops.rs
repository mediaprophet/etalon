//! WGSL statement emitters and specialized kernel templates (`gemm`, `gemv`, `fft`, radix sort).

use std::fmt::Write;

use crate::ir::{BufferAccess, ForgeError, Intrinsic, KernelSpec, Op};
use crate::schedule::Schedule;

pub(crate) fn emit_gemm_wgsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size.max(1);
    writeln!(
        source,
        r#"struct GemmParams {{ m: u32, n: u32, k: u32, _pad: u32 }}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: GemmParams;

@compute @workgroup_size({wg})
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let o = gid.x;
    if (o >= params.m * params.n) {{ return; }}
    let row = o / params.n;
    let col = o % params.n;
    var acc = 0.0;
    let a_row = row * params.k;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {{
        acc = acc + a[a_row + kk] * b[kk * params.n + col];
    }}
    c[o] = acc;
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

pub(crate) fn emit_gemv_wgsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size.max(1);
    writeln!(
        source,
        r#"struct GemvParams {{ m: u32, n: u32, _pad0: u32, _pad1: u32 }}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: GemvParams;

@compute @workgroup_size({wg})
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.m) {{ return; }}
    var acc = 0.0;
    let a_row = i * params.n;
    for (var j: u32 = 0u; j < params.n; j = j + 1u) {{
        acc = acc + a[a_row + j] * x[j];
    }}
    y[i] = acc;
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

pub(crate) fn emit_fft_wgsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size.max(1);
    writeln!(
        source,
        "struct FftParams {{ n: u32, log2n: u32, _pad0: u32, _pad1: u32 }}"
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))?;

    for buffer in &kernel.buffers {
        let access = match buffer.access {
            BufferAccess::StorageRead => "storage, read",
            BufferAccess::StorageReadWrite => "storage, read_write",
            BufferAccess::Uniform => "uniform",
        };
        let type_decl = match buffer.access {
            BufferAccess::Uniform => "FftParams",
            _ => "array<f32>",
        };
        writeln!(
            source,
            "@group({}) @binding({}) var<{}> {}: {};",
            buffer.group, buffer.binding, access, buffer.name, type_decl
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
    }

    for shared in &kernel.shared_memory {
        writeln!(
            source,
            "var<workgroup> {}: array<{}, {}>;",
            shared.name,
            shared.element.wgsl_name(),
            shared.length.resolve(wg)
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
    }

    writeln!(
        source,
        r#"@compute @workgroup_size({wg})
fn {entry}(@builtin(local_invocation_id) local_id: vec3<u32>) {{
    let t = local_id.x;
    let n = params.n;
    let logn = params.log2n;
    let rev = reverseBits(t) >> (32u - logn);
    s_re[rev] = input[2u * t];
    s_im[rev] = input[2u * t + 1u];
    workgroupBarrier();
    for (var s: u32 = 0u; s < logn; s = s + 1u) {{
        let span = 1u << s;
        let m = span << 1u;
        if (t < (n >> 1u)) {{
            let k = t & (span - 1u);
            let j = ((t >> s) << (s + 1u)) + k;
            let jp = j + span;
            let ang = -6.28318548f * f32(k) / f32(m);
            let wr = cos(ang);
            let wi = sin(ang);
            let ur = s_re[j];
            let ui = s_im[j];
            let vr = s_re[jp];
            let vi = s_im[jp];
            let tr = vr * wr - vi * wi;
            let ti = vr * wi + vi * wr;
            s_re[j] = ur + tr;
            s_im[j] = ui + ti;
            s_re[jp] = ur - tr;
            s_im[jp] = ui - ti;
        }}
        workgroupBarrier();
    }}
    output[2u * t] = s_re[t];
    output[2u * t + 1u] = s_im[t];
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))
}

pub(crate) fn emit_ops(source: &mut String, ops: &[Op], indent: &str) -> Result<(), ForgeError> {
    for op in ops {
        match op {
            Op::StructLoad {
                buffer,
                field,
                destination,
            } => {
                writeln!(source, "{indent}let {destination} = {buffer}.{field};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Load {
                buffer,
                index,
                destination,
            } => {
                writeln!(source, "{indent}let {destination} = {buffer}[{index}];")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Store {
                buffer,
                index,
                value,
            } => {
                writeln!(source, "{indent}{buffer}[{index}] = {value};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Fma {
                a,
                b,
                c,
                destination,
            } => {
                writeln!(source, "{indent}let {destination} = {a} * {b} + {c};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Mul {
                left,
                right,
                destination,
            } => {
                writeln!(source, "{indent}let {destination} = {left} * {right};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Add {
                left,
                right,
                destination,
            } => {
                writeln!(source, "{indent}let {destination} = {left} + {right};")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::DotProduct {
                left_buffer,
                left_base,
                right_buffer,
                right_base,
                len,
                destination,
            } => {
                writeln!(
                    source,
                    "{indent}var {destination} = 0.0;\n{indent}for (var i: u32 = 0u; i < {len}; i = i + 1u) {{\n{indent}    {destination} = {destination} + {left_buffer}[{left_base} + i] * {right_buffer}[{right_base} + i];\n{indent}}}"
                )
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Loop {
                induction_var,
                start,
                end,
                step,
                body,
            } => {
                writeln!(
                    source,
                    "{indent}for (var {induction_var}: u32 = {start}; {induction_var} < {end}; {induction_var} = {induction_var} + {step}) {{"
                )
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
                emit_ops(source, body, &format!("{indent}    "))?;
                writeln!(source, "{indent}}}")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Relu {
                operand,
                destination,
            } => {
                writeln!(source, "{indent}let {destination} = max(0.0, {operand});")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Gelu {
                operand,
                destination,
            } => {
                writeln!(
                    source,
                    "{indent}let {destination} = 0.5 * {operand} * (1.0 + tanh(0.7978845608 * ({operand} + 0.044715 * {operand} * {operand} * {operand})));"
                )
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::MatrixMultiply { .. } => {
                return Err(ForgeError::Emission(
                    "Op::MatrixMultiply has no scalar WGSL lowering; use coopmat".to_string(),
                ));
            }
            Op::Barrier => {
                writeln!(source, "{indent}workgroupBarrier();")
                    .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            Op::Intrinsic(Intrinsic::RayQuery {
                acceleration_structure,
                origin,
                direction,
                t_min,
                t_max,
                destination,
            }) => {
                writeln!(
                    source,
                    "{indent}var rq: ray_query;\n{indent}rayQueryInitialize(&rq, {acceleration_structure}, RayDesc(0u, 0xFFu, {t_min}, {t_max}, {origin}, {direction}));\n{indent}loop {{ if (!rayQueryProceed(&rq)) {{ break; }} }}\n{indent}let committed = rayQueryGetCommittedIntersection(&rq);\n{indent}var {destination} = -1.0;\n{indent}if (committed.kind != RAY_QUERY_INTERSECTION_NONE) {{ {destination} = committed.t; }}"
                )
                .map_err(|error| ForgeError::Emission(error.to_string()))?;
            }
            other => {
                return Err(ForgeError::Emission(format!(
                    "Op {other:?} requires dedicated shader template"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn emit_elementwise_wgsl(op_sym: &str, wg: u32) -> Result<String, ForgeError> {
    Ok(format!(
        "@group(0) @binding(0) var<storage, read> lhs: array<f32>;\n@group(0) @binding(1) var<storage, read> rhs: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@compute @workgroup_size({wg})\nfn ewise_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let i = gid.x;\n    if (i >= arrayLength(&output)) {{ return; }}\n    output[i] = lhs[i] {op_sym} rhs[i];\n}}"
    ))
}

pub(crate) fn emit_silu_wgsl(wg: u32) -> Result<String, ForgeError> {
    Ok(format!(
        "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@compute @workgroup_size({wg})\nfn silu_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let i = gid.x;\n    if (i >= arrayLength(&output)) {{ return; }}\n    let v = input[i];\n    output[i] = v / (1.0 + exp(-v));\n}}"
    ))
}

pub(crate) fn emit_rmsnorm_wgsl(eps: f32, wg: u32) -> Result<String, ForgeError> {
    Ok(format!(
        "@group(0) @binding(0) var<storage, read> x: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\n@group(0) @binding(2) var<storage, read_write> out: array<f32>;\nvar<workgroup> red: array<f32, {wg}>;\n@compute @workgroup_size({wg})\nfn rmsnorm_main(@builtin(local_invocation_id) lid: vec3<u32>) {{\n    let t = lid.x;\n    let n = arrayLength(&out);\n    var ss = 0.0;\n    for (var i: u32 = t; i < n; i = i + {wg}u) {{ let v = x[i]; ss = ss + v * v; }}\n    red[t] = ss;\n    workgroupBarrier();\n    for (var s: u32 = {wg}u / 2u; s > 0u; s = s / 2u) {{\n        if (t < s) {{ red[t] = red[t] + red[t + s]; }}\n        workgroupBarrier();\n    }}\n    let rms_inv = inverseSqrt(red[0] / f32(n) + {eps:e}f);\n    for (var i: u32 = t; i < n; i = i + {wg}u) {{ out[i] = x[i] * rms_inv * weight[i]; }}\n}}"
    ))
}

pub(crate) fn emit_rope_wgsl(theta: f32, wg: u32) -> Result<String, ForgeError> {
    Ok(format!(
        "struct RopeParams {{ n_heads: u32, head_dim: u32, pos: u32, _pad: u32 }}\n@group(0) @binding(0) var<storage, read_write> vec_io: array<f32>;\n@group(0) @binding(1) var<uniform> params: RopeParams;\n@compute @workgroup_size({wg})\nfn rope_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let half_dim = params.head_dim / 2u;\n    let idx = gid.x;\n    if (idx >= params.n_heads * half_dim) {{ return; }}\n    let head = idx / half_dim;\n    let i = idx % half_dim;\n    let ang = f32(params.pos) * pow({theta:e}f, -2.0 * f32(i) / f32(params.head_dim));\n    let s = sin(ang);\n    let c = cos(ang);\n    let off = head * params.head_dim + 2u * i;\n    let x0 = vec_io[off];\n    let x1 = vec_io[off + 1u];\n    vec_io[off] = x0 * c - x1 * s;\n    vec_io[off + 1u] = x0 * s + x1 * c;\n}}"
    ))
}

pub(crate) fn emit_prefix_scan_wgsl(wg: u32) -> Result<String, ForgeError> {
    Ok(format!(
        "@group(0) @binding(0) var<storage, read> input: array<u32>;\n@group(0) @binding(1) var<storage, read_write> output: array<u32>;\nvar<workgroup> scratch: array<u32, {wg}>;\n@compute @workgroup_size({wg})\nfn prefix_scan_main(@builtin(local_invocation_id) lid: vec3<u32>) {{\n    let t = lid.x;\n    let n = arrayLength(&output);\n    scratch[t] = select(0u, input[t], t < n);\n    workgroupBarrier();\n    for (var offset: u32 = 1u; offset < {wg}u; offset = offset * 2u) {{\n        var val = scratch[t];\n        if (t >= offset) {{ val = val + scratch[t - offset]; }}\n        workgroupBarrier();\n        scratch[t] = val;\n        workgroupBarrier();\n    }}\n    if (t < n) {{ output[t] = scratch[t]; }}\n}}"
    ))
}

pub(crate) fn emit_radix_histogram_wgsl(shift: u32, wg: u32) -> Result<String, ForgeError> {
    let shift_mod = shift & 31;
    Ok(format!(
        "struct Params {{ n: u32, shift: u32, num_wgs: u32, _pad1: u32 }}\n@group(0) @binding(0) var<storage, read> keys_in: array<u32>;\n@group(0) @binding(1) var<storage, read_write> wg_hist: array<u32>;\n@group(0) @binding(2) var<uniform> params: Params;\nvar<workgroup> local_hist: array<atomic<u32>, 256>;\n@compute @workgroup_size({wg})\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n    let t = lid.x;\n    if (t < 256u) {{ atomicStore(&local_hist[t], 0u); }}\n    workgroupBarrier();\n    let i = gid.x;\n    if (i < params.n) {{\n        let lo = keys_in[i * 2u];\n        let hi = keys_in[i * 2u + 1u];\n        let d = select((hi >> {shift_mod}u) & 255u, (lo >> {shift_mod}u) & 255u, {shift}u < 32u);\n        atomicAdd(&local_hist[d], 1u);\n    }}\n    workgroupBarrier();\n    if (t < 256u) {{ wg_hist[t * params.num_wgs + wid.x] = atomicLoad(&local_hist[t]); }}\n}}"
    ))
}

pub(crate) fn emit_radix_scatter_wgsl(shift: u32, wg: u32) -> Result<String, ForgeError> {
    let shift_mod = shift & 31;
    Ok(format!(
        "struct Params {{ n: u32, shift: u32, num_wgs: u32, _pad1: u32 }}\n@group(0) @binding(0) var<storage, read> keys_in: array<u32>;\n@group(0) @binding(1) var<storage, read> idx_in: array<u32>;\n@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;\n@group(0) @binding(3) var<storage, read_write> idx_out: array<u32>;\n@group(0) @binding(4) var<storage, read> wg_offsets: array<u32>;\n@group(0) @binding(5) var<uniform> params: Params;\nvar<workgroup> wg_digits: array<u32, {wg}>;\n@compute @workgroup_size({wg})\nfn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n    let i = gid.x;\n    let t = lid.x;\n    let valid = i < params.n;\n    var d = 0xFFFFFFFFu;\n    if (valid) {{\n        let lo = keys_in[i * 2u];\n        let hi = keys_in[i * 2u + 1u];\n        d = select((hi >> {shift_mod}u) & 255u, (lo >> {shift_mod}u) & 255u, {shift}u < 32u);\n    }}\n    wg_digits[t] = d;\n    workgroupBarrier();\n    if (!valid) {{ return; }}\n    var local_rank: u32 = 0u;\n    for (var u: u32 = 0u; u < t; u = u + 1u) {{ local_rank = local_rank + select(0u, 1u, wg_digits[u] == d); }}\n    let pos = wg_offsets[d * params.num_wgs + wid.x] + local_rank;\n    keys_out[pos * 2u] = keys_in[i * 2u];\n    keys_out[pos * 2u + 1u] = keys_in[i * 2u + 1u];\n    idx_out[pos] = idx_in[i];\n}}"
    ))
}
