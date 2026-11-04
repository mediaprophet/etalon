use crate::numerical::{ForgeError, KernelSpec, Schedule};
use std::fmt::Write;
pub(super) fn emit_gemm_wgsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        r#"
struct GemmParams {{
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}}

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
    .map_err(|error| ForgeError::Emission(error.to_string()))?;
    Ok(())
}

/// Dense row-major matrix-vector product, all f32: one invocation per output ROW
/// `i` computes `y[i] = sum_j A[i*N+j] * x[j]`. Self-contained (no shared memory);
/// dimensions come from the uniform params block (m, n, _pad0, _pad1). The inner
/// accumulation index is `j`.
pub(super) fn emit_gemv_wgsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;
    writeln!(
        source,
        r#"
struct GemvParams {{
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}}

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
    .map_err(|error| ForgeError::Emission(error.to_string()))?;
    Ok(())
}

/// Forward DFT via iterative radix-2 Decimation-In-Time over ONE workgroup of
/// `N = workgroup_size` threads (one complex element per thread; `N` a power of
/// two). Complex data is interleaved f32: element `j` is
/// `(input[2*j], input[2*j+1]) = (real, imag)`, so the input/output buffers hold
/// `2*N` f32. Stage 1 is a bit-reversal load into the `s_re`/`s_im` shared arrays;
/// then `log2(N)` butterfly stages run, each with a `workgroupBarrier()`. The same
/// forward sign convention `exp(-2*pi*i*k/m)` as the [`crate::numerical::oracle::dft_cpu`]
/// reference is used so GPU and CPU compute the identical transform. The shared
/// arrays come from the IR (`kernel.shared_memory`, sized to the workgroup size)
/// and `Op::Barrier` is the reusable IR primitive; the butterfly control flow is
/// specialised here because it is not expressible in the scalar op set.
pub(super) fn emit_fft_wgsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let wg = schedule.workgroup_size;

    writeln!(
        source,
        r#"
struct FftParams {{
    n: u32,
    log2n: u32,
    _pad0: u32,
    _pad1: u32,
}}
"#
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))?;

    for buffer in &kernel.buffers {
        let access = match buffer.access {
            crate::numerical::ir::BufferAccess::StorageRead => "storage, read",
            crate::numerical::ir::BufferAccess::StorageReadWrite => "storage, read_write",
            crate::numerical::ir::BufferAccess::Uniform => "uniform",
        };
        let type_decl = match buffer.access {
            crate::numerical::ir::BufferAccess::Uniform => "FftParams",
            _ => "array<f32>",
        };
        writeln!(
            source,
            "@group({}) @binding({}) var<{}> {}: {};",
            buffer.group, buffer.binding, access, buffer.name, type_decl
        )
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
    }

    writeln!(source, "").map_err(|error| ForgeError::Emission(error.to_string()))?;
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

    // `-2*pi` as a valid WGSL f32 literal (the `f` suffix keeps it f32). The
    // butterfly indexing and the forward twiddle `exp(-2*pi*i*k/m)` follow the
    // certified algorithm exactly.
    writeln!(
        source,
        r#"
@compute @workgroup_size({wg})
fn {entry}(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {{
    let t = local_id.x;                 // thread index 0..N-1
    let n = params.n;
    let logn = params.log2n;
    // 1. Bit-reversal load: element t goes to reverse(t).
    let rev = reverseBits(t) >> (32u - logn);
    s_re[rev] = input[2u * t];
    s_im[rev] = input[2u * t + 1u];
    workgroupBarrier();
    // 2. log2(N) butterfly stages. Stage s: span = 1<<s, m = 2*span.
    //    N/2 butterflies; threads 0..N/2-1 active.
    for (var s: u32 = 0u; s < logn; s = s + 1u) {{
        let span = 1u << s;
        let m = span << 1u;
        if (t < (n >> 1u)) {{
            let k = t & (span - 1u);                  // position within the butterfly group
            let j = ((t >> s) << (s + 1u)) + k;       // lower index = (t/span)*m + k
            let jp = j + span;                        // upper index
            // twiddle w = exp(-2*pi*i*k/m)
            let ang = -6.28318548f * f32(k) / f32(m);
            let wr = cos(ang);
            let wi = sin(ang);
            let ur = s_re[j];
            let ui = s_im[j];
            let vr = s_re[jp];
            let vi = s_im[jp];
            // v' = v * w
            let tr = vr * wr - vi * wi;
            let ti = vr * wi + vi * wr;
            s_re[j] = ur + tr;
            s_im[j] = ui + ti;
            s_re[jp] = ur - tr;
            s_im[jp] = ui - ti;
        }}
        workgroupBarrier();
    }}
    // 3. Store.
    output[2u * t] = s_re[t];
    output[2u * t + 1u] = s_im[t];
}}"#,
        wg = wg,
        entry = kernel.entry_point
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))?;

    Ok(())
}
