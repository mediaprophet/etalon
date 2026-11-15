//! Higher-Order Ambisonics (HOA) speaker/binaural matrix projection kernel;
//! device placement belongs to the caller.

use crate::numerical::ForgeError;
pub use etalon_signal::hoa::{
    hoa_decode_matrix_apply_into, HoaDecodeShape, HoaNormalization, HoaOrder,
};

pub const HOA_DECODE_ENTRY: &str = "hoa_decode";

pub const HOA_DECODE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> hoa_in: array<f32>;
@group(0) @binding(1) var<storage, read> decode_mat: array<f32>;
@group(0) @binding(2) var<storage, read_write> spk_out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;

@compute @workgroup_size(64)
fn hoa_decode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&spk_out)) {
        return;
    }
    let n_channels = u32(params[1]);
    let n_speakers = u32(params[2]);

    let frame = i / n_speakers;
    let spk = i - frame * n_speakers;

    let hoa_base = frame * n_channels;
    let mat_base = spk * n_channels;

    var acc = 0.0;
    for (var c: u32 = 0u; c < n_channels; c = c + 1u) {
        acc = acc + hoa_in[hoa_base + c] * decode_mat[mat_base + c];
    }
    spk_out[i] = acc;
}
"#;

pub fn hoa_decode_cpu(
    hoa_frames: &[f32],
    decode_matrix: &[f32],
    n_frames: usize,
    n_channels: usize,
    n_speakers: usize,
) -> Vec<f32> {
    let shape = HoaDecodeShape {
        n_frames,
        n_channels,
        n_speakers,
    };
    let mut out = vec![0.0f32; shape.output_len().expect("hoa output shape")];
    hoa_decode_matrix_apply_into(hoa_frames, decode_matrix, shape, &mut out)
        .expect("hoa decode shape");
    out
}

pub fn hoa_decode_gpu_capacity(
    hoa_frames: &[f32],
    decode_matrix: &[f32],
    shape: HoaDecodeShape,
) -> Result<usize, ForgeError> {
    let invalid = || ForgeError::GpuValidation("hoa_decode: invalid or oversized shape".into());
    if shape.n_frames == 0 || shape.n_channels == 0 || shape.n_speakers == 0 {
        return Err(invalid());
    }
    let in_len = shape.input_len().map_err(|_| invalid())?;
    let mat_len = shape.matrix_len().map_err(|_| invalid())?;
    let out_len = shape.output_len().map_err(|_| invalid())?;
    if hoa_frames.len() != in_len
        || decode_matrix.len() != mat_len
        || [shape.n_frames, shape.n_channels, shape.n_speakers]
            .iter()
            .any(|&n| n > (1 << 24))
        || [in_len, mat_len, out_len]
            .iter()
            .any(|&n| n > u32::MAX as usize)
    {
        return Err(invalid());
    }
    in_len
        .checked_add(mat_len)
        .and_then(|n| n.checked_add(out_len))
        .and_then(|n| n.checked_add(4))
        .and_then(|n| n.checked_mul(4))
        .map(|n| n.max(4 << 20))
        .ok_or_else(invalid)
}

pub fn hoa_decode_with_context(
    ctx: &mut crate::numerical::execute::WgpuComputeContext,
    hoa_frames: &[f32],
    decode_matrix: &[f32],
    n_frames: usize,
    n_channels: usize,
    n_speakers: usize,
) -> Result<Vec<f32>, ForgeError> {
    use crate::numerical::execute::{BindingUsage, QualiaCompute, WgpuPipeline};
    use crate::numerical::Schedule;

    let shape = HoaDecodeShape {
        n_frames,
        n_channels,
        n_speakers,
    };
    hoa_decode_gpu_capacity(hoa_frames, decode_matrix, shape)?;
    let out_len = n_frames * n_speakers;

    let view_in = ctx.allocate_and_write(
        bytemuck::cast_slice(hoa_frames),
        0,
        0,
        BindingUsage::StorageRead,
    )?;
    let view_mat = ctx.allocate_and_write(
        bytemuck::cast_slice(decode_matrix),
        1,
        0,
        BindingUsage::StorageRead,
    )?;
    let zeros = vec![0.0f32; out_len];
    let view_out = ctx.allocate_and_write(
        bytemuck::cast_slice(&zeros),
        2,
        0,
        BindingUsage::StorageReadWrite,
    )?;
    let params = [n_frames as f32, n_channels as f32, n_speakers as f32];
    let view_params = ctx.allocate_and_write(
        bytemuck::cast_slice(&params),
        3,
        0,
        BindingUsage::StorageRead,
    )?;

    let buffers = [view_in, view_mat, view_out, view_params];
    let pipeline = WgpuPipeline::compile(ctx, HOA_DECODE_WGSL, HOA_DECODE_ENTRY)?;
    let schedule = Schedule {
        workgroup_size: 64,
        ..Default::default()
    };
    pipeline.dispatch(&buffers, &schedule, out_len)?;
    let mut out = ctx.read_buffer_f32(&view_out)?;
    out.truncate(out_len);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::validate::validate_wgsl;

    #[test]
    fn hoa_decode_wgsl_validates() {
        let report = validate_wgsl(HOA_DECODE_WGSL).expect("hoa_decode WGSL must naga-validate");
        assert!(
            report.entry_points.iter().any(|e| e == HOA_DECODE_ENTRY),
            "validated module must expose {HOA_DECODE_ENTRY}; got {:?}",
            report.entry_points
        );
    }

    #[test]
    fn hoa_decode_cpu_matches_reference() {
        // 2 frames of Order-1 HOA ([W, Y, Z, X]):
        // Frame 0: +X = [1, 0, 0, 1]
        // Frame 1: +Y = [1, 1, 0, 0]
        let hoa_frames = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        // 2 virtual cardioid microphones pointing toward +X and +Y:
        // mic 0 (+X): [0.5, 0.0, 0.0, 0.5]
        // mic 1 (+Y): [0.5, 0.5, 0.0, 0.0]
        let decode_mat = vec![0.5f32, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0];
        let out = hoa_decode_cpu(&hoa_frames, &decode_mat, 2, 4, 2);
        assert_eq!(out, vec![1.0, 0.5, 0.5, 1.0]);
    }
}
