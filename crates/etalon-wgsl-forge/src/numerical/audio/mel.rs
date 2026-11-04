//! Mel projection execution; device placement belongs to the caller.
use crate::numerical::ForgeError;
pub use etalon_signal::mel::{mel_apply_into, MelShape, PreparedMel};
pub const MEL_APPLY_WGSL: &str = include_str!("mel.wgsl");
pub const MEL_APPLY_ENTRY: &str = "mel_apply";

/// Compatibility allocation wrapper around the allocation-free signal oracle.
pub fn mel_apply_cpu(spectrum: &[f32], mel_fb: &[f32], n_frames: usize,
                     n_bins: usize, n_mel: usize) -> Vec<f32> {
    let shape = MelShape { n_frames, n_bins, n_mel };
    let prepared = PreparedMel::new(shape, mel_fb).expect("mel filterbank shape");
    assert_eq!(spectrum.len(), shape.spectrum_len().expect("mel spectrum shape"));
    let mut out = vec![0.0f32; shape.output_len().expect("mel output shape")];
    prepared.apply_into(spectrum, &mut out).expect("mel spectrum shape");
    out
}
/// Checks the f32 parameter ABI and scratch sizes before allocating device buffers.
pub fn mel_gpu_capacity(spectrum: &[f32], mel_fb: &[f32], shape: MelShape)
                        -> Result<usize, ForgeError> {
    let invalid = || ForgeError::GpuValidation("mel_apply: invalid or oversized shape".into());
    if shape.n_frames == 0 || shape.n_bins == 0 || shape.n_mel == 0 { return Err(invalid()); }
    let spec_len = shape.spectrum_len().map_err(|_| invalid())?;
    let fb_len = shape.filterbank_len().map_err(|_| invalid())?;
    let out_len = shape.output_len().map_err(|_| invalid())?;
    if spectrum.len() != spec_len || mel_fb.len() != fb_len
        || [shape.n_frames, shape.n_bins, shape.n_mel].iter().any(|&n| n > (1 << 24))
        || [spec_len, fb_len, out_len].iter().any(|&n| n > u32::MAX as usize) {
        return Err(invalid());
    }
    spec_len.checked_add(fb_len).and_then(|n| n.checked_add(out_len))
        .and_then(|n| n.checked_add(4)).and_then(|n| n.checked_mul(4))
        .map(|n| n.max(4 << 20)).ok_or_else(invalid)
}
/// Dispatches on a supplied context, without adapter selection.
pub fn mel_apply_with_context(ctx: &mut crate::numerical::execute::WgpuComputeContext,
    spectrum: &[f32], mel_fb: &[f32], n_frames: usize, n_bins: usize, n_mel: usize,
) -> Result<Vec<f32>, ForgeError> {
    use crate::numerical::execute::{BindingUsage, QualiaCompute, WgpuPipeline};
    use crate::numerical::Schedule;
    mel_gpu_capacity(spectrum, mel_fb, MelShape { n_frames, n_bins, n_mel })?;
    let out_len = n_frames * n_mel;
    let view_spectrum = ctx.allocate_and_write(
        bytemuck::cast_slice(spectrum),
        0,
        0,
        BindingUsage::StorageRead,
    )?;
    let view_fb = ctx.allocate_and_write(
        bytemuck::cast_slice(mel_fb),
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
    let params = [n_frames as f32, n_bins as f32, n_mel as f32];
    let view_params = ctx.allocate_and_write(
        bytemuck::cast_slice(&params),
        3,
        0,
        BindingUsage::StorageRead,
    )?;

    let buffers = [view_spectrum, view_fb, view_out, view_params];
    let pipeline = WgpuPipeline::compile(ctx, MEL_APPLY_WGSL, MEL_APPLY_ENTRY)?;
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
#[path = "mel_tests.rs"]
mod tests;
