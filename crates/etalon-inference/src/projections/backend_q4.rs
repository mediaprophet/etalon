//! Cold legacy native Q4 projection adapters. Explicit payload admission precedes densification.
//! Driver/compiler metadata remains owned by the selected native backend, not the Sentinel arena.
use super::ProjectionTensor;

struct Geometry<'a> {
    raw: &'a [u8],
    rows: usize,
    cols: usize,
    batches: usize,
    elements: usize,
}
fn geometry<'a>(
    model: &'a [u8],
    data_offset: u64,
    tensor: &ProjectionTensor<'_>,
    input: &[f32],
    block_size: usize,
    block_bytes: usize,
    byte_budget: usize,
) -> Option<Geometry<'a>> {
    let rows = tensor.shape.first().copied().unwrap_or(4096);
    let cols = tensor.shape.get(1).copied().unwrap_or(4096);
    if rows == 0 || cols == 0 || input.is_empty() || input.len() % cols != 0 {
        return None;
    }
    let batches = input.len() / cols;
    if [rows, cols, batches]
        .into_iter()
        .any(|n| n > i32::MAX as usize)
    {
        return None;
    }
    let elements = rows.checked_mul(cols)?;
    if elements % block_size != 0 {
        return None;
    }
    let raw_bytes = (elements / block_size).checked_mul(block_bytes)?;
    let output_elements = rows.checked_mul(batches)?;
    // Dense CPU weight/output plus explicit native upload/output/readback payloads.
    let payload = elements
        .checked_mul(2)?
        .checked_add(input.len())?
        .checked_add(output_elements.checked_mul(3)?)?
        .checked_mul(4)?;
    if payload > byte_budget {
        return None;
    }
    let offset = usize::try_from(data_offset.checked_add(tensor.byte_offset)?).ok()?;
    let raw = model.get(offset..offset.checked_add(raw_bytes)?)?;
    Some(Geometry {
        raw,
        rows,
        cols,
        batches,
        elements,
    })
}

#[cfg(all(target_os = "windows", feature = "native-directml"))]
pub fn directml(
    device: &crate::backends::directml::DmlDevice,
    model: &[u8],
    data_offset: u64,
    tensor: &ProjectionTensor<'_>,
    input: &[f32],
    byte_budget: usize,
) -> Option<Vec<f32>> {
    use crate::backends::directml::{
        dequantize_q4_k_tensor, DmlGemmOp, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE,
    };
    let plan = geometry(
        model,
        data_offset,
        tensor,
        input,
        Q4_K_BLOCK_SIZE,
        Q4_K_BLOCK_BYTES,
        byte_budget,
    )?;
    let weights = dequantize_q4_k_tensor(plan.raw, plan.elements);
    DmlGemmOp {
        m: plan.batches as u32,
        k: plan.cols as u32,
        n: plan.rows as u32,
    }
    .execute(device, input, &weights)
    .ok()
}

#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    feature = "native-accelerate"
))]
pub fn accelerate(
    model: &[u8],
    data_offset: u64,
    tensor: &ProjectionTensor<'_>,
    input: &[f32],
    byte_budget: usize,
) -> Option<Vec<f32>> {
    use crate::backends::native_accelerate::{
        accelerate_sgemm, dequantize_q4_k_tensor, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE,
    };
    let plan = geometry(
        model,
        data_offset,
        tensor,
        input,
        Q4_K_BLOCK_SIZE,
        Q4_K_BLOCK_BYTES,
        byte_budget,
    )?;
    let weights = dequantize_q4_k_tensor(plan.raw, plan.elements);
    Some(accelerate_sgemm(
        plan.batches,
        plan.cols,
        plan.rows,
        input,
        &weights,
    ))
}
