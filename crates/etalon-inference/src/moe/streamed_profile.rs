//! Geometry and supported shared-expert arithmetic for streamed SwiGLU.
use super::streamed_executor::StreamedMoeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedExpertGate {
    /// One same-intermediate-width shared SwiGLU expert scaled by sigmoid(w*x).
    SigmoidInputDotSameWidth,
}

/// Selected-logit softmax, SwiGLU and ordered weighted sums; top-k is const `K`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamedSwiGluProfile {
    pub hidden: usize,
    pub intermediate: usize,
    pub expert_count: usize,
    pub shared_expert: SharedExpertGate,
}

impl StreamedSwiGluProfile {
    pub(super) fn validate<const K: usize>(self) -> Result<(), StreamedMoeError> {
        if self.hidden == 0 || self.intermediate == 0 || K == 0 || K > self.expert_count {
            return Err(StreamedMoeError::InvalidShape);
        }
        // Bound flattened element addressing before any reader call.
        self.hidden
            .checked_mul(self.intermediate)
            .and_then(|plane| plane.checked_mul(self.expert_count))
            .ok_or(StreamedMoeError::InvalidShape)?;
        Ok(())
    }
}

// Native readers flatten expert planes into rows before calculating byte offsets.
pub(super) fn validate_tensor_span(
    tensor: &etalon_model::inference::TensorInfo,
) -> Result<(), StreamedMoeError> {
    let width = usize::try_from(tensor.dims[0]).map_err(|_| StreamedMoeError::InvalidShape)?;
    if let Some(layout) = etalon_model::ggml::ggml_block_layout(tensor.ggml_type) {
        width
            .div_ceil(layout.block_elems)
            .checked_mul(layout.block_bytes)
            .ok_or(StreamedMoeError::InvalidShape)?;
    }
    let row_bytes = etalon_model::ggml::ggml_row_bytes(tensor.ggml_type, width).ok_or(
        StreamedMoeError::Trunk(crate::weights::TrunkNvmeError::InvalidTensor),
    )?;
    let mut rows = 1usize;
    for &dimension in &tensor.dims[1..tensor.n_dims as usize] {
        let dimension = usize::try_from(dimension).map_err(|_| StreamedMoeError::InvalidShape)?;
        rows = rows
            .checked_mul(dimension)
            .ok_or(StreamedMoeError::InvalidShape)?;
    }
    rows.checked_mul(row_bytes)
        .ok_or(StreamedMoeError::InvalidShape)?;
    Ok(())
}
