use etalon_model::inference::{LayerTensors, ModelHyperparams};
/// Maximum supported SSM state width (d_state).
pub const MAX_SSM_STATE: usize = 256;
/// Maximum supported SSM head dimension (d_head).
pub const MAX_SSM_HEAD: usize = 256;
/// Maximum supported SSM inner (expanded) dimension.
pub const MAX_SSM_INNER: usize = 8192;
/// Causal convolution history depth (CAUSAL_CONV_KERNEL - 1 = 3).
pub const CONV_HISTORY_DEPTH: usize = 3;

/// Per-layer recurrent state layout stored contiguously in ssm_recurrent_state:
/// [conv_history: CONV_HISTORY_DEPTH * inner_dim f32]
/// [delta_state : d_state * d_head * n_heads f32   ]
#[derive(Clone, Copy, Debug)]
pub struct SsmLayerStateLayout {
    /// Layer stride in f32 elements.
    pub stride: usize,
    /// Offset of conv history region within the stride (always 0).
    #[allow(dead_code)]
    pub conv_offset: usize,
    /// Length of conv history region.
    pub conv_len: usize,
    /// Offset of GatedDeltaNet state matrix.
    pub delta_offset: usize,
    /// Length of GatedDeltaNet state matrix.
    pub delta_len: usize,
}

impl SsmLayerStateLayout {
    /// Creates `SsmLayerStateLayout` with the given state.
    pub fn new(inner_dim: usize, d_state: usize, d_head: usize, n_heads: usize) -> Self {
        let conv_len = CONV_HISTORY_DEPTH * inner_dim;
        let delta_len = d_state * d_head * n_heads;
        let delta_offset = conv_len;
        let stride = conv_len + delta_len;
        Self {
            stride,
            conv_offset: 0,
            conv_len,
            delta_offset,
            delta_len,
        }
    }
}

/// Derive the SsmLayerStateLayout from a layer's tensor metadata.
pub fn ssm_layout_for(
    tensors: &LayerTensors,
    h: &ModelHyperparams,
    n_embd: usize,
) -> SsmLayerStateLayout {
    let inner_dim = if h.ssm_inner_size > 0 {
        h.ssm_inner_size as usize
    } else if let Some(info) = tensors.ssm_conv1d.as_ref() {
        if info.n_dims >= 3 && info.dims[2] > 0 {
            info.dims[2] as usize
        } else {
            info.dims[0] as usize
        }
    } else if let Some(info) = tensors.ssm_in.as_ref().or(tensors.attn_qkv.as_ref()) {
        let (n_in, n_out) = super::matmul_dims(info);
        if tensors.attn_qkv.is_some() {
            n_out.min(MAX_SSM_INNER)
        } else {
            n_in
        }
    } else {
        n_embd * 2
    };
    let inner_dim = inner_dim.min(MAX_SSM_INNER).max(1);

    let d_state = if h.ssm_state_size > 0 {
        h.ssm_state_size as usize
    } else {
        128
    }
    .min(MAX_SSM_STATE)
    .max(1);

    let d_head = if h.head_dim() > 0 {
        h.head_dim() as usize
    } else {
        d_state
    }
    .min(MAX_SSM_HEAD)
    .max(1);

    let n_heads = if h.ssm_group_count > 0 {
        h.ssm_group_count as usize
    } else {
        let alpha_dim = tensors
            .ssm_alpha
            .map(|i| {
                if i.dims[1] > 0 {
                    i.dims[1] as usize
                } else {
                    i.dims[0] as usize
                }
            })
            .unwrap_or(0);
        if d_head > 0 && alpha_dim > 0 {
            alpha_dim / d_head
        } else {
            16
        }
    }
    .max(1);

    SsmLayerStateLayout::new(inner_dim, d_state, d_head, n_heads)
}
