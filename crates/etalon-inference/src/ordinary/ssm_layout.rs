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
pub fn checked_ssm_layout_for(
    tensors: &LayerTensors,
    h: &ModelHyperparams,
    n_embd: usize,
) -> Result<SsmLayerStateLayout, SsmStateError> {
    // The legacy Qwen kernel does not implement packed GQA key/value channels.
    if h.architecture == etalon_model::inference::ARCH_QWEN35_MOE {
        return Err(SsmStateError::UnsupportedArchitecture);
    }
    if h.ssm_conv_kernel != 0 && h.ssm_conv_kernel != 4 {
        return Err(SsmStateError::UnsupportedDimensions);
    }
    let inner_dim = if h.ssm_inner_size > 0 {
        h.ssm_inner_size as usize
    } else if let Some(info) = tensors.ssm_conv1d.as_ref() {
        if info.n_dims >= 3 && info.dims[2] > 0 {
            usize::try_from(info.dims[2]).map_err(|_| SsmStateError::SizeOverflow)?
        } else {
            usize::try_from(info.dims[0]).map_err(|_| SsmStateError::SizeOverflow)?
        }
    } else if let Some(info) = tensors.ssm_in.as_ref().or(tensors.attn_qkv.as_ref()) {
        let n_in = usize::try_from(info.dims[0]).map_err(|_| SsmStateError::SizeOverflow)?;
        let n_out = usize::try_from(if info.n_dims > 1 && info.dims[1] > 0 {
            info.dims[1]
        } else {
            1
        })
        .map_err(|_| SsmStateError::SizeOverflow)?;
        if tensors.attn_qkv.is_some() {
            n_out
        } else {
            n_in
        }
    } else {
        n_embd.checked_mul(2).ok_or(SsmStateError::SizeOverflow)?
    };
    if inner_dim == 0 || inner_dim > MAX_SSM_INNER {
        return Err(SsmStateError::UnsupportedDimensions);
    }

    let d_state = if h.ssm_state_size > 0 {
        h.ssm_state_size as usize
    } else {
        128
    };
    if d_state == 0 || d_state > MAX_SSM_STATE {
        return Err(SsmStateError::UnsupportedDimensions);
    }

    let d_head = if h.head_dim() > 0 {
        h.head_dim() as usize
    } else {
        d_state
    };
    if d_head == 0 || d_head > MAX_SSM_HEAD {
        return Err(SsmStateError::UnsupportedDimensions);
    }

    let n_heads = if h.ssm_group_count > 0 {
        h.ssm_group_count as usize
    } else {
        let alpha_dim = if let Some(i) = tensors.ssm_alpha {
            usize::try_from(if i.dims[1] > 0 { i.dims[1] } else { i.dims[0] })
                .map_err(|_| SsmStateError::SizeOverflow)?
        } else {
            0
        };
        if d_head > 0 && alpha_dim > 0 {
            alpha_dim / d_head
        } else {
            16
        }
    }
    .max(1);

    if n_heads > MAX_SSM_HEAD {
        return Err(SsmStateError::UnsupportedDimensions);
    }
    let conv_len = CONV_HISTORY_DEPTH
        .checked_mul(inner_dim)
        .ok_or(SsmStateError::SizeOverflow)?;
    let delta_len = d_state
        .checked_mul(d_head)
        .and_then(|n| n.checked_mul(n_heads))
        .ok_or(SsmStateError::SizeOverflow)?;
    let stride = conv_len
        .checked_add(delta_len)
        .ok_or(SsmStateError::SizeOverflow)?;
    Ok(SsmLayerStateLayout {
        stride,
        conv_offset: 0,
        conv_len,
        delta_offset: conv_len,
        delta_len,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsmStateError {
    UnsupportedDimensions,
    UnsupportedArchitecture,
    SizeOverflow,
    BudgetExceeded { required: usize, budget: usize },
    AllocationFailed,
}
impl std::fmt::Display for SsmStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "recurrent state admission: {self:?}")
    }
}
impl std::error::Error for SsmStateError {}

/// Legacy layout helper; execution and cold admission validate with the checked API.
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

/// Cold external model state, independent of the Sentinel arena.
pub fn allocate_recurrent_state(
    strides: impl IntoIterator<Item = Result<usize, SsmStateError>>,
    byte_budget: usize,
) -> Result<Box<[f32]>, SsmStateError> {
    let mut total = 0usize;
    for stride in strides {
        total = total
            .checked_add(stride?)
            .ok_or(SsmStateError::SizeOverflow)?;
    }
    let bytes = total
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(SsmStateError::SizeOverflow)?;
    if bytes > byte_budget {
        return Err(SsmStateError::BudgetExceeded {
            required: bytes,
            budget: byte_budget,
        });
    }
    let mut state = Vec::new();
    state
        .try_reserve_exact(total)
        .map_err(|_| SsmStateError::AllocationFailed)?;
    state.resize(total, 0.0);
    Ok(state.into_boxed_slice())
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    #[test]
    fn exact_budget_zero_state_and_overflow() {
        let state = allocate_recurrent_state([Ok(3), Ok(5)], 32).unwrap();
        assert_eq!(&*state, &[0.0; 8]);
        assert_eq!(
            allocate_recurrent_state([Ok(8)], 31).unwrap_err(),
            SsmStateError::BudgetExceeded {
                required: 32,
                budget: 31
            }
        );
        assert!(allocate_recurrent_state([], 0).unwrap().is_empty());
        assert_eq!(
            allocate_recurrent_state([Ok(usize::MAX), Ok(1)], usize::MAX).unwrap_err(),
            SsmStateError::SizeOverflow
        );
        assert_eq!(
            allocate_recurrent_state([Ok(usize::MAX)], usize::MAX).unwrap_err(),
            SsmStateError::SizeOverflow
        );
    }
    #[test]
    fn explicit_external_budget_can_exceed_legacy_ceiling() {
        let n = 36 * 1024 * 1024 / 4 + 1;
        assert!(matches!(
            allocate_recurrent_state([Ok(n)], 36 * 1024 * 1024),
            Err(SsmStateError::BudgetExceeded { .. })
        ));
        let state = allocate_recurrent_state([Ok(n)], n * 4).unwrap();
        assert_eq!(state.len(), n);
        assert!(state.iter().all(|value| *value == 0.0));
    }
    #[test]
    fn qwen35_cannot_bypass_kernel_support_via_experimental_override() {
        let mut h = ModelHyperparams::default();
        h.architecture = etalon_model::inference::ARCH_QWEN35_MOE;
        assert_eq!(
            checked_ssm_layout_for(&LayerTensors::default(), &h, 512).unwrap_err(),
            SsmStateError::UnsupportedArchitecture
        );
    }
    #[test]
    fn unsupported_dimensions_are_refused_without_clamping() {
        let t = LayerTensors::default();
        let mut h = ModelHyperparams::default();
        h.ssm_inner_size = 1024;
        h.ssm_state_size = 64;
        h.ssm_group_count = 4;
        h.head_dim = 128;
        let layout = checked_ssm_layout_for(&t, &h, 512).unwrap();
        assert_eq!(layout.stride, 3 * 1024 + 64 * 128 * 4);
        for (inner, state, head, groups) in [
            (8193, 64, 128, 4),
            (1024, 257, 128, 4),
            (1024, 64, 257, 4),
            (1024, 64, 128, u32::MAX),
        ] {
            h.ssm_inner_size = inner;
            h.ssm_state_size = state;
            h.head_dim = head;
            h.ssm_group_count = groups;
            assert_eq!(
                checked_ssm_layout_for(&t, &h, 512).unwrap_err(),
                SsmStateError::UnsupportedDimensions
            );
        }
    }
}
