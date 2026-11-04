use super::*;
/// Owned mutable state for a live Qwen4Exp sequence.  Allocated once at
/// session start; the per-token step only mutates slices inside it.
pub struct Qwen4ExpSession {
    pub(super) hidden: usize,
    pub(super) n_layers: usize,
    pub(super) residual: Vec<f32>,
    /// GatedDeltaNet state per layer: 3-deep conv history + 48×128×128 delta.
    pub(super) gdn_convolution: Vec<Vec<f32>>,
    pub(super) gdn_delta: Vec<Vec<f32>>,
    /// QSA state per layer: rotated K, raw V, raw 128-dim indexer keys.
    pub(super) qsa_keys: Vec<Vec<f32>>,
    pub(super) qsa_values: Vec<Vec<f32>>,
    pub(super) qsa_indexer_keys: Vec<Vec<f32>>,
    pub(super) qsa_tokens: Vec<usize>,
    /// PLE dilated-conv input history (`(kernel-1) * dilation` wide entries).
    pub(super) ple_conv_history: Vec<f32>,
    pub(super) ple_tokens: PleTokenHistory,
    pub(super) ple_layer: Option<u32>,
    /// Tokens processed so far (prompt + generated).
    pub position: usize,
}

impl Qwen4ExpSession {
    /// Allocate persistent state for every layer in the validated graph.
    /// `qsa_capacity` bounds each QSA layer's retained context in tokens.
    pub fn new<H: super::super::QwenHyperparams>(
        index: &TensorIndex<H>,
        qsa_capacity: usize,
    ) -> Result<Self, Qwen4ExpDecodeError> {
        Self::new_with_budget(index, qsa_capacity, u64::MAX)
    }

    pub fn new_with_budget<H: super::super::QwenHyperparams>(
        index: &TensorIndex<H>,
        qsa_capacity: usize,
        byte_budget: u64,
    ) -> Result<Self, Qwen4ExpDecodeError> {
        let hyperparams: etalon_model::inference::ModelHyperparams = index.hyperparams.into();
        let hidden = index.emb_dim();
        let n_layers = hyperparams.n_layer as usize;
        if hidden == 0 || n_layers == 0 || qsa_capacity == 0 {
            return Err(Qwen4ExpDecodeError::MissingTensor);
        }
        let ple_config = index.ple_config.ok_or(Qwen4ExpDecodeError::MissingTensor)?;
        let ple_layer = (ple_config.layer_count > 0).then(|| ple_config.layers[0]);
        let wide = hidden
            .checked_mul(QWEN4EXP_HYPER_STREAMS)
            .ok_or(Qwen4ExpDecodeError::SizeOverflow)?;
        let ple_history = wide
            .checked_mul(QWEN4EXP_PLE_HISTORY)
            .ok_or(Qwen4ExpDecodeError::SizeOverflow)?;
        let qsa_kv = qsa_capacity
            .checked_mul(QSA_KV_WIDTH)
            .ok_or(Qwen4ExpDecodeError::SizeOverflow)?;
        let qsa_index = qsa_capacity
            .checked_mul(QWEN4EXP_QSA_INDEXER_HEAD_DIM)
            .ok_or(Qwen4ExpDecodeError::SizeOverflow)?;
        let mut required = 0usize;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, ple_history, 4)?;
        allocation::add_bytes(
            &mut required,
            n_layers,
            5 * core::mem::size_of::<Vec<f32>>() + core::mem::size_of::<usize>(),
        )?;
        for layer in 0..n_layers as u32 {
            let tensors = index.get_layer_tensors(layer);
            if tensors.is_hybrid_ssm_layer() {
                allocation::add_bytes(
                    &mut required,
                    GDN_CONV_WIDTH * GDN_CONV_HISTORY + GDN_DELTA_WIDTH,
                    4,
                )?;
            }
            if tensors.has_qwen_sparse_attention() {
                allocation::add_bytes(&mut required, qsa_kv, 8)?;
                allocation::add_bytes(&mut required, qsa_index, 4)?;
            }
        }
        allocation::admit(required, byte_budget)?;
        let mut session = Self {
            hidden,
            n_layers,
            residual: allocation::zeros(wide)?,
            gdn_convolution: allocation::capacity(n_layers)?,
            gdn_delta: allocation::capacity(n_layers)?,
            qsa_keys: allocation::capacity(n_layers)?,
            qsa_values: allocation::capacity(n_layers)?,
            qsa_indexer_keys: allocation::capacity(n_layers)?,
            qsa_tokens: allocation::capacity(n_layers)?,
            ple_conv_history: allocation::zeros(ple_history)?,
            ple_tokens: PleTokenHistory::default(),
            ple_layer,
            position: 0,
        };
        for layer_idx in 0..n_layers as u32 {
            let layer = index.get_layer_tensors(layer_idx);
            if layer.is_hybrid_ssm_layer() {
                session
                    .gdn_convolution
                    .push(allocation::zeros(GDN_CONV_WIDTH * GDN_CONV_HISTORY)?);
                session.gdn_delta.push(allocation::zeros(GDN_DELTA_WIDTH)?);
            } else {
                session.gdn_convolution.push(Vec::new());
                session.gdn_delta.push(Vec::new());
            }
            if layer.has_qwen_sparse_attention() {
                session.qsa_keys.push(allocation::zeros(qsa_kv)?);
                session.qsa_values.push(allocation::zeros(qsa_kv)?);
                session.qsa_indexer_keys.push(allocation::zeros(qsa_index)?);
                session.qsa_tokens.push(0);
            } else {
                session.qsa_keys.push(Vec::new());
                session.qsa_values.push(Vec::new());
                session.qsa_indexer_keys.push(Vec::new());
                session.qsa_tokens.push(0);
            }
        }
        Ok(session)
    }

    /// Resident bytes held by the persistent per-layer state and residual.
    pub fn state_bytes(&self) -> u64 {
        let mut total = (self.residual.len() + self.ple_conv_history.len()) as u64 * 4;
        for layer in 0..self.n_layers {
            total += (self.gdn_convolution[layer].len()
                + self.gdn_delta[layer].len()
                + self.qsa_keys[layer].len()
                + self.qsa_values[layer].len()
                + self.qsa_indexer_keys[layer].len()) as u64
                * 4;
        }
        total
    }
}
