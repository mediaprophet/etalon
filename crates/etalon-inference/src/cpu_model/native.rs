use super::{empty_layer, BindingLimits, DenseCpuError as Error, PackedTensor, MAX_INPUT_WIDTH};
use etalon_inference_kernel::dense_cpu::{DenseConfig, DenseLayer, DenseModel, TensorBinding};
use etalon_model::{
    ggml,
    inference::{ModelHyperparams, PreparedP64, TensorInfo, ARCH_LLAMA},
};

fn binding<'a>(
    bytes: &'a [u8],
    info: TensorInfo,
    width: usize,
    rows: usize,
    rank: u32,
) -> Result<TensorBinding<PackedTensor<'a>>, Error> {
    if info.n_dims != rank
        || info.dims[0] != width as u64
        || (rank == 2 && info.dims[1] != rows as u64)
        || width == 0
        || width > MAX_INPUT_WIDTH
    {
        return Err(Error::InvalidDimensions);
    }
    let row_bytes = ggml::ggml_row_bytes(info.ggml_type, width).ok_or(Error::UnsupportedTensor)?;
    let length = row_bytes
        .checked_mul(rows)
        .ok_or(Error::ArithmeticOverflow)?;
    let start = usize::try_from(info.byte_offset).map_err(|_| Error::ArithmeticOverflow)?;
    let end = start.checked_add(length).ok_or(Error::ArithmeticOverflow)?;
    let bytes = bytes.get(start..end).ok_or(Error::Payload)?;
    if info.ggml_type == ggml::GGML_TYPE_Q6_K && !(bytes.as_ptr() as usize).is_multiple_of(2) {
        return Err(Error::Alignment);
    }
    Ok(TensorBinding {
        handle: PackedTensor {
            bytes,
            dtype: info.ggml_type,
            width,
            rows,
            row_bytes,
        },
        input_width: width,
        output_rows: rows,
    })
}

fn layer<'a>(
    model: &PreparedP64<'a>,
    index: usize,
    c: DenseConfig,
) -> Result<DenseLayer<PackedTensor<'a>>, Error> {
    let t = model.index().get_layer_tensors(index as u32);
    // These optional tensors require a richer architecture executor.
    if t.attn_qkv.is_some()
        || t.attn_q_norm.is_some()
        || t.attn_k_norm.is_some()
        || t.moe_router.is_some()
        || t.is_hybrid_ssm_layer()
    {
        return Err(Error::UnsupportedProfile);
    }
    let mut result = empty_layer();
    let kv = c.kv_heads * c.head_dim;
    for (slot, info, width, rows, rank) in [
        (&mut result.attn_norm, t.attn_norm, c.embedding, 1, 1),
        (&mut result.q, t.attn_q, c.embedding, c.embedding, 2),
        (&mut result.k, t.attn_k, c.embedding, kv, 2),
        (&mut result.v, t.attn_v, c.embedding, kv, 2),
        (
            &mut result.output,
            t.attn_output,
            c.embedding,
            c.embedding,
            2,
        ),
        (&mut result.ffn_norm, t.ffn_norm, c.embedding, 1, 1),
        (&mut result.gate, t.ffn_gate, c.embedding, c.ffn, 2),
        (&mut result.up, t.ffn_up, c.embedding, c.ffn, 2),
        (&mut result.down, t.ffn_down, c.ffn, c.embedding, 2),
    ] {
        *slot = Some(binding(
            model.view().bytes(),
            info.ok_or(Error::MissingTensor)?,
            width,
            rows,
            rank,
        )?);
    }
    Ok(result)
}

/// Bind native P64 payloads directly to the complete dense CPU executor.
/// No intermediate container or model payload allocation is performed.
pub fn prepare_dense_p64_into<'a, 's>(
    model: &PreparedP64<'a>,
    context: usize,
    storage: &'s mut [DenseLayer<PackedTensor<'a>>],
    limits: BindingLimits,
) -> Result<DenseModel<'s, PackedTensor<'a>>, Error> {
    let index = model.index();
    let h: ModelHyperparams = index.hyperparams;
    if h.architecture != ARCH_LLAMA
        || h.arch_flags != 0
        || h.sliding_window != 0
        || h.shared_kv_layers != 0
        || h.logit_softcap != 0.0
    {
        return Err(Error::UnsupportedProfile);
    }
    let layers = h.n_layer as usize;
    if layers == 0 || storage.len() < layers {
        return Err(Error::Capacity);
    }
    if core::mem::size_of_val(storage) > limits.byte_budget {
        return Err(Error::Budget);
    }
    let count = index.entries.len() as u64;
    let scans = (layers as u64)
        .checked_mul(18)
        .and_then(|n| n.checked_add(4))
        .ok_or(Error::ArithmeticOverflow)?;
    let nested = (layers as u64)
        .checked_mul(9)
        .and_then(|n| n.checked_add(4))
        .ok_or(Error::ArithmeticOverflow)?;
    let visits = count
        .checked_mul(scans)
        .and_then(|n| {
            count
                .checked_mul(count)
                .and_then(|m| m.checked_mul(nested))
                .and_then(|m| n.checked_add(m))
        })
        .ok_or(Error::ArithmeticOverflow)?;
    if visits > limits.max_tensor_visits {
        return Err(Error::Budget);
    }
    let emb = index.token_embd_info().ok_or(Error::MissingTensor)?;
    let embedding = h.n_embd as usize;
    let vocabulary = usize::try_from(emb.dims[1]).map_err(|_| Error::ArithmeticOverflow)?;
    let heads = h.n_head as usize;
    let kv_heads = h.effective_n_kv_head() as usize;
    let head_dim = h.head_dim() as usize;
    let gate = index
        .get_layer_tensors(0)
        .ffn_gate
        .ok_or(Error::MissingTensor)?;
    let ffn = usize::try_from(gate.dims[1]).map_err(|_| Error::ArithmeticOverflow)?;
    if embedding == 0
        || embedding > MAX_INPUT_WIDTH
        || ffn == 0
        || ffn > MAX_INPUT_WIDTH
        || heads == 0
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || heads.checked_mul(head_dim) != Some(embedding)
        || !head_dim.is_multiple_of(2)
        || vocabulary == 0
        || vocabulary > u32::MAX as usize
        || context == 0
        || context > u32::MAX as usize
    {
        return Err(Error::InvalidDimensions);
    }
    let c = DenseConfig {
        embedding,
        ffn,
        heads,
        kv_heads,
        head_dim,
        max_context: context,
        vocabulary,
        rope_base: h.effective_rope_freq_base(),
        rope_scale: h.effective_rope_scale(),
    };
    // Refuse every tensor the strict dense executor cannot consume.
    for (entry_pos, (hash, info)) in index.entries.iter().enumerate() {
        if index.entries[..entry_pos]
            .iter()
            .any(|(prior, _)| prior == hash)
        {
            return Err(Error::DuplicateTensor);
        }
        let equal = |candidate: Option<&TensorInfo>| {
            candidate.is_some_and(|t| {
                t.dims == info.dims
                    && t.n_dims == info.n_dims
                    && t.ggml_type == info.ggml_type
                    && t.byte_offset == info.byte_offset
            })
        };
        let mut known = equal(index.token_embd_info())
            || equal(index.output_norm_info())
            || equal(index.output_weight_info());
        for layer in 0..layers {
            let t = index.get_layer_tensors(layer as u32);
            known |= [
                t.attn_norm,
                t.attn_q,
                t.attn_k,
                t.attn_v,
                t.attn_output,
                t.ffn_norm,
                t.ffn_gate,
                t.ffn_up,
                t.ffn_down,
            ]
            .iter()
            .any(|t| equal(t.as_ref()));
        }
        if !known {
            return Err(Error::UnsupportedTensor);
        }
    }
    if !c.rope_base.is_finite()
        || c.rope_base <= 0.0
        || !c.rope_scale.is_finite()
        || c.rope_scale <= 0.0
    {
        return Err(Error::InvalidDimensions);
    }
    let bytes = model.view().bytes();
    let embedding = binding(bytes, *emb, embedding, vocabulary, 2)?;
    let output_norm = index
        .output_norm_info()
        .map(|t| binding(bytes, *t, c.embedding, 1, 1))
        .transpose()?;
    let output = index
        .output_weight_info()
        .map(|t| binding(bytes, *t, c.embedding, vocabulary, 2))
        .transpose()?
        .unwrap_or(embedding);
    for i in 0..layers {
        layer(model, i, c)?;
    }
    for (i, slot) in storage[..layers].iter_mut().enumerate() {
        *slot = layer(model, i, c)?;
    }
    Ok(DenseModel {
        config: c,
        layers: &storage[..layers],
        embedding,
        output_norm,
        output: Some(output),
    })
}
