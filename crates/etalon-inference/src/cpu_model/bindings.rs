use super::{DenseGgufConfig, GgufCpuError as Error, MAX_INPUT_WIDTH};
use etalon_inference_kernel::dense_cpu::{DenseLayer, DenseModel, TensorBinding};
use etalon_model::{ggml, gguf::TensorView};

#[derive(Clone, Copy, Debug)]
pub struct PackedTensor<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) dtype: u32,
    pub(crate) width: usize,
    pub(crate) rows: usize,
    pub(crate) row_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct BindingLimits {
    pub byte_budget: usize,
    pub max_tensor_visits: u64,
}
pub const fn empty_layer<'a>() -> DenseLayer<PackedTensor<'a>> {
    DenseLayer {
        attn_norm: None,
        q: None,
        k: None,
        v: None,
        output: None,
        ffn_norm: None,
        gate: None,
        up: None,
        down: None,
    }
}

fn pack<'a>(
    config: &DenseGgufConfig<'a>,
    t: TensorView<'a>,
    width: usize,
    rows: usize,
    rank: u32,
) -> Result<TensorBinding<PackedTensor<'a>>, Error> {
    if t.rank != rank
        || t.dimension(0) != Some(width as u64)
        || (rank == 2 && t.dimension(1) != Some(rows as u64))
        || width == 0
        || width > MAX_INPUT_WIDTH
    {
        return Err(Error::InvalidDimensions);
    }
    let row_bytes = ggml::ggml_row_bytes(t.ggml_type, width).ok_or(Error::UnsupportedTensor)?;
    let length = row_bytes
        .checked_mul(rows)
        .ok_or(Error::ArithmeticOverflow)?;
    let start = usize::try_from(t.offset)
        .ok()
        .and_then(|n| config.data_start.checked_add(n))
        .ok_or(Error::ArithmeticOverflow)?;
    let end = start.checked_add(length).ok_or(Error::ArithmeticOverflow)?;
    let bytes = config.bytes.get(start..end).ok_or(Error::Payload)?;
    if t.ggml_type == ggml::GGML_TYPE_Q6_K && !(bytes.as_ptr() as usize).is_multiple_of(2) {
        return Err(Error::Alignment);
    }
    Ok(TensorBinding {
        handle: PackedTensor {
            bytes,
            dtype: t.ggml_type,
            width,
            rows,
            row_bytes,
        },
        input_width: width,
        output_rows: rows,
    })
}
fn put<T>(slot: &mut Option<T>, value: T) -> Result<(), Error> {
    if slot.replace(value).is_some() {
        Err(Error::DuplicateTensor)
    } else {
        Ok(())
    }
}
fn layer_role(name: &[u8]) -> Result<Option<(usize, &[u8])>, Error> {
    let Some(rest) = name.strip_prefix(b"blk.") else {
        return Ok(None);
    };
    let dot = rest
        .iter()
        .position(|&b| b == b'.')
        .ok_or(Error::UnsupportedTensor)?;
    if dot == 0 {
        return Err(Error::UnsupportedTensor);
    }
    let mut layer = 0usize;
    for &digit in &rest[..dot] {
        if !digit.is_ascii_digit() {
            return Err(Error::UnsupportedTensor);
        }
        layer = layer
            .checked_mul(10)
            .and_then(|n| n.checked_add(usize::from(digit - b'0')))
            .ok_or(Error::ArithmeticOverflow)?;
    }
    Ok(Some((layer, &rest[dot + 1..])))
}
fn layer<'a>(
    config: &DenseGgufConfig<'a>,
    index: usize,
) -> Result<DenseLayer<PackedTensor<'a>>, Error> {
    let c = config.config;
    let kv = c.kv_heads * c.head_dim;
    let mut output = empty_layer();
    for tensor in config.metadata.tensors() {
        let t = tensor?;
        let Some((i, role)) = layer_role(t.name)? else {
            continue;
        };
        if i >= config.layers {
            return Err(Error::InvalidDimensions);
        }
        if i != index {
            continue;
        }
        let (slot, width, rows, rank) = match role {
            b"attn_norm.weight" => (&mut output.attn_norm, c.embedding, 1, 1),
            b"attn_q.weight" => (&mut output.q, c.embedding, c.embedding, 2),
            b"attn_k.weight" => (&mut output.k, c.embedding, kv, 2),
            b"attn_v.weight" => (&mut output.v, c.embedding, kv, 2),
            b"attn_output.weight" => (&mut output.output, c.embedding, c.embedding, 2),
            b"ffn_norm.weight" => (&mut output.ffn_norm, c.embedding, 1, 1),
            b"ffn_gate.weight" => (&mut output.gate, c.embedding, c.ffn, 2),
            b"ffn_up.weight" => (&mut output.up, c.embedding, c.ffn, 2),
            b"ffn_down.weight" => (&mut output.down, c.ffn, c.embedding, 2),
            _ => return Err(Error::UnsupportedTensor),
        };
        put(slot, pack(config, t, width, rows, rank)?)?;
    }
    if [
        output.attn_norm,
        output.q,
        output.k,
        output.v,
        output.output,
        output.ffn_norm,
        output.gate,
        output.up,
        output.down,
    ]
    .iter()
    .any(Option::is_none)
    {
        return Err(Error::MissingTensor);
    }
    Ok(output)
}

/// Validates all bindings before writing caller storage; charges the full slice.
/// The two bounded scans per layer avoid allocating a descriptor index.
pub fn prepare_dense_gguf_into<'a, 's>(
    config: &DenseGgufConfig<'a>,
    storage: &'s mut [DenseLayer<PackedTensor<'a>>],
    limits: BindingLimits,
) -> Result<DenseModel<'s, PackedTensor<'a>>, Error> {
    let bytes = core::mem::size_of_val(storage);
    if bytes > limits.byte_budget {
        return Err(Error::Budget);
    }
    if storage.len() < config.layers {
        return Err(Error::Capacity);
    }
    let passes = (config.layers as u64)
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(Error::ArithmeticOverflow)?;
    let visits = passes
        .checked_mul(config.metadata.header().tensor_count)
        .ok_or(Error::ArithmeticOverflow)?;
    if visits > limits.max_tensor_visits {
        return Err(Error::Budget);
    }
    let c = config.config;
    let (mut embedding, mut output_norm, mut output) = (None, None, None);
    for tensor in config.metadata.tensors() {
        let t = tensor?;
        match t.name {
            b"token_embd.weight" => put(
                &mut embedding,
                pack(config, t, c.embedding, c.vocabulary, 2)?,
            )?,
            b"output.weight" => put(&mut output, pack(config, t, c.embedding, c.vocabulary, 2)?)?,
            b"output_norm.weight" => put(&mut output_norm, pack(config, t, c.embedding, 1, 1)?)?,
            _ if layer_role(t.name)?.is_some() => {}
            _ => return Err(Error::UnsupportedTensor),
        }
    }
    let embedding = embedding.ok_or(Error::MissingTensor)?;
    for index in 0..config.layers {
        layer(config, index)?;
    }
    for (index, slot) in storage[..config.layers].iter_mut().enumerate() {
        *slot = layer(config, index)?;
    }
    Ok(DenseModel {
        config: c,
        layers: &storage[..config.layers],
        embedding,
        output_norm,
        output: Some(output.unwrap_or(embedding)),
    })
}
