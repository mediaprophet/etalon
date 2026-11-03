use super::{GgufCpuError as Error, MAX_INPUT_WIDTH};
use etalon_inference_kernel::dense_cpu::DenseConfig;
use etalon_model::gguf::{KvView, MetadataView, ParseLimits};

#[derive(Clone, Debug)]
pub struct DenseGgufConfig<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) metadata: MetadataView<'a>,
    pub(super) data_start: usize,
    pub(super) layers: usize,
    pub(super) config: DenseConfig,
}
impl DenseGgufConfig<'_> {
    pub fn layer_count(&self) -> usize {
        self.layers
    }
    pub fn numerical_config(&self) -> DenseConfig {
        self.config
    }
}

fn integer(kv: KvView<'_>) -> Result<usize, Error> {
    let value = match kv.value_type {
        4 => u64::from(u32::from_le_bytes(
            kv.encoded_value
                .try_into()
                .map_err(|_| Error::MissingMetadata)?,
        )),
        10 => u64::from_le_bytes(
            kv.encoded_value
                .try_into()
                .map_err(|_| Error::MissingMetadata)?,
        ),
        _ => return Err(Error::UnsupportedProfile),
    };
    usize::try_from(value).map_err(|_| Error::ArithmeticOverflow)
}
fn float(kv: KvView<'_>) -> Result<f32, Error> {
    let value = match kv.value_type {
        6 => f32::from_le_bytes(
            kv.encoded_value
                .try_into()
                .map_err(|_| Error::MissingMetadata)?,
        ),
        12 => f64::from_le_bytes(
            kv.encoded_value
                .try_into()
                .map_err(|_| Error::MissingMetadata)?,
        ) as f32,
        4 => integer(kv)? as f32,
        _ => return Err(Error::UnsupportedProfile),
    };
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(Error::InvalidDimensions)
    }
}
fn assign<T>(slot: &mut Option<T>, value: T) -> Result<(), Error> {
    if slot.replace(value).is_some() {
        Err(Error::DuplicateMetadata)
    } else {
        Ok(())
    }
}

/// Supports the original dense Llama numerical profile with fixed RMS epsilon.
/// Parsing uses caller limits; tensor payloads are validated during preparation.
pub fn inspect_dense_gguf(
    bytes: &[u8],
    limits: ParseLimits,
    context: usize,
) -> Result<DenseGgufConfig<'_>, Error> {
    let metadata = MetadataView::parse(bytes, limits)?;
    let (
        mut architecture,
        mut alignment,
        mut layers,
        mut embedding,
        mut heads,
        mut kv_heads,
        mut ffn,
        mut base,
        mut scale,
        mut head_dim,
        mut epsilon,
    ) = (
        None, None, None, None, None, None, None, None, None, None, None,
    );
    let (mut value_dim, mut rotary_dim, mut trained_context, mut declared_vocab, mut scaling_type) =
        (None, None, None, None, None);
    for entry in metadata.kvs() {
        let kv = entry?;
        match kv.key {
            b"general.architecture" => {
                let value = kv.value_type == 8 && kv.encoded_value == b"\x05\0\0\0\0\0\0\0llama";
                assign(&mut architecture, value)?;
            }
            b"general.alignment" => assign(&mut alignment, integer(kv)?)?,
            b"llama.block_count" => assign(&mut layers, integer(kv)?)?,
            b"llama.embedding_length" => assign(&mut embedding, integer(kv)?)?,
            b"llama.attention.head_count" => assign(&mut heads, integer(kv)?)?,
            b"llama.attention.head_count_kv" => assign(&mut kv_heads, integer(kv)?)?,
            b"llama.feed_forward_length" => assign(&mut ffn, integer(kv)?)?,
            b"llama.attention.key_length" => assign(&mut head_dim, integer(kv)?)?,
            b"llama.attention.value_length" => assign(&mut value_dim, integer(kv)?)?,
            b"llama.rope.dimension_count" => assign(&mut rotary_dim, integer(kv)?)?,
            b"llama.context_length" => assign(&mut trained_context, integer(kv)?)?,
            b"llama.vocab_size" => assign(&mut declared_vocab, integer(kv)?)?,
            b"llama.rope.freq_base" => assign(&mut base, float(kv)?)?,
            b"llama.rope.scale_linear" | b"llama.rope.scaling.factor" => {
                assign(&mut scale, float(kv)?)?
            }
            b"llama.attention.layer_norm_rms_epsilon" => assign(&mut epsilon, float(kv)?)?,
            b"llama.rope.scaling.type"
                if kv.value_type != 8
                    || ![
                        b"\x04\0\0\0\0\0\0\0none".as_slice(),
                        b"\x06\0\0\0\0\0\0\0linear".as_slice(),
                    ]
                    .contains(&kv.encoded_value) =>
            {
                return Err(Error::UnsupportedProfile)
            }
            b"llama.rope.scaling.type" => {
                assign(&mut scaling_type, kv.encoded_value.ends_with(b"linear"))?
            }
            key if key.starts_with(b"llama.") => return Err(Error::UnsupportedProfile),
            _ => {}
        }
    }
    if architecture != Some(true) || epsilon.is_some_and(|v: f32| v.to_bits() != 1e-5f32.to_bits())
    {
        return Err(Error::UnsupportedProfile);
    }
    let embedding = embedding.ok_or(Error::MissingMetadata)?;
    let heads = heads.ok_or(Error::MissingMetadata)?;
    let layers = layers.ok_or(Error::MissingMetadata)?;
    let ffn = ffn.ok_or(Error::MissingMetadata)?;
    let kv_heads = kv_heads.unwrap_or(heads);
    if embedding == 0
        || embedding > MAX_INPUT_WIDTH
        || ffn == 0
        || ffn > MAX_INPUT_WIDTH
        || heads == 0
        || heads > embedding
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || !embedding.is_multiple_of(heads)
        || layers == 0
        || context == 0
        || context > u32::MAX as usize
    {
        return Err(Error::InvalidDimensions);
    }
    let head_dim = head_dim.unwrap_or(embedding / heads);
    if heads.checked_mul(head_dim) != Some(embedding) || !head_dim.is_multiple_of(2) {
        return Err(Error::InvalidDimensions);
    }
    if value_dim.is_some_and(|n| n != head_dim)
        || rotary_dim.is_some_and(|n| n != head_dim)
        || trained_context.is_some_and(|n| context > n)
        || (scaling_type == Some(false) && scale.is_some_and(|v| v != 1.0))
        || (scaling_type == Some(true) && scale.is_none())
    {
        return Err(Error::UnsupportedProfile);
    }
    let data_start = usize::try_from(metadata.tensor_data_start(alignment.unwrap_or(32) as u64)?)
        .map_err(|_| Error::ArithmeticOverflow)?;
    if data_start > bytes.len() {
        return Err(Error::Payload);
    }
    let mut vocabulary = None;
    for tensor in metadata.tensors() {
        let tensor = tensor?;
        if tensor.name == b"token_embd.weight" {
            if tensor.rank != 2 || tensor.dimension(0) != Some(embedding as u64) {
                return Err(Error::InvalidDimensions);
            }
            let value = usize::try_from(tensor.dimension(1).ok_or(Error::InvalidDimensions)?)
                .map_err(|_| Error::ArithmeticOverflow)?;
            if value == 0 || value > u32::MAX as usize {
                return Err(Error::InvalidDimensions);
            }
            if vocabulary.replace(value).is_some() {
                return Err(Error::DuplicateTensor);
            }
        }
    }
    let vocabulary = vocabulary.ok_or(Error::MissingTensor)?;
    if declared_vocab.is_some_and(|n| n != vocabulary) {
        return Err(Error::InvalidDimensions);
    }
    Ok(DenseGgufConfig {
        bytes,
        metadata,
        data_start,
        layers,
        config: DenseConfig {
            embedding,
            ffn,
            heads,
            kv_heads,
            head_dim,
            max_context: context,
            vocabulary,
            rope_base: base.unwrap_or(100_000.0),
            rope_scale: scale.unwrap_or(1.0),
        },
    })
}
