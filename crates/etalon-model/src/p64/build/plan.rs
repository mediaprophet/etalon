use super::layout::{LayoutInput, LayoutProfile, PlanInput, TensorLayout};
use super::{BuildBudget, BuildError, BuildInput, BuildProfile};
use crate::p64::layout::*;

#[derive(Debug)]
pub struct BuildPlan<'a, 's> {
    pub(super) input: BuildInput<'a>,
    pub(super) entries: &'s [P64TensorEntry],
    pub(super) header: P64WeightHeader,
    pub(super) output_len: usize,
}

impl BuildPlan<'_, '_> {
    pub fn header(&self) -> &P64WeightHeader {
        &self.header
    }
    pub fn entries(&self) -> &[P64TensorEntry] {
        self.entries
    }
    pub fn output_len(&self) -> usize {
        self.output_len
    }
}

fn add(left: usize, right: usize) -> Result<usize, BuildError> {
    left.checked_add(right)
        .ok_or(BuildError::ArithmeticOverflow)
}

fn mul(left: usize, right: usize) -> Result<usize, BuildError> {
    left.checked_mul(right)
        .ok_or(BuildError::ArithmeticOverflow)
}

fn align(value: usize, alignment: usize) -> Result<usize, BuildError> {
    Ok(add(value, alignment - 1)? & !(alignment - 1))
}

fn offset(value: usize) -> Result<u32, BuildError> {
    u32::try_from(value).map_err(|_| BuildError::OffsetLimit)
}

pub(super) fn layer_count(input: &BuildInput<'_>) -> u32 {
    u32::from_le_bytes(input.hparams_bytes[..4].try_into().unwrap())
}

/// Preflight all constraints before touching scratch. The returned plan exclusively
/// borrows the used scratch entries so validated offsets cannot subsequently change.
pub fn plan_into<'a, 's>(
    input: BuildInput<'a>,
    scratch: &'s mut [P64TensorEntry],
    budget: BuildBudget,
) -> Result<BuildPlan<'a, 's>, BuildError> {
    let planned = plan_shared(input.into(), scratch, budget)?;
    Ok(BuildPlan {
        input,
        entries: planned.entries,
        header: planned.header,
        output_len: planned.output_len,
    })
}

#[derive(Debug)]
pub struct LayoutPlan<'a, 's> {
    pub(super) input: PlanInput<'a>,
    pub(super) entries: &'s [P64TensorEntry],
    pub(super) header: P64WeightHeader,
    pub(super) output_len: usize,
}
impl LayoutPlan<'_, '_> {
    pub fn header(&self) -> &P64WeightHeader {
        &self.header
    }
    pub fn entries(&self) -> &[P64TensorEntry] {
        self.entries
    }
    pub fn output_len(&self) -> usize {
        self.output_len
    }
}

pub fn plan_layout_into<'a, 's>(
    input: LayoutInput<'a>,
    scratch: &'s mut [P64TensorEntry],
    budget: BuildBudget,
) -> Result<LayoutPlan<'a, 's>, BuildError> {
    plan_shared(input.into(), scratch, budget)
}

fn plan_shared<'a, 's>(
    input: PlanInput<'a>,
    scratch: &'s mut [P64TensorEntry],
    budget: BuildBudget,
) -> Result<LayoutPlan<'a, 's>, BuildError> {
    if input.count() == 0 {
        return Err(BuildError::EmptyTensors);
    }
    let count = input.count();
    let scratch_bytes = mul(count, P64_TENSOR_ENTRY_BYTES)?;
    if count > scratch.len() {
        return Err(BuildError::ScratchCapacity {
            required_entries: count,
            available_entries: scratch.len(),
        });
    }
    if scratch_bytes > budget.scratch_bytes {
        return Err(BuildError::ScratchBudget {
            required_bytes: scratch_bytes,
            available_bytes: budget.scratch_bytes,
        });
    }
    let page_log2 = if input.page_log2 == 0 {
        P64_DEFAULT_PAGE_LOG2
    } else {
        input.page_log2
    };
    if !(8..=30).contains(&page_log2) {
        return Err(BuildError::InvalidPageSize);
    }
    let page = 1usize << page_log2;
    let structural = P64_FLAG_LITTLE_ENDIAN
        | P64_FLAG_RAW_TRANSCODE
        | P64_FLAG_LAYER_MAJOR
        | P64_FLAG_LAYER_PACK
        | P64_FLAG_LAYER_SCHEDULE;
    let expected_flags = match input.profile {
        LayoutProfile::FfnLegacy | LayoutProfile::Standard(BuildProfile::GgufLayerPacked) => {
            P64_FLAG_LITTLE_ENDIAN
                | P64_FLAG_LAYER_MAJOR
                | P64_FLAG_LAYER_PACK
                | P64_FLAG_LAYER_SCHEDULE
        }
        LayoutProfile::Standard(BuildProfile::Safetensor) => {
            P64_FLAG_LITTLE_ENDIAN | P64_FLAG_RAW_TRANSCODE
        }
        LayoutProfile::Standard(BuildProfile::Vision) => P64_FLAG_LITTLE_ENDIAN,
    };
    if input.flags & structural != expected_flags {
        return Err(BuildError::ProfileFlags);
    }
    let layers = input.layers();
    if !input.has_schedule() && !input.tokenizer_bytes.is_empty()
        || input.profile == LayoutProfile::Standard(BuildProfile::Vision) && layers != 0
    {
        return Err(BuildError::ProfileSections);
    }
    let manifolds = layers
        .checked_add(1)
        .ok_or(BuildError::ArithmeticOverflow)? as usize;
    if input.manifold_bytes.len() != mul(manifolds, P64_MANIFOLD_ENTRY_BYTES)? {
        return Err(BuildError::ManifoldCount);
    }
    let schedule_bytes = if input.has_schedule() {
        mul(layers as usize, 64)?
    } else {
        0
    };
    if (input.profile == LayoutProfile::FfnLegacy && input.schedule_bytes.len() != schedule_bytes)
        || (input.profile != LayoutProfile::FfnLegacy && !input.schedule_bytes.is_empty())
    {
        return Err(BuildError::ProfileSections);
    }
    let tensor_table = add(128, schedule_bytes)?;
    let strings = add(tensor_table, scratch_bytes)?;
    let mut string_bytes =
        usize::from(input.profile != LayoutProfile::Standard(BuildProfile::Vision));
    for index in 0..input.count() {
        let tensor = input.tensor(index);
        if !(1..=4).contains(&tensor.rank) {
            return Err(BuildError::InvalidRank(index));
        }
        if tensor.manifold_idx > layers {
            return Err(BuildError::InvalidManifold(index));
        }
        offset(usize::try_from(tensor.blob_len).map_err(|_| BuildError::OffsetLimit)?)?;
        string_bytes = add(string_bytes, add(tensor.name.len(), 1)?)?;
    }
    let manifold_table = align(add(strings, string_bytes)?, 64)?;
    let tokenizer = add(manifold_table, input.manifold_bytes.len())?;
    let checksum = align(add(tokenizer, input.tokenizer_bytes.len())?, 64)?;
    let checksum_end = add(checksum, mul(add(count, 1)?, 4)?)?;
    let blob_start = align(checksum_end, page)?;
    let output_len = walk_entries(&input, blob_start, page, None)?;
    offset(output_len)?;
    if output_len > budget.output_bytes {
        return Err(BuildError::OutputBudget {
            required_bytes: output_len,
            available_bytes: budget.output_bytes,
        });
    }
    let header = P64WeightHeader {
        magic: P64_MAGIC,
        version: P64_VERSION,
        flags: input.flags,
        role_table_offset: match input.profile {
            LayoutProfile::FfnLegacy | LayoutProfile::Standard(BuildProfile::GgufLayerPacked) => {
                128
            }
            LayoutProfile::Standard(BuildProfile::Safetensor) => 0,
            LayoutProfile::Standard(BuildProfile::Vision) => offset(tensor_table)?,
        },
        tensor_table_offset: offset(tensor_table)?,
        tokenizer_offset: offset(tokenizer)?,
        hparams_offset: 64,
        string_table_offset: offset(strings)?,
        checksum_offset: offset(checksum)?,
        manifold_table_offset: offset(manifold_table)?,
        tensor_count: offset(count)?,
        page_size: offset(page)?,
        reserved: input.header_reserved,
    };
    // Identical immutable inputs have already passed this deterministic walk.
    walk_entries(&input, blob_start, page, Some(&mut scratch[..count]))?;
    Ok(LayoutPlan {
        input,
        entries: &scratch[..count],
        header,
        output_len,
    })
}

fn walk_entries(
    input: &PlanInput<'_>,
    mut cursor: usize,
    page: usize,
    mut output: Option<&mut [P64TensorEntry]>,
) -> Result<usize, BuildError> {
    let mut name_offset =
        usize::from(input.profile != LayoutProfile::Standard(BuildProfile::Vision));
    let mut previous_manifold = None;
    for index in 0..input.count() {
        let tensor = input.tensor(index);
        let alignment = if input.profile == LayoutProfile::Standard(BuildProfile::GgufLayerPacked)
            && previous_manifold == Some(tensor.manifold_idx)
        {
            256
        } else {
            page
        };
        cursor = align(cursor, alignment)?;
        let entry = entry(&tensor, offset(name_offset)?, offset(cursor)?)?;
        if let Some(entries) = output.as_deref_mut() {
            entries[index] = entry;
        }
        cursor = add(
            cursor,
            usize::try_from(tensor.blob_len).map_err(|_| BuildError::OffsetLimit)?,
        )?;
        name_offset = add(name_offset, add(tensor.name.len(), 1)?)?;
        previous_manifold = Some(tensor.manifold_idx);
    }
    match input.profile {
        LayoutProfile::FfnLegacy | LayoutProfile::Standard(BuildProfile::GgufLayerPacked) => {
            align(cursor, 64)
        }
        LayoutProfile::Standard(BuildProfile::Safetensor) => Ok(cursor),
        LayoutProfile::Standard(BuildProfile::Vision) => align(cursor, page),
    }
}

fn entry(
    tensor: &TensorLayout<'_>,
    name_offset: u32,
    blob_offset: u32,
) -> Result<P64TensorEntry, BuildError> {
    Ok(P64TensorEntry {
        name_offset,
        role_id: tensor.role_id,
        dtype: tensor.dtype,
        manifold_idx: tensor.manifold_idx,
        rank: tensor.rank,
        dimensions: tensor.dimensions,
        blob_offset,
        blob_size: offset(usize::try_from(tensor.blob_len).map_err(|_| BuildError::OffsetLimit)?)?,
        source_offset: tensor.source_offset,
        source_name_hash: tensor.source_name_hash,
        alt_dtype: tensor.alt_dtype,
        precision_views_mask: tensor.precision_views_mask,
        alt_blob_offset: tensor.alt_blob_offset,
    })
}
