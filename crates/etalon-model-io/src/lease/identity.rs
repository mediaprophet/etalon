use super::{LeaseError, ModelLease};
use sha2::{Digest, Sha256};

/// Sources remain distinct. LoRA identity includes its full header as well as
/// payload; a helper/config/tokenizer is never silently identified by base bytes.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelPartRole {
    BaseModel = 1,
    Shard = 2,
    Tokenizer = 3,
    Configuration = 4,
    HelperMetadata = 5,
    LoraAdapter = 6,
    DraftModel = 7,
}
#[derive(Debug)]
pub struct ModelPart<'a> {
    pub role: ModelPartRole,
    /// Explicit source format/interpretation version, validated by future loader.
    pub format_version: u32,
    pub ordinal: u32,
    pub lease: &'a ModelLease,
}
#[derive(Clone, Copy, Debug)]
pub struct BundleLimits {
    pub max_parts: usize,
    pub max_source_bytes: u64,
}
/// Binds the complete ordered declared resource set, not completeness of that
/// set versus a loader's actual dependencies. The future prepared loader must
/// prohibit omitted/reopened files and bind parsed/transformed state separately.
pub fn bundle_digest(
    parts: &[ModelPart<'_>],
    limits: BundleLimits,
) -> Result<[u8; 32], LeaseError> {
    if parts.is_empty()
        || parts.len() > limits.max_parts
        || parts.len() > 256
        || parts[0].role != ModelPartRole::BaseModel
        || parts[0].ordinal != 0
    {
        return Err(LeaseError::InvalidBundle);
    }
    let mut total = 0u64;
    let mut previous = None;
    // Preflight all counts/order/budgets before hashing metadata.
    for part in parts {
        let order = (part.role, part.ordinal);
        if part.format_version == 0
            || previous.is_some_and(|p| p >= order)
            || (previous.is_some() && part.role == ModelPartRole::BaseModel)
        {
            return Err(LeaseError::InvalidBundle);
        }
        total = total
            .checked_add(part.lease.identity().byte_len)
            .ok_or(LeaseError::Budget)?;
        if total > limits.max_source_bytes {
            return Err(LeaseError::Budget);
        }
        previous = Some(order);
    }
    let mut h = Sha256::new();
    h.update(b"qualia:g3:immutable-model-bundle:v1\0");
    h.update(1u32.to_le_bytes());
    h.update((parts.len() as u32).to_le_bytes());
    for part in parts {
        h.update([part.role as u8]);
        h.update(part.format_version.to_le_bytes());
        h.update(part.ordinal.to_le_bytes());
        let identity = part.lease.identity();
        h.update(identity.byte_len.to_le_bytes());
        h.update(identity.content_sha256);
    }
    Ok(h.finalize().into())
}
