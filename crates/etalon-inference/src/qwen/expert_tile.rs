//! Historical Qwen aliases for generic caller-validated expert tiles.
pub use crate::moe::expert_tile::{
    ExpertPlane as QwenExpertPlane, ExpertTileAdmission as QwenExpertTileAdmission,
    ExpertTileDescriptor as QwenExpertTileDescriptor, ExpertTileError as QwenExpertTileError,
    ExpertTilePlaneKind as QwenExpertTilePlaneKind, ExpertTileReader as QwenExpertTileReader,
    QWEN_EXPERT_TILE_FORMAT, QWEN_EXPERT_TILE_SUFFIX,
};
