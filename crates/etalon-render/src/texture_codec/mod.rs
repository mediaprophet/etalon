//! Zero-heap texture codec, KTX2 container inspection, CPU mip reduction, and streaming planners.

pub mod cpu_mips;
pub mod ktx2;
pub mod ktx2_rgba8;
pub mod stream_policy;
pub mod streaming_plan;
pub mod transcode;

pub use cpu_mips::{
    downsample_alpha_mask_level_into, downsample_rgba8_level_into, CpuTextureMipError,
    CpuTextureMipSemantic,
};
pub use ktx2::{
    select_hardware_compressed_format, HardwareCompressedUploadFormat, Ktx2ColorSpace,
    Ktx2Document, Ktx2Error, Ktx2Level, Ktx2Supercompression, Ktx2TranscodeSource,
    KTX2_IDENTIFIER,
};
pub use ktx2_rgba8::{
    decode_ktx2_rgba8_base_level, decode_ktx2_rgba8_base_level_with_scratch,
    inspect_ktx2_rgba8_base_level, inspect_ktx2_rgba8_base_level_with_compression,
    Rgba8Ktx2Error, Rgba8Ktx2Format, Rgba8Ktx2Image, VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_R8G8B8A8_UNORM,
};
pub use stream_policy::{
    plan_texture_residency, plan_texture_residency_lifecycle, plan_texture_residency_partial,
    TextureStreamAction, TextureStreamActionKind, TextureStreamBudget, TextureStreamDemand,
    TextureStreamLifecycleAction, TextureStreamLifecycleActionKind, TextureStreamLifecyclePlan,
    TextureStreamPartialPlan, TextureStreamPlanError,
};
pub use streaming_plan::{
    decode_rgba8_ktx2_mip_into, reserve_texture_mip, rgba8_ktx2_mip_candidate, select_texture_mip,
    DecodedKtx2Mip, Ktx2MipCandidateError, Ktx2MipDecodeError, TextureMipBackendCost,
    TextureMipBudget, TextureMipCandidate, TextureMipSelection,
};
pub use transcode::{
    decode_astc_4x4_block, decode_bc1_block, decode_bc3_block, decode_bc7_block,
    decode_compressed_image_to_rgba8, decode_etc2_eac_rgba_block, transcode_ktx2_into,
    CompressedBlockFormat, TranscodeError, TranscodeTargetFormat, TranscodedTextureInfo,
};
