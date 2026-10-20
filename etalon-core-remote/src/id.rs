pub use etalon_core_remote_types::id::*;

/// Reference etalon objects via numeric IDs assigned by [`etalon_core_remote_types::identity::IdentityManager`].
#[derive(Clone, Debug)]
pub struct IdReferences;

impl etalon_core::command::ReferenceType for IdReferences {
    type Buffer = BufferId;
    type Surface = SurfaceId;
    type Texture = TextureId;
    type TextureView = TextureViewId;
    type ExternalTexture = ExternalTextureId;
    type QuerySet = QuerySetId;
    type BindGroup = BindGroupId;
    type RenderPipeline = RenderPipelineId;
    type RenderBundle = RenderBundleId;
    type ComputePipeline = ComputePipelineId;
    type Blas = BlasId;
    type Tlas = TlasId;
}
