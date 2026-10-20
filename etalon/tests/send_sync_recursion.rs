//! Tests that evaluating `etalon::SomeType: Send + Sync` is cheap.
//!
//! This test must be located within the `etalon` package so that it can use `cfg(send_sync)`.
//! It is a separate test target so that, if it fails to compile, it doesn’t block other builds.
//!
//! The implementations that supply the property this test is testing for are located in
//! `../src/dispatch.rs`, in the macro `explicit_send_sync_impl`.

#![cfg(send_sync)]
#![allow(clippy::type_complexity, clippy::explicit_auto_deref)]

/// 50 `Box`es that have to be looked through to evaluate auto traits.
/// 
/// This count is chosen so that if `etalon` uses a significant amount of the default
/// recursion limit, this code will fail to compile.
#[rustfmt::skip]
struct Tower<T: ?Sized>(
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
        T
    >>>>>>>>>>
    >>>>>>>>>>
    >>>>>>>>>>
    >>>>>>>>>>
    >>>>>>>>>>
);

impl<T: ?Sized> core::ops::Deref for Tower<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // We could let auto-deref handle this, but where’s the fun in that --
        // er, I mean, rust-analyzer wouldn’t type check it properly.
        &**************************************************self.0
    }
}

const fn assert_send_sync_is_cheap<T>()
where
    Tower<T>: Send,
    Tower<T>: Sync,
{
}

const _: () = {
    assert_send_sync_is_cheap::<etalon::Adapter>();
    assert_send_sync_is_cheap::<etalon::BindGroup>();
    assert_send_sync_is_cheap::<etalon::BindGroupLayout>();
    assert_send_sync_is_cheap::<etalon::Blas>();
    assert_send_sync_is_cheap::<etalon::Buffer>();
    assert_send_sync_is_cheap::<etalon::BufferView>();
    assert_send_sync_is_cheap::<etalon::BufferViewMut>();
    assert_send_sync_is_cheap::<etalon::CommandBuffer>();
    assert_send_sync_is_cheap::<etalon::CommandEncoder>();
    assert_send_sync_is_cheap::<etalon::CompilationInfo>();
    assert_send_sync_is_cheap::<etalon::ComputePass>();
    assert_send_sync_is_cheap::<etalon::ComputePipeline>();
    assert_send_sync_is_cheap::<etalon::Device>();
    assert_send_sync_is_cheap::<etalon::Error>();
    assert_send_sync_is_cheap::<etalon::ExternalTexture>();
    assert_send_sync_is_cheap::<etalon::Features>();
    assert_send_sync_is_cheap::<etalon::Instance>();
    assert_send_sync_is_cheap::<etalon::Limits>();
    assert_send_sync_is_cheap::<etalon::PipelineCache>();
    assert_send_sync_is_cheap::<etalon::PipelineLayout>();
    assert_send_sync_is_cheap::<etalon::QuerySet>();
    assert_send_sync_is_cheap::<etalon::Queue>();
    assert_send_sync_is_cheap::<etalon::QueueWriteBufferView>();
    assert_send_sync_is_cheap::<etalon::RenderBundle>();
    // RenderBundleEncoder is never Send or Sync
    assert_send_sync_is_cheap::<etalon::RenderPass>();
    assert_send_sync_is_cheap::<etalon::RenderPipeline>();
    assert_send_sync_is_cheap::<etalon::Sampler>();
    assert_send_sync_is_cheap::<etalon::ShaderModule>();
    assert_send_sync_is_cheap::<etalon::Surface>();
    assert_send_sync_is_cheap::<etalon::SurfaceTexture>();
    assert_send_sync_is_cheap::<etalon::Texture>();
    assert_send_sync_is_cheap::<etalon::TextureView>();
    assert_send_sync_is_cheap::<etalon::Tlas>();
};
