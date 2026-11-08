mod provider;
use crate::projections::{StreamingArgmaxResult, VOCAB_CHUNK_ROWS};
pub use provider::{BrowserProjectionProvider, BrowserProjectionResources};
const MAX_STACK_GEMM_IN: usize = 10240;
pub struct BrowserProjectionEngine<'a, P> {
    pub host: &'a P,
}
fn dynamic_uniform_binding(buffer: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: std::num::NonZeroU64::new(256),
    })
}
mod dispatch_gemm_into_async;
mod dispatch_gemm_raw_into_async;
mod dispatch_gemm_resident_chunk_async;
mod output;

mod batched_argmax;
mod fused_block;

mod top1;
pub use top1::{read_browser_top1, BrowserTop1Plan, BrowserTop1Resources};
