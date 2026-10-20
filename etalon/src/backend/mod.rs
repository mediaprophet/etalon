#[cfg(webgpu)]
pub mod webgpu;
#[cfg(webgpu)]
pub(crate) use webgpu::{get_browser_gpu_property, ContextWebGpu};

#[cfg(etalon_core)]
pub mod etalon_core;

#[cfg(etalon_core)]
pub(crate) use etalon_core::ContextetalonCore;

#[cfg(custom)]
pub mod custom;
