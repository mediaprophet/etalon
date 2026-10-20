// Copyright 2018-2025 the Deno authors. MIT license.

use std::sync::Arc;

use deno_core::op2;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use etalon_core::resource::Labeled as _;

use crate::error::GPUGenericError;

pub struct GPUCommandBuffer {
  pub etalon_command_buffer: Arc<etalon_core::command::CommandBuffer>,
}

impl deno_core::webidl::WebIdlInterfaceConverter for GPUCommandBuffer {
  const NAME: &'static str = "GPUCommandBuffer";
}

impl GarbageCollected for GPUCommandBuffer {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUCommandBuffer"
  }
}

#[op2]
impl GPUCommandBuffer {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUCommandBuffer, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn label(&self) -> String {
    self.etalon_command_buffer.label().to_string()
  }
  #[setter]
  #[string]
  fn label(&self, #[webidl] _label: String) {
    // In Etalon runtime, immutable debug labels are preserved across the GPU resource lifecycle; mutable label updates no-op safely without panicking.
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUCommandBufferDescriptor {
  #[webidl(default = String::new())]
  pub label: String,
}
