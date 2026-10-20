// Copyright 2018-2025 the Deno authors. MIT license.

use deno_core::cppgc::Ptr;
use deno_core::op2;
use deno_core::webidl::Nullable;
use deno_core::webidl::WebIdlInterfaceConverter;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use std::sync::Arc;
use etalon_core::resource::Labeled as _;

use crate::error::GPUGenericError;

pub struct GPUPipelineLayout {
  pub etalon_pipeline_layout: Arc<etalon_core::binding_model::PipelineLayout>,
}

impl WebIdlInterfaceConverter for GPUPipelineLayout {
  const NAME: &'static str = "GPUPipelineLayout";
}

impl GarbageCollected for GPUPipelineLayout {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUPipelineLayout"
  }
}

#[op2]
impl GPUPipelineLayout {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUPipelineLayout, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn label(&self) -> String {
    self.etalon_pipeline_layout.label().to_string()
  }
  #[setter]
  #[string]
  fn label(&self, #[webidl] _label: String) {
    // In Etalon runtime, immutable debug labels are preserved across the GPU resource lifecycle; mutable label updates no-op safely without panicking.
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUPipelineLayoutDescriptor {
  #[webidl(default = String::new())]
  pub label: String,

  pub bind_group_layouts:
    Vec<Nullable<Ptr<super::bind_group_layout::GPUBindGroupLayout>>>,
  #[webidl(default = 0)]
  pub immediate_size: u32,
}
