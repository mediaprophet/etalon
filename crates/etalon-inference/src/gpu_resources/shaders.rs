//! Canonical inference kernel assets.
//! Large WGSL assets are cohesive static numerical kernels, kept whole for shader auditability.
pub const COOP_GEMV_SUBGROUP_WGSL: &str = include_str!("shaders/coop_gemv_subgroup.wgsl");
pub const DEQUANT_TEMPLATE_WGSL: &str = include_str!("shaders/dequant_template.wgsl");
pub const DUAL_GEMV_WGSL: &str = include_str!("shaders/dual_gemv.wgsl");
pub const FUSED_ATTENTION_WGSL: &str = include_str!("shaders/fused_attention.wgsl");
pub const FUSED_FFN_WGSL: &str = include_str!("shaders/fused_ffn.wgsl");
pub const FUSED_TENSOR_CONTRACTION_WGSL: &str =
    include_str!("shaders/fused_tensor_contraction.wgsl");
pub const FUSED_TRANSFORMER_WGSL: &str = include_str!("shaders/fused_transformer.wgsl");
pub const QUANTIZED_EMBEDDING_WGSL: &str = include_str!("shaders/quantized_embedding.wgsl");
pub const TRIPLE_GEMV_WGSL: &str = include_str!("shaders/triple_gemv.wgsl");
pub const WASM_DEQUANT_TEMPLATE_WGSL: &str = include_str!("shaders/wasm/dequant_template.wgsl");
pub const WASM_FUSED_FFN_WGSL: &str = include_str!("shaders/wasm/fused_ffn.wgsl");
pub const WASM_FUSED_TRANSFORMER_WGSL: &str = include_str!("shaders/wasm/fused_transformer.wgsl");
pub const WASM_ELEMENTWISE_WGSL: &str = include_str!("shaders/wasm_elementwise.wgsl");
