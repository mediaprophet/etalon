use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate_portable_shader(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    // Core pack/unpack2x16float operates on f32 values; it does not require shader-f16.
    Validator::new(
        ValidationFlags::all(),
        Capabilities::SHADER_FLOAT16_IN_FLOAT32,
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{error:?}"));
}

#[test]
fn actual_topk_shader_parses_and_validates_without_optional_features() {
    validate_portable_shader(etalon_tensor::TOPK_REDUCTION_WGSL);
}

#[test]
fn actual_attention_shader_parses_and_validates_without_optional_features() {
    validate_portable_shader(etalon_tensor::FUSED_ATTENTION_WGSL);
}
