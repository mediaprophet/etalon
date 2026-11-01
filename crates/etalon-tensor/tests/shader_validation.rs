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

#[test]
fn packed_attention_shader_validates_with_exactly_four_storage_bindings() {
    let source = etalon_tensor::packed_fused_attention_wgsl();
    validate_portable_shader(&source);
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let storage_count = module
        .global_variables
        .iter()
        .filter(|(_, variable)| matches!(variable.space, naga::AddressSpace::Storage { .. }))
        .count();
    assert_eq!(storage_count, 4);
    assert!(!module.global_variables.iter().any(|(_, variable)| variable
        .binding
        .as_ref()
        .is_some_and(|binding| binding.binding == 5)));
}
