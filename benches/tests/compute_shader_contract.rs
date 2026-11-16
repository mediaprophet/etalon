fn validate_single_writer(source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| e.to_string())?;
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == "cs_main")
        .ok_or("missing compute entry point")?;
    if entry.workgroup_size != [1, 1, 1] {
        return Err(
            "benchmark writes fixed storage locations and must dispatch one invocation".into(),
        );
    }
    Ok(())
}

#[test]
fn fixed_storage_shader_has_one_writer() {
    validate_single_writer(include_str!("../benches/etalon-benchmark/computepass.wgsl")).unwrap();
}

#[test]
fn bindless_shader_has_one_writer() {
    validate_single_writer(include_str!(
        "../benches/etalon-benchmark/computepass-bindless.wgsl"
    ))
    .unwrap();
}

#[test]
fn previous_multi_writer_workload_is_rejected() {
    for source in [
        include_str!("../benches/etalon-benchmark/computepass.wgsl"),
        include_str!("../benches/etalon-benchmark/computepass-bindless.wgsl"),
    ] {
        let previous = source.replace("@workgroup_size(1)", "@workgroup_size(16)");
        assert!(validate_single_writer(&previous)
            .unwrap_err()
            .contains("one invocation"));
    }
}
