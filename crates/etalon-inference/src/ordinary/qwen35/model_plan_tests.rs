use super::*;
fn tensor(dims: &[usize], ggml_type: u32) -> TensorInfo {
    let mut shape = [0; 4];
    for (i, d) in dims.iter().enumerate() {
        shape[i] = *d as u64;
    }
    TensorInfo {
        dims: shape,
        n_dims: dims.len() as u32,
        ggml_type,
        byte_offset: 0,
    }
}
fn complete_ffn() -> LayerTensors {
    LayerTensors {
        moe_router: Some(tensor(&[4, 2], 0)),
        moe_gate_exps: Some(tensor(&[4, 3, 2], 0)),
        moe_up_exps: Some(tensor(&[4, 3, 2], 0)),
        moe_down_exps: Some(tensor(&[3, 4, 2], 0)),
        moe_shared_gate: Some(tensor(&[4, 2], 0)),
        moe_shared_up: Some(tensor(&[4, 2], 0)),
        moe_shared_down: Some(tensor(&[2, 4], 0)),
        moe_shared_gate_input: Some(tensor(&[4], 0)),
        ..Default::default()
    }
}
#[test]
fn shared_expert_and_explicit_routing_are_mandatory() {
    let h = ModelHyperparams {
        n_embd: 4,
        ..Default::default()
    };
    let routing = etalon_model::inference::moe_routing::MoeRoutingSettings {
        expert_count: Some(2),
        experts_per_token: Some(1),
    };
    let mut t = complete_ffn();
    assert!(ffn(&t, h, routing).is_ok());
    assert_eq!(
        ffn(&t, h, Default::default()).unwrap_err(),
        PlanError::MissingRouting
    );
    t.moe_shared_gate_input = None;
    assert_eq!(ffn(&t, h, routing).unwrap_err(), PlanError::MissingTensor);
}
#[test]
fn ffn_fp8_fails_precisely_and_sizes_fail_closed() {
    let h = ModelHyperparams {
        n_embd: 4,
        ..Default::default()
    };
    let routing = etalon_model::inference::moe_routing::MoeRoutingSettings {
        expert_count: Some(2),
        experts_per_token: Some(1),
    };
    let mut t = complete_ffn();
    t.moe_gate_exps.as_mut().unwrap().ggml_type = etalon_model::ggml::GGML_TYPE_FP8_E4M3;
    assert_eq!(
        ffn(&t, h, routing).unwrap_err(),
        PlanError::FfnRowScaleUnsupported
    );
    assert_eq!(mul(usize::MAX, 2), Err(PlanError::Overflow));
}

#[test]
fn scalar_norm_endpoint_overflow_is_cold_error() {
    let mut info = tensor(&[4], 0);
    info.byte_offset = u64::MAX - 8;
    assert_eq!(shape(info, &[4]), Err(PlanError::Overflow));
}
#[test]
fn expert_endpoint_overflow_is_cold_error() {
    let h = ModelHyperparams {
        n_embd: 4,
        ..Default::default()
    };
    let routing = etalon_model::inference::moe_routing::MoeRoutingSettings {
        expert_count: Some(2),
        experts_per_token: Some(1),
    };
    let mut t = complete_ffn();
    t.moe_gate_exps.as_mut().unwrap().byte_offset = u64::MAX;
    assert_eq!(ffn(&t, h, routing).unwrap_err(), PlanError::Overflow);
}
