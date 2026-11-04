use super::*;
use core::cell::Cell;
use etalon_model::inference::moe_routing::MoeRoutingSettings;

fn tensor(dims: &[u64], offset: u64) -> TensorInfo {
    let mut shape = [0; 4];
    shape[..dims.len()].copy_from_slice(dims);
    TensorInfo {
        n_dims: dims.len() as u32,
        dims: shape,
        ggml_type: 0,
        byte_offset: offset,
    }
}
fn plan() -> ProjectionPlan {
    let t = LayerTensors {
        moe_router: Some(tensor(&[1, 2], 0)),
        moe_gate_exps: Some(tensor(&[1, 1, 2], 8)),
        moe_up_exps: Some(tensor(&[1, 1, 2], 16)),
        moe_down_exps: Some(tensor(&[1, 1, 2], 24)),
        moe_shared_gate: Some(tensor(&[1, 1], 32)),
        moe_shared_up: Some(tensor(&[1, 1], 36)),
        moe_shared_down: Some(tensor(&[1, 1], 40)),
        moe_shared_gate_input: Some(tensor(&[1], 44)),
        ..Default::default()
    };
    let routing = MoeRoutingSettings {
        expert_count: Some(2),
        experts_per_token: Some(2),
    }
    .admit(2)
    .unwrap();
    ProjectionPlan::checked(1, &t, routing).unwrap()
}
struct Host {
    weights: [f32; 12],
    fail: Option<u64>,
    calls: Cell<usize>,
}
impl Host {
    fn new() -> Self {
        Self {
            weights: [0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0.],
            fail: None,
            calls: Cell::new(0),
        }
    }
}
impl ProjectionHost for Host {
    type Error = u8;
    fn admits(&self, _: &Projection) -> bool {
        true
    }
    fn gemm(&self, p: &Projection, x: &[f32], y: &mut [f32]) -> Result<(), u8> {
        self.calls.set(self.calls.get() + 1);
        y.fill(-99.);
        if self.fail == Some(p.weight.byte_offset) {
            return Err(7);
        }
        let start = (p.weight.byte_offset / 4) as usize;
        for (row, dst) in y.iter_mut().enumerate() {
            *dst = x
                .iter()
                .enumerate()
                .map(|(col, x)| self.weights[start + row * x_len(p) + col] * x)
                .sum();
        }
        Ok(())
    }
}
fn x_len(p: &Projection) -> usize {
    p.weight.dims[0] as usize
}
#[test]
fn routed_and_sigmoid_gated_shared_arithmetic() {
    let p = plan();
    let mut host = Host::new();
    host.weights[1] = 3.0f32.ln();
    let mut out = [123.];
    let mut scratch = [0.; 7];
    assert_eq!(p.scratch_len(), 7);
    assert_eq!(p.workspace_bytes(), 28);
    execute(&host, &p, &[1.], &mut out, &mut scratch).unwrap();
    let sigmoid = |x: f32| 1. / (1. + (-x).exp());
    let expected = 0.25 * (1. * sigmoid(1.) * 3. * 5.)
        + 0.75 * (2. * sigmoid(2.) * 4. * 6.)
        + 0.5 * (7. * sigmoid(7.) * 8. * 9.);
    assert!((out[0] - expected).abs() < 0.0001);
}
#[test]
fn short_workspace_and_shape_never_call_provider() {
    let p = plan();
    let host = Host::new();
    let mut out = [123.];
    assert_eq!(
        execute(&host, &p, &[1.], &mut out, &mut [0.; 6]),
        Err(Error::Workspace)
    );
    assert_eq!(
        execute(&host, &p, &[], &mut out, &mut [0.; 7]),
        Err(Error::Shape)
    );
    assert_eq!(host.calls.get(), 0);
    assert_eq!(out, [123.]);
}
#[test]
fn final_provider_failure_keeps_output_unchanged() {
    let p = plan();
    let mut host = Host::new();
    host.fail = Some(44);
    let mut out = [123.];
    assert_eq!(
        execute(&host, &p, &[1.], &mut out, &mut [0.; 7]),
        Err(Error::Provider(7))
    );
    assert_eq!(out, [123.]);
}
#[test]
fn nonfinite_provider_and_intermediate_fail_closed() {
    let p = plan();
    let mut host = Host::new();
    let mut out = [123.];
    host.weights[0] = f32::NAN;
    assert_eq!(
        execute(&host, &p, &[1.], &mut out, &mut [0.; 7]),
        Err(Error::NonFinite)
    );
    assert_eq!(out, [123.]);
    host.weights[0] = 0.;
    host.weights[2] = f32::MAX;
    assert_eq!(
        execute(&host, &p, &[1.], &mut out, &mut [0.; 7]),
        Err(Error::NonFinite)
    );
    assert_eq!(out, [123.]);
}
#[test]
fn admission_checks_expert_shape_encoding_and_offset() {
    let p = plan();
    let check =
        |experts| ProjectionPlan::checked_projections(1, experts, p.router, p.shared, p.routing);
    let mut experts = p.experts;
    experts[0].dims[2] = 1;
    assert_eq!(check(experts).unwrap_err(), PlanError::Shape);
    experts = p.experts;
    experts[0].byte_offset = u64::MAX;
    assert_eq!(check(experts).unwrap_err(), PlanError::Overflow);
    experts = p.experts;
    experts[0].ggml_type = etalon_model::ggml::GGML_TYPE_FP8_E4M3;
    assert_eq!(
        check(experts).unwrap_err(),
        PlanError::ExpertRowScaleUnsupported
    );
    assert_eq!(p.expert(0, 1).unwrap().weight.byte_offset, 12);
    assert_eq!(p.expert(2, 1).unwrap().weight.dims, [1, 1, 0, 0]);
}

#[test]
fn successful_partial_provider_writes_never_commit_output() {
    struct Partial {
        prefix: usize,
    }
    impl ProjectionHost for Partial {
        type Error = u8;
        fn admits(&self, _: &Projection) -> bool {
            true
        }
        fn gemm(&self, _: &Projection, _: &[f32], out: &mut [f32]) -> Result<(), u8> {
            out[..self.prefix].fill(0.0);
            Ok(())
        }
    }
    for prefix in [0, 1] {
        let mut out = [123.];
        let mut scratch = [5.; 7];
        assert_eq!(
            execute(&Partial { prefix }, &plan(), &[1.], &mut out, &mut scratch),
            Err(Error::NonFinite)
        );
        assert_eq!(out, [123.]);
    }
}
