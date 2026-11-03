use super::*;

fn binding(handle: u8, rows: usize) -> TensorBinding<u8> {
    TensorBinding {
        handle,
        input_width: 2,
        output_rows: rows,
    }
}
fn layer() -> DenseLayer<u8> {
    DenseLayer {
        attn_norm: None,
        q: Some(binding(1, 2)),
        k: Some(binding(1, 2)),
        v: Some(binding(1, 2)),
        output: Some(binding(2, 2)),
        ffn_norm: None,
        gate: Some(binding(2, 2)),
        up: Some(binding(2, 2)),
        down: Some(binding(2, 2)),
    }
}
fn model(layers: &[DenseLayer<u8>]) -> DenseModel<'_, u8> {
    DenseModel {
        config: DenseConfig {
            embedding: 2,
            ffn: 2,
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
            max_context: 3,
            vocabulary: 3,
            rope_base: 10000.0,
            rope_scale: 1.0,
        },
        layers,
        embedding: binding(0, 3),
        output_norm: None,
        output: Some(binding(3, 3)),
    }
}

#[derive(Default)]
struct Kernels {
    calls: usize,
    nonfinite_output: bool,
}
impl TensorKernels<u8> for Kernels {
    type Error = ();
    fn decode_row_into(
        &mut self,
        tensor: TensorBinding<u8>,
        row: usize,
        out: &mut [f32],
    ) -> Result<usize, ()> {
        self.calls += 1;
        assert_eq!(out.len(), 2);
        if tensor.handle == 0 {
            out.copy_from_slice(&[1.0 + row as f32, -2.0 + row as f32]);
        } else {
            out.fill(1.0);
        }
        Ok(2)
    }
    fn gemv_into(
        &mut self,
        tensor: TensorBinding<u8>,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), ()> {
        self.calls += 1;
        assert_eq!(input.len(), 2);
        assert_eq!(out.len(), if tensor.handle == 3 { 3 } else { 2 });
        let weights = match tensor.handle {
            1 => [1., 0., 0., 1., 0., 0.],
            2 => [0.; 6],
            3 => [1., 0., 0., 1., -1., -2.],
            _ => return Err(()),
        };
        for (row, dst) in out.iter_mut().take(tensor.output_rows).enumerate() {
            *dst = weights[row * 2..row * 2 + 2]
                .iter()
                .zip(input)
                .map(|(w, x)| w * x)
                .sum();
        }
        if tensor.handle == 3 && self.nonfinite_output {
            out.fill(f32::NAN);
        }
        Ok(())
    }
}

struct Storage {
    kv: [f32; 32],
    hidden: [f32; 4],
    normed: [f32; 4],
    norm_weight: [f32; 4],
    q: [f32; 4],
    k: [f32; 4],
    v: [f32; 4],
    attention: [f32; 4],
    projection: [f32; 4],
    gate: [f32; 4],
    up: [f32; 4],
    scores: [f32; 4],
    logits: [f32; 4],
}
impl Storage {
    fn new() -> Self {
        Self {
            kv: [0.; 32],
            hidden: [0.; 4],
            normed: [0.; 4],
            norm_weight: [0.; 4],
            q: [0.; 4],
            k: [0.; 4],
            v: [0.; 4],
            attention: [0.; 4],
            projection: [0.; 4],
            gate: [0.; 4],
            up: [0.; 4],
            scores: [0.; 4],
            logits: [0.; 4],
        }
    }
    fn view(&mut self) -> DenseWorkspace<'_> {
        DenseWorkspace {
            kv: &mut self.kv,
            hidden: &mut self.hidden,
            normed: &mut self.normed,
            norm_weight: &mut self.norm_weight,
            q: &mut self.q,
            k: &mut self.k,
            v: &mut self.v,
            attention: &mut self.attention,
            projection: &mut self.projection,
            gate: &mut self.gate,
            up: &mut self.up,
            scores: &mut self.scores,
            logits: &mut self.logits,
        }
    }
}

#[test]
fn complete_two_layer_execution_retains_known_logits_and_spare_buffers() {
    let layers = [layer(); 2];
    let plan = DensePlan::new(model(&layers), usize::MAX).unwrap();
    let mut storage = Storage::new();
    storage.hidden[2..].fill(91.);
    storage.logits[3] = 92.;
    let mut kernels = Kernels::default();
    let first = run_token(&plan, &mut kernels, storage.view(), 0, 0, usize::MAX).unwrap();
    assert_eq!(
        first,
        DenseStep {
            token_id: 2,
            max_logit: 3.
        }
    );
    assert_eq!(&storage.logits[..3], &[1., -2., 3.]);
    assert_eq!(&storage.kv[..2], &[1., -2.]);
    assert_eq!(&storage.kv[6..8], &[1., -2.]);
    assert_eq!(kernels.calls, 16);
    let second = run_token(&plan, &mut kernels, storage.view(), 1, 1, usize::MAX).unwrap();
    assert_eq!(
        second,
        DenseStep {
            token_id: 0,
            max_logit: 2.
        }
    );
    assert_eq!(&storage.logits[..3], &[2., -1., 0.]);
    assert_eq!(&storage.hidden[2..], &[91., 91.]);
    assert_eq!(storage.logits[3], 92.);
}

#[test]
fn ingest_skips_projection_reset_clears_only_used_kv() {
    let layers = [layer()];
    let plan = DensePlan::new(model(&layers), usize::MAX).unwrap();
    let mut storage = Storage::new();
    storage.logits.fill(81.);
    storage.kv[12..].fill(82.);
    let mut kernels = Kernels::default();
    ingest_token(&plan, &mut kernels, storage.view(), 0, 0, usize::MAX).unwrap();
    assert_eq!(kernels.calls, 8);
    assert_eq!(storage.logits, [81.; 4]);
    reset_kv(&plan, &mut storage.kv).unwrap();
    assert_eq!(&storage.kv[..12], &[0.; 12]);
    assert!(storage.kv[12..].iter().all(|x| *x == 82.));
    assert_eq!(&storage.hidden[..2], &[1., -2.]);
}

#[test]
fn failure_order_preserves_partial_state() {
    let mut layers = [layer()];
    layers[0].gate = None;
    let plan = DensePlan::new(model(&layers), usize::MAX).unwrap();
    let mut storage = Storage::new();
    let error = run_token(
        &plan,
        &mut Kernels::default(),
        storage.view(),
        0,
        0,
        usize::MAX,
    )
    .unwrap_err();
    assert_eq!(
        error,
        DenseError::MissingTensor {
            layer: Some(0),
            role: DenseRole::Gate
        }
    );
    assert_eq!(&storage.kv[..2], &[1., -2.]);
    layers[0].q = None;
    let plan = DensePlan::new(model(&layers), usize::MAX).unwrap();
    let mut storage = Storage::new();
    let error = run_token(
        &plan,
        &mut Kernels::default(),
        storage.view(),
        0,
        0,
        usize::MAX,
    )
    .unwrap_err();
    assert_eq!(
        error,
        DenseError::MissingTensor {
            layer: Some(0),
            role: DenseRole::Q
        }
    );
    assert_eq!(storage.kv, [0.; 32]);
}

#[test]
fn budget_and_context_refuse_before_tensor_calls() {
    let layers = [layer()];
    let plan = DensePlan::new(model(&layers), usize::MAX).unwrap();
    let mut storage = Storage::new();
    let mut kernels = Kernels::default();
    let error = run_token(
        &plan,
        &mut kernels,
        storage.view(),
        0,
        0,
        plan.requirements().bytes,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Workspace(DenseWorkspaceError::ByteBudget { .. })
    ));
    assert_eq!(kernels.calls, 0);
    let error = run_token(&plan, &mut kernels, storage.view(), 0, 3, usize::MAX).unwrap_err();
    assert_eq!(
        error,
        DenseError::ContextExceeded {
            position: 3,
            max_context: 3
        }
    );
    assert_eq!(kernels.calls, 0);
}

#[test]
fn all_nonfinite_logits_fail_after_the_real_forward() {
    let layers = [layer()];
    let plan = DensePlan::new(model(&layers), usize::MAX).unwrap();
    let mut storage = Storage::new();
    let mut kernels = Kernels {
        nonfinite_output: true,
        ..Kernels::default()
    };
    assert_eq!(
        run_token(&plan, &mut kernels, storage.view(), 0, 0, usize::MAX),
        Err(DenseError::Kernel {
            layer: None,
            role: DenseRole::OutputArgmax
        })
    );
    assert_eq!(&storage.kv[..2], &[1., -2.]);
}
