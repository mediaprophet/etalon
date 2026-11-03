use super::*;
use dense::{DenseConfig, DenseLayer, DenseModel, TensorBinding};

fn binding() -> TensorBinding<u8> {
    TensorBinding {
        handle: 0,
        input_width: 2,
        output_rows: 2,
    }
}
fn layer() -> DenseLayer<u8> {
    DenseLayer {
        attn_norm: None,
        q: Some(binding()),
        k: Some(binding()),
        v: Some(binding()),
        output: Some(binding()),
        ffn_norm: None,
        gate: Some(binding()),
        up: Some(binding()),
        down: Some(binding()),
    }
}
fn plan(layers: &[DenseLayer<u8>]) -> DensePlan<'_, u8> {
    DensePlan::new(
        DenseModel {
            config: DenseConfig {
                embedding: 2,
                ffn: 2,
                heads: 1,
                kv_heads: 1,
                head_dim: 2,
                max_context: 2,
                vocabulary: 2,
                rope_base: 10000.,
                rope_scale: 1.,
            },
            layers,
            embedding: binding(),
            output_norm: None,
            output: Some(binding()),
        },
        1024,
    )
    .unwrap()
}
fn storage(plan: &DensePlan<'_, u8>) -> [Vec<f32>; 13] {
    requirements(plan.requirements()).map(|(_, n)| vec![f32::from_bits(0x80000000); n + 1])
}
fn workspace(b: &mut [Vec<f32>; 13]) -> DenseWorkspace<'_> {
    let [kv, hidden, normed, norm_weight, q, k, v, attention, projection, gate, up, scores, logits] =
        b;
    DenseWorkspace {
        kv,
        hidden,
        normed,
        norm_weight,
        q,
        k,
        v,
        attention,
        projection,
        gate,
        up,
        scores,
        logits,
    }
}
fn bits(b: &[Vec<f32>; 13]) -> Vec<u32> {
    b.iter().flatten().map(|x| x.to_bits()).collect()
}
#[derive(Default)]
struct Kernels {
    calls: usize,
    fail_at: usize,
    panic_at: usize,
}
impl Kernels {
    fn call(&mut self) -> Result<(), u8> {
        self.calls += 1;
        assert_ne!(self.calls, self.panic_at, "injected callback unwind");
        if self.calls == self.fail_at {
            Err(7)
        } else {
            Ok(())
        }
    }
}
impl TensorKernels<u8> for Kernels {
    type Error = u8;
    fn decode_row_into(
        &mut self,
        _: TensorBinding<u8>,
        _: usize,
        out: &mut [f32],
    ) -> Result<usize, u8> {
        self.call()?;
        out[..2].copy_from_slice(&[1., 2.]);
        Ok(2)
    }
    fn gemv_into(
        &mut self,
        _: TensorBinding<u8>,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), u8> {
        self.call()?;
        out[..2].copy_from_slice(&input[..2]);
        Ok(())
    }
}

#[test]
fn abort_drop_and_retry_preserve_exact_full_state() {
    let layers = [layer()];
    let plan = plan(&layers);
    let mut buffers = storage(&plan);
    buffers[0][0] = f32::from_bits(0x7fc01234);
    let original = bits(&buffers);
    let count = original.len();
    let mut undo = vec![42.; count + 3];
    for explicit in [true, false] {
        let pending = propose_dense(
            &plan,
            &mut Kernels::default(),
            workspace(&mut buffers),
            &mut undo,
            0,
            0,
            4096,
        )
        .unwrap();
        assert!(pending.proposal().max_logit.is_finite());
        assert_eq!(pending.observations().len(), 13);
        if explicit {
            pending.abort();
        } else {
            drop(pending);
        }
        assert_eq!(bits(&buffers), original);
        assert_eq!(&undo[count..], &[42.; 3]);
    }
    let mut clean = buffers.clone();
    let mut clean_undo = vec![0.; count];
    let expected = propose_dense(
        &plan,
        &mut Kernels::default(),
        workspace(&mut clean),
        &mut clean_undo,
        0,
        0,
        4096,
    )
    .unwrap()
    .commit();
    let actual = propose_dense(
        &plan,
        &mut Kernels::default(),
        workspace(&mut buffers),
        &mut undo,
        0,
        0,
        4096,
    )
    .unwrap()
    .commit();
    assert_eq!(actual, expected);
    assert_eq!(bits(&buffers), bits(&clean));
    assert_ne!(bits(&buffers), original);
}

#[test]
fn every_callback_error_and_unwind_restores_before_return() {
    let layers = [layer()];
    let plan = plan(&layers);
    let mut buffers = storage(&plan);
    let original = bits(&buffers);
    let mut undo = vec![0.; original.len()];
    for point in 1..=9 {
        let mut kernels = Kernels {
            fail_at: point,
            ..Default::default()
        };
        assert!(matches!(
            propose_dense(
                &plan,
                &mut kernels,
                workspace(&mut buffers),
                &mut undo,
                0,
                0,
                4096
            ),
            Err(TransactionError::Execution(DenseError::Tensor { .. }))
        ));
        assert_eq!(kernels.calls, point);
        assert_eq!(bits(&buffers), original);
        let mut kernels = Kernels {
            panic_at: point,
            ..Default::default()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = propose_dense(
                &plan,
                &mut kernels,
                workspace(&mut buffers),
                &mut undo,
                0,
                0,
                4096,
            );
        }));
        assert!(result.is_err());
        assert_eq!(bits(&buffers), original);
    }
}

#[test]
fn preflight_refuses_without_callback_or_buffer_mutation() {
    let layers = [layer()];
    let plan = plan(&layers);
    let mut buffers = storage(&plan);
    let original = bits(&buffers);
    let count = original.len();
    let mut undo = vec![42.; count + 3];
    let mut kernels = Kernels::default();
    assert!(matches!(
        propose_dense(
            &plan,
            &mut kernels,
            workspace(&mut buffers),
            &mut undo,
            0,
            2,
            4096
        ),
        Err(TransactionError::Execution(
            DenseError::ContextExceeded { .. }
        ))
    ));
    assert!(matches!(
        propose_dense(
            &plan,
            &mut kernels,
            workspace(&mut buffers),
            &mut undo[..count - 1],
            0,
            0,
            4096
        ),
        Err(TransactionError::UndoCapacity { .. })
    ));
    let exact_budget = (count * 2 + 3) * 4;
    assert!(
        matches!(propose_dense(&plan, &mut kernels, workspace(&mut buffers), &mut undo, 0, 0, exact_budget-1), Err(TransactionError::Budget { required, .. }) if required == exact_budget)
    );
    assert_eq!(kernels.calls, 0);
    assert_eq!(bits(&buffers), original);
    assert!(undo.iter().all(|&x| x == 42.));
    propose_dense(
        &plan,
        &mut kernels,
        workspace(&mut buffers),
        &mut undo,
        0,
        0,
        exact_budget,
    )
    .unwrap()
    .abort();
    assert_eq!(bits(&buffers), original);
    assert_eq!(&undo[count..], &[42.; 3]);
}

#[test]
fn checked_snapshot_arithmetic_refuses_overflow() {
    assert_eq!(
        checked_sizes::<u8>(&[usize::MAX; 13], 0, usize::MAX),
        Err(TransactionError::ArithmeticOverflow)
    );
    assert_eq!(
        checked_sizes::<u8>(&[1; 13], usize::MAX, usize::MAX),
        Err(TransactionError::ArithmeticOverflow)
    );
}
