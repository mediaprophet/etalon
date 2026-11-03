//! Numerical transactions only; real controller decisions remain a host obligation.
use etalon_inference::runtime::dense_transaction::{propose_dense, TransactionError};
use etalon_inference_kernel::dense_cpu::*;
#[path = "../../etalon-inference-kernel/tests/support/dense_cpu_fixture.rs"]
mod fixture;
use fixture::{Tensor, LENGTHS};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
struct Counter;
thread_local! {static ACTIVE:Cell<bool>=const{Cell::new(false)};static COUNT:Cell<usize>=const{Cell::new(0)};}
fn tick() {
    let _ = ACTIVE.try_with(|a| {
        if a.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        tick();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        tick();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        tick();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
fn config() -> DenseConfig {
    DenseConfig {
        embedding: 4,
        ffn: 6,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        max_context: 4,
        vocabulary: 5,
        rope_base: 10000.,
        rope_scale: 1.,
    }
}
fn binding(ts: &[Tensor], name: &str) -> Option<TensorBinding<usize>> {
    ts.iter()
        .position(|t| t.name == name)
        .map(|i| TensorBinding {
            handle: i,
            input_width: ts[i].width,
            output_rows: ts[i].rows,
        })
}
fn layers(ts: &[Tensor]) -> [DenseLayer<usize>; 2] {
    std::array::from_fn(|i| {
        let get = |s: &str| binding(ts, &format!("blk.{i}.{s}.weight"));
        DenseLayer {
            attn_norm: get("attn_norm"),
            q: get("attn_q"),
            k: get("attn_k"),
            v: get("attn_v"),
            output: get("attn_output"),
            ffn_norm: get("ffn_norm"),
            gate: get("ffn_gate"),
            up: get("ffn_up"),
            down: get("ffn_down"),
        }
    })
}
fn model<'a>(ts: &[Tensor], layers: &'a [DenseLayer<usize>]) -> DenseModel<'a, usize> {
    DenseModel {
        config: config(),
        layers,
        embedding: binding(ts, "token_embd.weight").unwrap(),
        output_norm: binding(ts, "output_norm.weight"),
        output: binding(ts, "output.weight"),
    }
}
struct Kernels<'a> {
    ts: &'a [Tensor],
    calls: usize,
    fail: Option<usize>,
    decode_count: Option<usize>,
    panic_at: Option<usize>,
}
impl<'a> Kernels<'a> {
    fn new(ts: &'a [Tensor]) -> Self {
        Self {
            ts,
            calls: 0,
            fail: None,
            decode_count: None,
            panic_at: None,
        }
    }
    fn call(&mut self) -> Result<(), u32> {
        self.calls += 1;
        assert_ne!(self.panic_at, Some(self.calls), "injected unwind");
        if self.fail == Some(self.calls) {
            Err(77)
        } else {
            Ok(())
        }
    }
}
impl TensorKernels<usize> for Kernels<'_> {
    type Error = u32;
    fn decode_row_into(
        &mut self,
        t: TensorBinding<usize>,
        row: usize,
        out: &mut [f32],
    ) -> Result<usize, u32> {
        self.call()?;
        let data = &self.ts[t.handle];
        if row >= data.rows || data.width > out.len() {
            return Ok(0);
        }
        out[..data.width].copy_from_slice(&data.values[row * data.width..(row + 1) * data.width]);
        Ok(self.decode_count.unwrap_or(data.width))
    }
    fn gemv_into(
        &mut self,
        t: TensorBinding<usize>,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), u32> {
        self.call()?;
        let data = &self.ts[t.handle];
        assert!(input.len() == 4 || input.len() == 6);
        assert!(out.len() <= 6);
        for (row, value) in out.iter_mut().take(t.output_rows).enumerate() {
            *value = data.values[row * t.input_width..(row + 1) * t.input_width]
                .iter()
                .zip(&input[..t.input_width])
                .map(|(w, x)| w * x)
                .sum();
        }
        Ok(())
    }
}
struct Buffers([Vec<f32>; 13]);
impl Buffers {
    fn new(extra: usize) -> Self {
        Self(std::array::from_fn(|i| vec![0.; LENGTHS[i] + extra]))
    }
    fn workspace(&mut self) -> DenseWorkspace<'_> {
        let [kv, hidden, normed, norm_weight, q, k, v, attention, projection, gate, up, scores, logits] =
            &mut self.0;
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
    fn bits(&self) -> Vec<u32> {
        self.0
            .iter()
            .flat_map(|b| b.iter().map(|v| v.to_bits()))
            .collect()
    }
    fn bytes(&self) -> usize {
        self.0.iter().map(|b| b.len() * 4).sum()
    }
}

fn measured(f: impl FnOnce()) -> usize {
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            ACTIVE.with(|v| v.set(false));
        }
    }
    COUNT.with(|v| v.set(0));
    ACTIVE.with(|v| v.set(true));
    let stop = Stop;
    f();
    drop(stop);
    COUNT.with(Cell::get)
}
fn poison(b: &mut Buffers) {
    for (i, v) in b.0.iter_mut().flatten().enumerate() {
        *v = f32::from_bits([0x80000000, 0x7fc12345, 0xffc23456, 0x3f000000][i % 4]);
    }
}

#[test]
fn every_short_workspace_refuses_before_callbacks_or_snapshot() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    for (i, n) in LENGTHS.into_iter().enumerate() {
        let mut b = Buffers::new(2);
        poison(&mut b);
        b.0[i].truncate(n - 1);
        let before = b.bits();
        let mut undo = [f32::from_bits(0x7fc45678); 128];
        let mut k = Kernels::new(&ts);
        let result = propose_dense(&p, &mut k, b.workspace(), &mut undo, 1, 0, usize::MAX);
        assert!(matches!(
            result,
            Err(TransactionError::Execution(DenseError::Workspace(_)))
        ));
        drop(result);
        assert_eq!(k.calls, 0);
        assert_eq!(b.bits(), before);
        assert!(undo.iter().all(|v| v.to_bits() == 0x7fc45678));
    }
}

#[test]
fn missing_binding_and_decode_count_failures_restore_partial_numerical_state() {
    for case in [0, 2, 3] {
        let ts = fixture::tensors(case);
        let ls = layers(&ts);
        let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
        let mut b = Buffers::new(2);
        poison(&mut b);
        let before = b.bits();
        let mut k = Kernels::new(&ts);
        let mut undo = [0.0; 128];
        let token = if case == 0 { 99 } else { 1 };
        let result = propose_dense(&p, &mut k, b.workspace(), &mut undo, token, 0, usize::MAX);
        assert!(matches!(result, Err(TransactionError::Execution(_))));
        drop(result);
        assert!(k.calls > 0);
        assert_eq!(b.bits(), before, "case {case}");
    }
}

#[test]
fn actual_norm_embedding_gemv_and_output_failures_restore_every_bit_and_retry() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    let mut clean = Buffers::new(2);
    let mut good = Kernels::new(&ts);
    ingest_token(&p, &mut good, clean.workspace(), 1, 0, usize::MAX).unwrap();
    let prefix = clean.bits();
    good.calls = 0;
    let expected = run_token(&p, &mut good, clean.workspace(), 3, 1, usize::MAX).unwrap();
    let total_calls = good.calls;
    assert_eq!(total_calls, 21);
    for fail in 1..=total_calls {
        let mut b = Buffers::new(2);
        let mut k = Kernels::new(&ts);
        ingest_token(&p, &mut k, b.workspace(), 1, 0, usize::MAX).unwrap();
        assert_eq!(b.bits(), prefix);
        k.calls = 0;
        k.fail = Some(fail);
        let mut undo = [0.0; 128];
        let result = propose_dense(&p, &mut k, b.workspace(), &mut undo, 3, 1, usize::MAX);
        assert!(matches!(
            result,
            Err(TransactionError::Execution(DenseError::Tensor {
                source: 77,
                ..
            }))
        ));
        drop(result);
        assert_eq!(b.bits(), prefix, "callback {fail}");
        k.fail = None;
        k.calls = 0;
        let accepted = propose_dense(&p, &mut k, b.workspace(), &mut undo, 3, 1, usize::MAX)
            .unwrap()
            .commit();
        assert_eq!(accepted.token_id, expected.token_id);
        assert_eq!(accepted.max_logit.to_bits(), expected.max_logit.to_bits());
        assert_eq!(b.bits(), clean.bits());
    }
}

#[test]
fn error_abort_drop_and_unwind_restore_nan_signed_zero_and_spare_tails() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    for mode in 0..24 {
        let mut b = Buffers::new(2);
        poison(&mut b);
        let before = b.bits();
        let mut undo = [f32::from_bits(0x7fc45678); 128];
        let mut k = Kernels::new(&ts);
        if mode < 21 {
            k.panic_at = Some(mode + 1);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let pending =
                propose_dense(&p, &mut k, b.workspace(), &mut undo, 1, 0, usize::MAX).unwrap();
            if mode == 21 {
                pending.abort();
            } else if mode == 22 {
                drop(pending);
            } else {
                panic!("owner unwind");
            }
        }));
        assert_eq!(result.is_err(), mode != 21 && mode != 22);
        assert_eq!(b.bits(), before, "mode {mode}");
        let count = before.len();
        assert_eq!(
            undo[..count]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            before
        );
        assert!(undo[count..].iter().all(|v| v.to_bits() == 0x7fc45678));
    }
}

#[test]
fn complete_capacities_are_charged_and_preflight_never_mutates() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    for mode in 0..4 {
        let mut b = Buffers::new(2);
        poison(&mut b);
        let before = b.bits();
        let elements = b.bytes() / 4;
        let mut undo = vec![
            f32::from_bits(0x7fc45678);
            if mode == 0 {
                elements - 1
            } else {
                elements + 7
            }
        ];
        let budget = if mode == 1 {
            b.bytes() + undo.len() * 4 - 1
        } else {
            usize::MAX
        };
        let position = if mode == 2 { 4 } else { 0 };
        let mut k = Kernels::new(&ts);
        let result = propose_dense(&p, &mut k, b.workspace(), &mut undo, 1, position, budget);
        match mode {
            0 => assert!(
                matches!(result,Err(TransactionError::UndoCapacity{required,available}) if required==elements && available==elements-1)
            ),
            1 => assert!(matches!(result, Err(TransactionError::Budget { .. }))),
            2 => assert!(matches!(
                result,
                Err(TransactionError::Execution(
                    DenseError::ContextExceeded { .. }
                ))
            )),
            _ => {
                result.unwrap().abort();
                assert_eq!(b.bits(), before);
                continue;
            }
        }
        drop(result);
        assert_eq!(k.calls, 0);
        assert_eq!(b.bits(), before);
        assert!(undo.iter().all(|v| v.to_bits() == 0x7fc45678));
    }
}

#[test]
fn pending_observations_commit_abort_and_refusal_are_measured_zero_alloc() {
    assert!(
        measured(|| {
            let v = vec![7u8; 1024];
            std::hint::black_box(&v);
        }) > 0
    );
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    let mut b = Buffers::new(2);
    let mut k = Kernels::new(&ts);
    let mut undo = [0.0; 128];
    let budget = b.bytes() + undo.len() * 4;
    let count = measured(|| {
        let pending = propose_dense(&p, &mut k, b.workspace(), &mut undo, 1, 0, budget).unwrap();
        let candidate = pending.proposal();
        let observations = pending.observations();
        assert_eq!(observations.len(), 13);
        std::hint::black_box(observations);
        let accepted = pending.commit();
        assert_eq!(accepted.token_id, candidate.token_id);
        propose_dense(&p, &mut k, b.workspace(), &mut undo, 3, 1, budget)
            .unwrap()
            .abort();
        drop(propose_dense(&p, &mut k, b.workspace(), &mut undo, 3, 1, budget).unwrap());
        k.calls = 0;
        k.fail = Some(21);
        assert!(propose_dense(&p, &mut k, b.workspace(), &mut undo, 3, 1, budget).is_err());
        assert!(propose_dense(&p, &mut k, b.workspace(), &mut undo, 3, 1, budget - 1).is_err());
    });
    assert_eq!(count, 0);
}
