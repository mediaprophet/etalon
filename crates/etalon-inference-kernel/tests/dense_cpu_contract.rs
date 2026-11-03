//! Full original-engine traces on the same native target. This F32 provider does
//! not qualify every codec, WASM SIMD, physical KV rollback, or model policy.
//! Original: Qualia 07018bb8dbb123f105a91609532b9b564e54a2af. Reproduce with
//! support/regenerate_dense_cpu_golden.py generate --qualia-root CHECKOUT
//! --dependencies CHECKOUT/target/debug/deps --output NEW_FILE. The helper
//! verifies source and trace hashes; its verify subcommand checks the retained trace.
use etalon_inference_kernel::dense_cpu::*;
#[path = "support/dense_cpu_fixture.rs"]
mod fixture;
#[path = "support/dense_cpu_golden.rs"]
mod golden;
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
}
impl<'a> Kernels<'a> {
    fn new(ts: &'a [Tensor]) -> Self {
        Self {
            ts,
            calls: 0,
            fail: None,
            decode_count: None,
        }
    }
    fn call(&mut self) -> Result<(), u32> {
        self.calls += 1;
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
    fn used_bits(&self) -> Vec<u32> {
        self.0
            .iter()
            .zip(LENGTHS)
            .flat_map(|(b, n)| b[..n].iter().map(|v| v.to_bits()))
            .collect()
    }
    fn bytes(&self) -> usize {
        self.0.iter().map(|b| b.len() * 4).sum()
    }
}
#[test]
fn actual_original_full_workspace_traces_match_including_partial_failures() {
    for case in 0..7 {
        let ts = fixture::tensors(case);
        let ls = layers(&ts);
        let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
        let mut b = Buffers::new(0);
        let mut k = Kernels::new(&ts);
        for &(c, stage, token, logit, expected) in golden::GOLDEN.iter().filter(|r| r.0 == case) {
            let result = match stage {
                0 => ingest_token(&p, &mut k, b.workspace(), 1, 0, usize::MAX).map(|_| None),
                1 => run_token(&p, &mut k, b.workspace(), 3, 1, usize::MAX).map(Some),
                2 => run_token(&p, &mut k, b.workspace(), 2, 2, usize::MAX).map(Some),
                3 => {
                    reset_kv(&p, &mut b.0[0]).unwrap();
                    Ok(None)
                }
                4 => run_token(&p, &mut k, b.workspace(), 0, 0, usize::MAX).map(Some),
                _ => run_token(&p, &mut k, b.workspace(), 1, 4, usize::MAX).map(Some),
            };
            match result {
                Ok(Some(s)) => {
                    assert_eq!(i64::from(s.token_id), token);
                    assert_eq!(s.max_logit.to_bits(), logit)
                }
                Ok(None) => assert_eq!(token, -1),
                Err(ref e) => {
                    assert_eq!(token, -2, "case {c} stage {stage}: {e:?}");
                    if stage == 5 {
                        assert_eq!(
                            *e,
                            DenseError::ContextExceeded {
                                position: 4,
                                max_context: 4
                            }
                        );
                    } else {
                        let want = match case {
                            2 => DenseError::MissingTensor {
                                layer: Some(1),
                                role: DenseRole::Down,
                            },
                            3 => DenseError::Kernel {
                                layer: Some(1),
                                role: DenseRole::AttentionDimensions,
                            },
                            5 => DenseError::Kernel {
                                layer: Some(1),
                                role: DenseRole::AttentionSoftmax,
                            },
                            6 => DenseError::Kernel {
                                layer: None,
                                role: DenseRole::OutputArgmax,
                            },
                            _ => panic!("unexpected refusal"),
                        };
                        assert_eq!(*e, want);
                    }
                }
            }
            assert_eq!(b.bits(), expected, "case {c}, stage {stage}: full13buffers");
        }
    }
}
#[test]
fn spare_capacity_is_charged_but_never_exposed_or_mutated() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    let mut b = Buffers::new(3);
    for (buf, n) in b.0.iter_mut().zip(LENGTHS) {
        buf[n..].fill(123.25);
    }
    let before = b.bits();
    let mut k = Kernels::new(&ts);
    let budget = b.bytes();
    assert_eq!(
        run_token(&p, &mut k, b.workspace(), 1, 0, budget - 1),
        Err(DenseError::Workspace(DenseWorkspaceError::ByteBudget {
            required: budget,
            available: budget - 1
        }))
    );
    assert_eq!(b.bits(), before);
    assert_eq!(k.calls, 0);
    ingest_token(&p, &mut k, b.workspace(), 1, 0, budget).unwrap();
    assert_eq!(b.used_bits(), golden::GOLDEN[0].4);
    for (buf, n) in b.0.iter().zip(LENGTHS) {
        assert_eq!(&buf[n..], &[123.25; 3]);
    }
    reset_kv(&p, &mut b.0[0]).unwrap();
    assert!(b.0[0][..32].iter().all(|v| *v == 0.));
    assert_eq!(&b.0[0][32..], &[123.25; 3]);
}
#[test]
fn each_workspace_shortage_and_context_refusal_precede_callbacks_and_mutation() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    for i in 0..13 {
        let mut b = Buffers::new(0);
        for row in &mut b.0 {
            row.fill(7.);
        }
        b.0[i].pop();
        let before = b.bits();
        let mut k = Kernels::new(&ts);
        assert!(matches!(
            run_token(&p, &mut k, b.workspace(), 0, 0, usize::MAX),
            Err(DenseError::Workspace(DenseWorkspaceError::Capacity { .. }))
        ));
        assert_eq!(b.bits(), before);
        assert_eq!(k.calls, 0);
        assert_eq!(
            run_token(&p, &mut k, b.workspace(), 0, 4, 0),
            Err(DenseError::ContextExceeded {
                position: 4,
                max_context: 4
            })
        );
        assert_eq!(b.bits(), before);
    }
}
#[test]
fn checked_plan_limits_and_exact_budget_are_explicit() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let m = model(&ts, &ls);
    let p = DensePlan::new(m, usize::MAX).unwrap();
    assert_eq!(p.requirements().bytes, LENGTHS.iter().sum::<usize>() * 4);
    let n = p.requirements().bytes;
    assert!(DensePlan::new(m, n).is_ok());
    assert!(
        matches!(DensePlan::new(m,n-1),Err(DensePlanError::ByteBudget{required,available})if required==n&&available==n-1)
    );
    for c in [
        DenseConfig {
            heads: 3,
            ..config()
        },
        DenseConfig {
            kv_heads: 3,
            ..config()
        },
        DenseConfig {
            max_context: 0,
            ..config()
        },
        DenseConfig {
            vocabulary: 0,
            ..config()
        },
    ] {
        assert!(matches!(
            DensePlan::new(DenseModel { config: c, ..m }, usize::MAX),
            Err(DensePlanError::InvalidDimensions)
        ));
    }
    assert!(matches!(
        DensePlan::new(DenseModel { layers: &[], ..m }, usize::MAX),
        Err(DensePlanError::EmptyLayers)
    ));
    let c = DenseConfig {
        embedding: usize::MAX - 1,
        heads: 1,
        kv_heads: 1,
        head_dim: usize::MAX - 1,
        ..config()
    };
    assert!(matches!(
        DensePlan::new(DenseModel { config: c, ..m }, usize::MAX),
        Err(DensePlanError::ArithmeticOverflow)
    ));
}
#[test]
fn primitive_failures_propagate_without_false_success_and_late_failures_keep_kv() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    let mut complete = Kernels::new(&ts);
    let mut buf = Buffers::new(0);
    run_token(&p, &mut complete, buf.workspace(), 1, 0, usize::MAX).unwrap();
    for at in 1..=complete.calls {
        let mut b = Buffers::new(0);
        let mut k = Kernels::new(&ts);
        k.fail = Some(at);
        assert!(matches!(
            run_token(&p, &mut k, b.workspace(), 1, 0, usize::MAX),
            Err(DenseError::Tensor { source: 77, .. })
        ));
        assert_eq!(k.calls, at);
        if at == complete.calls {
            assert!(b.0[0].iter().any(|v| *v != 0.));
        }
    }
    let mut b = Buffers::new(0);
    let mut k = Kernels::new(&ts);
    k.decode_count = Some(3);
    assert_eq!(
        run_token(&p, &mut k, b.workspace(), 1, 0, usize::MAX),
        Err(DenseError::DecodeCount {
            layer: None,
            role: DenseRole::Embedding,
            expected: 4,
            actual: 3
        })
    );
}
#[test]
fn hot_execution_and_refusals_allocate_zero_after_cold_setup() {
    let ts = fixture::tensors(0);
    let ls = layers(&ts);
    let p = DensePlan::new(model(&ts, &ls), usize::MAX).unwrap();
    let mut b = Buffers::new(0);
    let mut k = Kernels::new(&ts);
    ACTIVE.with(|a| a.set(true));
    COUNT.with(|c| c.set(0));
    let control = std::hint::black_box(vec![19u8; 257]);
    std::hint::black_box(&control);
    ACTIVE.with(|a| a.set(false));
    assert!(COUNT.with(Cell::get) > 0);
    drop(control);
    COUNT.with(|c| c.set(0));
    ACTIVE.with(|a| a.set(true));
    let a = ingest_token(&p, &mut k, b.workspace(), 1, 0, usize::MAX);
    let d = run_token(&p, &mut k, b.workspace(), 3, 1, usize::MAX);
    let context = run_token(&p, &mut k, b.workspace(), 0, 4, usize::MAX);
    let budget = run_token(&p, &mut k, b.workspace(), 0, 0, 0);
    k.fail = Some(k.calls + 1);
    let failed = run_token(&p, &mut k, b.workspace(), 0, 0, usize::MAX);
    let reset = reset_kv(&p, &mut b.0[0]);
    ACTIVE.with(|a| a.set(false));
    assert_eq!(COUNT.with(Cell::get), 0);
    assert!(a.is_ok() && d.is_ok() && reset.is_ok());
    assert!(context.is_err() && budget.is_err() && failed.is_err());
}
