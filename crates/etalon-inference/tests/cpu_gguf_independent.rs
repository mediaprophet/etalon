#![cfg(feature = "gguf-cpu")]
use etalon_inference::cpu_gguf::*;
use etalon_inference_kernel::dense_cpu::TensorKernels;
use etalon_model::{
    ggml,
    gguf::{ParseLimits, VersionPolicy},
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counter;
thread_local! {static ACTIVE:Cell<bool>=const{Cell::new(false)};static COUNT:Cell<usize>=const{Cell::new(0)};}
fn tick() {
    if ACTIVE.try_with(Cell::get).unwrap_or(false) {
        COUNT.with(|n| n.set(n.get() + 1));
    }
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
fn measured(f: impl FnOnce()) -> usize {
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            ACTIVE.with(|v| v.set(false));
        }
    }
    COUNT.with(|n| n.set(0));
    ACTIVE.with(|v| v.set(true));
    let stop = Stop;
    f();
    drop(stop);
    COUNT.with(Cell::get)
}
#[derive(Clone)]
struct Tensor {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: Option<u64>,
    blob: Vec<u8>,
}
type Metadata = Vec<(String, u32, Vec<u8>)>;
fn string(b: &mut Vec<u8>, s: &[u8]) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s);
}
fn text(s: &str) -> Vec<u8> {
    let mut b = vec![];
    string(&mut b, s.as_bytes());
    b
}
fn integer(k: &str, n: u32) -> (String, u32, Vec<u8>) {
    (k.into(), 4, n.to_le_bytes().to_vec())
}
fn base(width: usize, layers: usize) -> (Metadata, Vec<Tensor>) {
    let meta = vec![
        ("general.architecture".into(), 8, text("llama")),
        integer("llama.block_count", layers as u32),
        integer("llama.embedding_length", width as u32),
        integer("llama.attention.head_count", 1),
        integer("llama.feed_forward_length", width as u32),
    ];
    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
    ];
    for i in 0..layers {
        for role in [
            "attn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            names.push(format!("blk.{i}.{role}.weight"));
        }
    }
    let tensors = names
        .into_iter()
        .map(|name| {
            let dims = if name.contains("norm") {
                vec![width as u64]
            } else {
                vec![
                    width as u64,
                    if name == "token_embd.weight" {
                        2
                    } else {
                        width as u64
                    },
                ]
            };
            let count = dims.iter().product::<u64>() as usize;
            let blob = (0..count)
                .flat_map(|i| (((i * 5 % 23) as f32 - 11.) * 0.125).to_le_bytes())
                .collect();
            Tensor {
                name,
                dims,
                dtype: 0,
                offset: None,
                blob,
            }
        })
        .collect();
    (meta, tensors)
}
fn image(meta: &Metadata, ts: &[Tensor]) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(ts.len() as u64).to_le_bytes());
    b.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (k, t, v) in meta {
        string(&mut b, k.as_bytes());
        b.extend_from_slice(&t.to_le_bytes());
        b.extend_from_slice(v);
    }
    let mut offset = 0;
    for t in ts {
        string(&mut b, t.name.as_bytes());
        b.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for n in &t.dims {
            b.extend_from_slice(&n.to_le_bytes());
        }
        b.extend_from_slice(&t.dtype.to_le_bytes());
        b.extend_from_slice(&t.offset.unwrap_or(offset).to_le_bytes());
        offset += t.blob.len() as u64;
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    for t in ts {
        b.extend_from_slice(&t.blob);
    }
    b
}
fn limits() -> ParseLimits {
    ParseLimits {
        max_metadata_bytes: 65536,
        max_nesting_depth: 4,
        max_work: 10000,
        versions: VersionPolicy::V2V3,
    }
}
fn budget() -> BindingLimits {
    BindingLimits {
        byte_budget: 65536,
        max_tensor_visits: 10000,
    }
}
fn refusal(meta: &Metadata, ts: &[Tensor], expected: GgufCpuError) {
    let bytes = image(meta, ts);
    let config = inspect_dense_gguf(&bytes, limits(), 4).unwrap();
    let mut layers = [empty_layer(); 3];
    let before = format!("{layers:?}");
    assert_eq!(
        prepare_dense_gguf_into(&config, &mut layers, budget()).unwrap_err(),
        expected
    );
    assert_eq!(format!("{layers:?}"), before);
}

#[test]
fn unsupported_semantics_and_duplicate_or_malformed_metadata_refuse() {
    let (meta, ts) = base(4, 2);
    for extra in [
        integer("llama.rope.dimension_count", 2),
        integer("llama.attention.value_length", 2),
        integer("llama.attention.sliding_window", 2),
        integer("llama.attention.logit_softcap", 5),
        integer("llama.expert_count", 8),
        integer("llama.unknown_numerical_setting", 1),
        integer("llama.context_length", 3),
    ] {
        let mut m = meta.clone();
        m.push(extra);
        let b = image(&m, &ts);
        assert_eq!(
            inspect_dense_gguf(&b, limits(), 4).unwrap_err(),
            GgufCpuError::UnsupportedProfile
        );
    }
    for extra in [
        integer("llama.attention.key_length", 2),
        integer("llama.vocab_size", 3),
    ] {
        let mut m = meta.clone();
        m.push(extra);
        let b = image(&m, &ts);
        assert_eq!(
            inspect_dense_gguf(&b, limits(), 4).unwrap_err(),
            GgufCpuError::InvalidDimensions
        );
    }
    for key in ["llama.block_count", "llama.rope.scaling.type"] {
        let mut m = meta.clone();
        let v = if key.ends_with("type") {
            (key.into(), 8, text("none"))
        } else {
            integer(key, 2)
        };
        if key.ends_with("type") {
            m.push(v.clone());
        }
        m.push(v);
        let b = image(&m, &ts);
        assert_eq!(
            inspect_dense_gguf(&b, limits(), 4).unwrap_err(),
            GgufCpuError::DuplicateMetadata
        );
    }
    for value in [f32::NAN, f32::INFINITY, 0., -1.] {
        let mut m = meta.clone();
        m.push((
            "llama.rope.freq_base".into(),
            6,
            value.to_le_bytes().to_vec(),
        ));
        assert!(inspect_dense_gguf(&image(&m, &ts), limits(), 4).is_err());
    }
    let mut m = meta.clone();
    m.extend([
        integer("llama.rope.dimension_count", 4),
        integer("llama.attention.value_length", 4),
        integer("llama.context_length", 4),
        integer("llama.vocab_size", 2),
    ]);
    assert!(inspect_dense_gguf(&image(&m, &ts), limits(), 4).is_ok());
    let b = image(&meta, &ts);
    for context in [0, usize::MAX] {
        assert!(inspect_dense_gguf(&b, limits(), context).is_err());
    }
    for cut in 0..24 {
        assert!(inspect_dense_gguf(&b[..cut], limits(), 4).is_err());
    }
    let mut l = limits();
    l.max_work = 1;
    assert!(inspect_dense_gguf(&b, l, 4).is_err());
}

#[test]
fn late_tensor_failure_preserves_all_binding_storage() {
    let (meta, ts) = base(4, 2);
    let mut t = ts.clone();
    t.last_mut().unwrap().dims[0] = 3;
    refusal(&meta, &t, GgufCpuError::InvalidDimensions);
    let mut t = ts.clone();
    t.last_mut().unwrap().blob.pop();
    refusal(&meta, &t, GgufCpuError::Payload);
    let mut t = ts.clone();
    t.last_mut().unwrap().offset = Some(u64::MAX);
    refusal(&meta, &t, GgufCpuError::ArithmeticOverflow);
    let mut t = ts.clone();
    t.last_mut().unwrap().dtype = 158;
    refusal(&meta, &t, GgufCpuError::UnsupportedTensor);
    let mut t = ts.clone();
    t.push(t.last().unwrap().clone());
    refusal(&meta, &t, GgufCpuError::DuplicateTensor);
    let mut t = ts.clone();
    t.pop();
    refusal(&meta, &t, GgufCpuError::MissingTensor);
    for name in [
        "blk.1.attn_q.bias",
        "unknown.weight",
        "blk.2.attn_q.weight",
        "blk.999999999999999999999999999.attn_q.weight",
    ] {
        let mut t = ts.clone();
        t.last_mut().unwrap().name = name.into();
        refusal(
            &meta,
            &t,
            if name.starts_with("blk.2.") {
                GgufCpuError::InvalidDimensions
            } else if name.starts_with("blk.999") {
                GgufCpuError::ArithmeticOverflow
            } else {
                GgufCpuError::UnsupportedTensor
            },
        );
    }
}

#[test]
fn exact_binding_capacity_and_work_budgets_include_spare_capacity() {
    let (meta, ts) = base(4, 2);
    let b = image(&meta, &ts);
    let c = inspect_dense_gguf(&b, limits(), 4).unwrap();
    let mut layers = [empty_layer(); 3];
    let size = std::mem::size_of_val(&layers);
    let visits = 5 * ts.len() as u64;
    for lim in [
        BindingLimits {
            byte_budget: size - 1,
            max_tensor_visits: visits,
        },
        BindingLimits {
            byte_budget: size,
            max_tensor_visits: visits - 1,
        },
    ] {
        assert_eq!(
            prepare_dense_gguf_into(&c, &mut layers, lim).unwrap_err(),
            GgufCpuError::Budget
        );
        assert!(layers.iter().all(|l| l.q.is_none()));
    }
    assert_eq!(
        prepare_dense_gguf_into(&c, &mut layers[..1], budget()).unwrap_err(),
        GgufCpuError::Capacity
    );
    let m = prepare_dense_gguf_into(
        &c,
        &mut layers,
        BindingLimits {
            byte_budget: size,
            max_tensor_visits: visits,
        },
    )
    .unwrap();
    assert_eq!(m.layers.len(), 2);
    assert!(layers[2].q.is_none());
}

fn codec_bytes(dtype: u32) -> Vec<u8> {
    let row = ggml::ggml_row_bytes(dtype, 256).unwrap();
    let mut raw = vec![0; row * 2];
    for (r, bytes) in raw.chunks_exact_mut(row).enumerate() {
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 13 + r * 17 + 7) % 127) as u8;
        }
        match dtype {
            0 => {
                for (i, b) in bytes.chunks_exact_mut(4).enumerate() {
                    b.copy_from_slice(&(((i * 5 % 23) as f32 - 11.) * 0.125).to_le_bytes());
                }
            }
            1 | 30 => {
                for b in bytes.chunks_exact_mut(2) {
                    b.copy_from_slice(&0x3400u16.to_le_bytes());
                }
            }
            40 => {}
            112 => {
                for b in bytes[128..160].chunks_exact_mut(2) {
                    b.copy_from_slice(&0x3000u16.to_le_bytes());
                }
            }
            14 => bytes[208..210].copy_from_slice(&0x3000u16.to_le_bytes()),
            _ => {
                let layout = ggml::ggml_block_layout(dtype).unwrap();
                for b in bytes.chunks_exact_mut(layout.block_bytes) {
                    b[..2].copy_from_slice(&0x3000u16.to_le_bytes());
                    if dtype == 12 || dtype == 13 {
                        b[2..4].copy_from_slice(&0x2800u16.to_le_bytes());
                    }
                }
            }
        }
    }
    raw
}

// Actual original native stack_gemm_quant outcomes, not WASM Q8 SIMD outcomes.
// Frozen driver SHA256 e788ac62977a1138914c6ae9c8638c7bf67a6b1ecdfdd4f4e080fe24677f7c29;
// codec-matrix.log SHA256 76dbfa70917c7d398282e8b797310210d584ff3a1dfd2c59bb2ac44ed2a0a0e7.
#[test]
fn thirteen_codecs_match_actual_original_native_gemv_bits() {
    let cases = [
        (0, [1073840128, 1073840128]),
        (1, [3187671040, 3187671040]),
        (2, [1056440320, 3219652608]),
        (6, [3222044672, 1085669376]),
        (8, [3204186112, 1092984832]),
        (12, [3268349696, 3269524736]),
        (13, [1118389504, 1135899328]),
        (14, [1145978240, 1142766976]),
        (20, [1084850176, 3255109632]),
        (23, [1136315136, 3289743488]),
        (30, [3011510272, 3011510272]),
        (40, [3262724704, 3259751840]),
        (112, [3212705792, 3231793152]),
    ];
    let input = std::array::from_fn::<_, 256, _>(|i| ((i * 3 % 17) as f32 - 8.) * 0.0625);
    for (dtype, expected) in cases {
        let (meta, mut ts) = base(256, 1);
        ts[0].dtype = dtype;
        ts[0].blob = codec_bytes(dtype);
        let b = image(&meta, &ts);
        let c = inspect_dense_gguf(&b, limits(), 4).unwrap();
        let mut slots = [empty_layer()];
        let m = prepare_dense_gguf_into(&c, &mut slots, budget()).unwrap();
        let mut out = [7.; 3];
        GgmlCpuKernels
            .gemv_into(m.embedding, &input, &mut out)
            .unwrap();
        assert_eq!(
            [out[0].to_bits(), out[1].to_bits()],
            expected,
            "dtype={dtype}"
        );
        assert_eq!(out[2], 7.);
        let mut row = [9.; 257];
        assert_eq!(
            GgmlCpuKernels
                .decode_row_into(m.embedding, 1, &mut row)
                .unwrap(),
            256
        );
        assert_eq!(row[256], 9.);
    }
}

#[test]
fn callback_refusals_preserve_outputs_and_q6_alignment_is_checked() {
    let (meta, ts) = base(4, 1);
    let b = image(&meta, &ts);
    let c = inspect_dense_gguf(&b, limits(), 4).unwrap();
    let mut slots = [empty_layer()];
    let m = prepare_dense_gguf_into(&c, &mut slots, budget()).unwrap();
    let mut out = [71.; 5];
    for (width, rows) in [(3, 2), (4, 3)] {
        let mut forged = m.embedding;
        forged.input_width = width;
        forged.output_rows = rows;
        assert_eq!(
            GgmlCpuKernels.decode_row_into(forged, 0, &mut out),
            Err(GgufCpuError::InvalidDimensions)
        );
        assert_eq!(out, [71.; 5]);
    }
    assert_eq!(
        GgmlCpuKernels.decode_row_into(m.embedding, 2, &mut out),
        Err(GgufCpuError::Capacity)
    );
    assert_eq!(
        GgmlCpuKernels.decode_row_into(m.embedding, 0, &mut out[..3]),
        Err(GgufCpuError::Capacity)
    );
    assert_eq!(
        GgmlCpuKernels.gemv_into(m.embedding, &[0.; 3], &mut out),
        Err(GgufCpuError::Capacity)
    );
    assert_eq!(
        GgmlCpuKernels.gemv_into(m.embedding, &[0.; 4], &mut out[..1]),
        Err(GgufCpuError::Capacity)
    );
    assert_eq!(out, [71.; 5]);
    let (meta, mut ts) = base(256, 1);
    ts[0].dtype = 14;
    ts[0].blob = codec_bytes(14);
    ts[0].offset = Some(1);
    refusal(&meta, &ts, GgufCpuError::Alignment);
}

#[test]
fn successful_and_refused_hot_operations_allocate_nothing() {
    assert!(
        measured(|| {
            std::hint::black_box(vec![3u8; 128]);
        }) > 0
    );
    let (meta, ts) = base(4, 2);
    let b = image(&meta, &ts);
    let mut slots = [empty_layer(); 3];
    assert_eq!(
        measured(|| {
            let c = inspect_dense_gguf(&b, limits(), 4).unwrap();
            assert!(inspect_dense_gguf(&b[..8], limits(), 4).is_err());
            assert!(prepare_dense_gguf_into(
                &c,
                &mut slots,
                BindingLimits {
                    byte_budget: 0,
                    max_tensor_visits: 0
                }
            )
            .is_err());
            let m = prepare_dense_gguf_into(&c, &mut slots, budget()).unwrap();
            let mut out = [0.; 5];
            GgmlCpuKernels
                .decode_row_into(m.embedding, 1, &mut out)
                .unwrap();
            GgmlCpuKernels
                .gemv_into(m.embedding, &[1.; 4], &mut out)
                .unwrap();
            assert!(GgmlCpuKernels
                .decode_row_into(m.embedding, 2, &mut out)
                .is_err());
        }),
        0
    );
}
