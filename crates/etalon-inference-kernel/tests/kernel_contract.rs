//! Independent frozen-source checks. Numerical fingerprints originate from the
//! original Qualia kernel compiled separately; this is same-target parity, not
//! cross-target libm equivalence or CUDA execution evidence.
use etalon_inference_kernel::{operators::*, paged_attention::*};
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
fn desc(kind: OperatorKind, inn: u32, out: u32) -> OperatorDescriptor {
    OperatorDescriptor {
        kind,
        in_features: inn,
        out_features: out,
        batch_hint: 0,
        tile_elems: if kind == OperatorKind::DenseF32 {
            inn
        } else {
            256
        },
        scale_layout: if kind == OperatorKind::DenseF32 {
            ScaleLayout::None
        } else {
            ScaleLayout::GgmlQ4K
        },
        accum: AccumKind::F32,
        max_workspace_bytes: 0,
        representation_digest: 0,
    }
}
fn fingerprint(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .fold(0xcbf29ce484222325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        })
}
fn check_fingerprint(name: &str, values: &[f32], expected: u64) {
    assert_eq!(fingerprint(values), expected, "{name}");
}
fn raw_block(seed: u8) -> [u8; 144] {
    let mut b = [0; 144];
    b[..2].copy_from_slice(&0x3555u16.to_le_bytes());
    b[2..4].copy_from_slice(&0x2e66u16.to_le_bytes());
    for (i, v) in b[4..].iter_mut().enumerate() {
        *v = (i as u8).wrapping_mul(37).wrapping_add(seed);
    }
    b
}
#[test]
fn public_constants_and_error_tokens_are_frozen() {
    assert_eq!(
        kernels::PAGED_GQA_SEGMENTED_SRC
            .bytes()
            .fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b))
                .wrapping_mul(0x100000001b3)),
        0x62979933eae2f4c3
    );
    assert_eq!(
        kernels::PAGED_GQA_TILED_SRC
            .bytes()
            .fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b))
                .wrapping_mul(0x100000001b3)),
        0x72d48679560f2ee
    );
    assert_eq!(
        [
            OperatorKind::DenseF32 as u8,
            OperatorKind::Q4KBitPlane as u8,
            AccumKind::F32 as u8,
            ScaleLayout::None as u8,
            ScaleLayout::GgmlQ4K as u8
        ],
        [1, 2, 1, 0, 1]
    );
    assert_eq!(
        (
            Q4K_SUPERBLOCK_ELEMS,
            Q4K_SUPERBLOCK_BYTES,
            FLOATS_PER_SUBBLOCK_LUT
        ),
        (256, 144, 513)
    );
    let errors = [
        OperatorError::UnsupportedShape,
        OperatorError::WorkspaceTooSmall,
        OperatorError::InvalidPayload,
        OperatorError::IndexOutOfRange,
        OperatorError::MalformedDescriptor,
        OperatorError::UnsupportedKind,
    ];
    assert_eq!(
        errors.map(OperatorError::token),
        [
            "unsupported_shape",
            "workspace_too_small",
            "invalid_payload",
            "index_out_of_range",
            "malformed_descriptor",
            "unsupported_kind"
        ]
    );
    assert_eq!((DEFAULT_BLOCK_SIZE, INVALID_BLOCK), (16, u32::MAX));
    assert_eq!(
        [0, 511, 512, 2047, 2048, u32::MAX].map(kernels::segments_for_position),
        [1, 1, 4, 4, 8, 8]
    );
    assert_eq!(kernels::PAGED_GQA_TILED_ENTRY, "paged_gqa_tiled_online");
    assert_eq!(
        kernels::PAGED_GQA_SEGMENTED_PARTIAL_ENTRY,
        "paged_gqa_segmented_partial"
    );
    assert_eq!(
        kernels::PAGED_GQA_SEGMENTED_MERGE_ENTRY,
        "paged_gqa_segmented_merge"
    );
    assert_eq!(kernels::PAGED_GQA_TILED_SRC.len(), 4134);
    assert_eq!(kernels::PAGED_GQA_SEGMENTED_SRC.len(), 6622);
}
#[test]
fn dense_known_answer_order_and_refusal_mutation_are_preserved() {
    let weights = [1f32, 2., 3., 4., 5., 6.];
    let raw: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    let payloads = [PayloadView { bytes: &raw }];
    let d = desc(OperatorKind::DenseF32, 3, 2);
    let view = validate_operator(&d, &payloads).unwrap();
    let mut out = [99.; 4];
    apply_into(
        &view,
        MatrixView {
            data: &[1., 0., 1.],
            rows: 1,
            cols: 3,
        },
        MatrixViewMut {
            data: &mut out,
            rows: 2,
            cols: 1,
        },
        OperatorWorkspace {
            numeric: &mut [],
            bytes: &mut [],
        },
    )
    .unwrap();
    assert_eq!(out, [4., 10., 99., 99.]);
    for bad in [
        OperatorDescriptor { tile_elems: 2, ..d },
        OperatorDescriptor {
            scale_layout: ScaleLayout::GgmlQ4K,
            ..d
        },
    ] {
        assert_eq!(
            validate_operator(&bad, &payloads).unwrap_err(),
            OperatorError::MalformedDescriptor
        );
    }
    assert_eq!(
        validate_operator(
            &OperatorDescriptor {
                in_features: 0,
                ..d
            },
            &payloads
        )
        .unwrap_err(),
        OperatorError::UnsupportedShape
    );
    for n in [0, 23] {
        assert_eq!(
            validate_operator(&d, &[PayloadView { bytes: &raw[..n] }]).unwrap_err(),
            OperatorError::InvalidPayload
        );
    }
    for (rows, cols, data) in [(1, 2, &[1., 2.][..]), (2, 2, &[1., 2., 3., 4.][..])] {
        let mut out = [77.; 4];
        assert_eq!(
            apply_into(
                &view,
                MatrixView { data, rows, cols },
                MatrixViewMut {
                    data: &mut out,
                    rows: 2,
                    cols: 1
                },
                OperatorWorkspace {
                    numeric: &mut [],
                    bytes: &mut []
                }
            ),
            Err(OperatorError::UnsupportedShape)
        );
        assert_eq!(out, [77.; 4]);
    }
    // Exact left-to-right cancellation: reassociation would produce1 instead of0.
    let raw: Vec<u8> = [16777216f32, 1., -16777216.]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ps = [PayloadView { bytes: &raw }];
    let view = validate_operator(&desc(OperatorKind::DenseF32, 3, 1), &ps).unwrap();
    let mut out = [9.];
    apply_into(
        &view,
        MatrixView {
            data: &[1.; 3],
            rows: 3,
            cols: 1,
        },
        MatrixViewMut {
            data: &mut out,
            rows: 1,
            cols: 1,
        },
        OperatorWorkspace {
            numeric: &mut [],
            bytes: &mut [],
        },
    )
    .unwrap();
    assert_eq!(out[0].to_bits(), 0);
}
#[test]
fn packed_q4k_reconstruction_lookup_and_tails_match_old_bits() {
    let mut raw = [0u8; 864];
    for (i, b) in raw.chunks_exact_mut(144).enumerate() {
        b.copy_from_slice(&raw_block(i as u8 * 19));
    }
    let mut decoded = [-99.; 520];
    assert_eq!(reconstruct_q4k_into(&raw, 513, &mut decoded), Ok(513));
    assert!(decoded[513..].iter().all(|v| *v == -99.));
    check_fingerprint("q4k_reconstruction", &decoded, 0x84da4fbeaf0fcdc2);
    let mut x = [0.; 512];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as i32 % 29 - 14) as f32 * 0.03125;
    }
    let d = desc(OperatorKind::Q4KBitPlane, 512, 3);
    let ps = [PayloadView { bytes: &raw }];
    let op = validate_operator(&d, &ps).unwrap();
    let mut ws = [-77.; 16 * 513 + 5];
    let mut out = [-99.; 5];
    apply_into(
        &op,
        MatrixView {
            data: &x,
            rows: 1,
            cols: 512,
        },
        MatrixViewMut {
            data: &mut out,
            rows: 3,
            cols: 1,
        },
        OperatorWorkspace {
            numeric: &mut ws,
            bytes: &mut [],
        },
    )
    .unwrap();
    check_fingerprint("q4k_lookup", &out, 0xb998f4f7fc8a59b8);
    check_fingerprint("q4k_workspace", &ws, 0xee1f747043c4095f);
    assert_eq!(&out[3..], &[-99.; 2]);
    assert_eq!(&ws[16 * 513..], &[-77.; 5]);
    let mut out = [33.; 3];
    assert_eq!(
        apply_into(
            &op,
            MatrixView {
                data: &x,
                rows: 1,
                cols: 512
            },
            MatrixViewMut {
                data: &mut out,
                rows: 3,
                cols: 1
            },
            OperatorWorkspace {
                numeric: &mut ws[..16 * 513 - 1],
                bytes: &mut []
            }
        ),
        Err(OperatorError::WorkspaceTooSmall)
    );
    assert_eq!(out, [33.; 3]);
    let mut out = [33.; 257];
    assert_eq!(
        reconstruct_q4k_into(&raw[..143], 256, &mut out),
        Err(OperatorError::InvalidPayload)
    );
    assert_eq!(out, [33.; 257]);
    assert_eq!(reconstruct_q4k_into(&[], 0, &mut out), Ok(0));
    // Validator permits a partial block but apply requires whole256-element rows.
    let op = validate_operator(&desc(OperatorKind::Q4KBitPlane, 32, 1), &ps).unwrap();
    assert_eq!(
        apply_into(
            &op,
            MatrixView {
                data: &x[..32],
                rows: 1,
                cols: 32
            },
            MatrixViewMut {
                data: &mut out,
                rows: 1,
                cols: 1
            },
            OperatorWorkspace {
                numeric: &mut ws,
                bytes: &mut []
            }
        ),
        Err(OperatorError::UnsupportedShape)
    );
}
#[test]
fn paged_grouped_attention_permutation_partial_failures_and_nonfinite_policy() {
    let c = PagedKvConfig {
        block_size: 2,
        n_layer: 1,
        n_kv_head: 2,
        head_dim: 3,
        max_context: 4,
        physical_blocks: 2,
    };
    let mut arena = [0.; 48];
    for (i, v) in arena.iter_mut().enumerate() {
        *v = (i as i32 % 17 - 8) as f32 * 0.125;
    }
    let q = [
        0.5, -0.25, 0.75, 1., 0., -0.5, -0.75, 0.25, 0.5, 0.25, 0.125, -0.25,
    ];
    let mut out = [-99.; 15];
    paged_gqa_attention_into(&q, &arena, &[1, 0], &c, 2, 4, &mut out).unwrap();
    check_fingerprint("paged_attention", &out, 0x1bf01949e9ed9bf9);
    assert_eq!(&out[12..], &[-99.; 3]);
    let c = PagedKvConfig {
        block_size: 1,
        n_layer: 1,
        n_kv_head: 1,
        head_dim: 1,
        max_context: 2,
        physical_blocks: 2,
    };
    let mut out = [99.; 3];
    assert_eq!(
        paged_gqa_attention_into(
            &[0., 0.],
            &[0., 7., 0., 9.],
            &[0, INVALID_BLOCK],
            &c,
            1,
            2,
            &mut out
        ),
        Err(AttentionError::MissingPage)
    );
    assert_eq!(out, [7., 99., 99.]);
    let mut out = [99.; 3];
    paged_gqa_attention_into(&[0.], &[0., 7., 0., 9.], &[0, 1], &c, 1, 1, &mut out).unwrap();
    assert_eq!(out, [8., 99., 99.]);
    let mut out = [99.; 3];
    assert_eq!(
        paged_gqa_attention_into(&[0.], &[0., 7., 0., 9.], &[0, 1], &c, 2, 1, &mut out),
        Err(AttentionError::InvalidShape)
    );
    assert_eq!(out, [99.; 3]);
    for q in [f32::NAN, f32::INFINITY] {
        let mut out = [99.; 2];
        assert_eq!(
            paged_gqa_attention_into(&[q], &[0., 7., 0., 9.], &[0, 1], &c, 0, 1, &mut out),
            Ok(())
        );
        assert!(out[0].is_nan());
        assert_eq!(out[1], 99.);
    }
    // Frozen known limitation: saturation is accepted, not a safe-arena assertion.
    let c = PagedKvConfig {
        block_size: 16,
        n_layer: 1,
        n_kv_head: u32::MAX,
        head_dim: 2,
        max_context: 1,
        physical_blocks: 1,
    };
    assert!(c.is_valid());
    assert_eq!(c.slot_kv_elems(), u32::MAX);
}
#[test]
fn hot_success_and_refusal_calls_are_positive_controlled_zero_alloc() {
    COUNT.with(|v| v.set(0));
    ACTIVE.with(|v| v.set(true));
    let control = std::hint::black_box(Box::new([3u8; 71]));
    std::hint::black_box(&control);
    drop(control);
    ACTIVE.with(|v| v.set(false));
    assert!(COUNT.with(Cell::get) > 0);
    let raw = [0u8; 144];
    let ps = [PayloadView { bytes: &raw }];
    let qop = validate_operator(&desc(OperatorKind::Q4KBitPlane, 256, 1), &ps).unwrap();
    let x = [1.; 256];
    let mut y = [99.; 4];
    let mut ws = [0.; 8 * 513];
    let mut decoded = [99.; 260];
    let c = PagedKvConfig {
        block_size: 1,
        n_layer: 1,
        n_kv_head: 1,
        head_dim: 1,
        max_context: 1,
        physical_blocks: 1,
    };
    let dense_raw = [0u8; 4];
    let dp = [PayloadView { bytes: &dense_raw }];
    let dop = validate_operator(&desc(OperatorKind::DenseF32, 1, 1), &dp).unwrap();
    COUNT.with(|v| v.set(0));
    ACTIVE.with(|v| v.set(true));
    let ds = apply_into(
        &dop,
        MatrixView {
            data: &[1.],
            rows: 1,
            cols: 1,
        },
        MatrixViewMut {
            data: &mut y,
            rows: 1,
            cols: 1,
        },
        OperatorWorkspace {
            numeric: &mut [],
            bytes: &mut [],
        },
    );
    let dr = apply_into(
        &dop,
        MatrixView {
            data: &[],
            rows: 1,
            cols: 1,
        },
        MatrixViewMut {
            data: &mut y,
            rows: 1,
            cols: 1,
        },
        OperatorWorkspace {
            numeric: &mut [],
            bytes: &mut [],
        },
    );
    let qr = apply_into(
        &qop,
        MatrixView {
            data: &x,
            rows: 256,
            cols: 1,
        },
        MatrixViewMut {
            data: &mut y,
            rows: 1,
            cols: 1,
        },
        OperatorWorkspace {
            numeric: &mut [],
            bytes: &mut [],
        },
    );
    let a = apply_into(
        &qop,
        MatrixView {
            data: &x,
            rows: 256,
            cols: 1,
        },
        MatrixViewMut {
            data: &mut y,
            rows: 1,
            cols: 1,
        },
        OperatorWorkspace {
            numeric: &mut ws,
            bytes: &mut [],
        },
    );
    let b = reconstruct_q4k_into(&raw, 256, &mut decoded);
    let e = reconstruct_q4k_into(&raw[..1], 256, &mut decoded);
    let d = paged_gqa_attention_into(&[0.], &[0., 7.], &[0], &c, 0, 1, &mut y);
    let f = paged_gqa_attention_into(&[0.], &[0., 7.], &[INVALID_BLOCK], &c, 0, 1, &mut y);
    ACTIVE.with(|v| v.set(false));
    let count = COUNT.with(Cell::get);
    assert_eq!(ds, Ok(()));
    assert_eq!(dr, Err(OperatorError::UnsupportedShape));
    assert_eq!(qr, Err(OperatorError::WorkspaceTooSmall));
    assert_eq!(a, Ok(()));
    assert_eq!(b, Ok(256));
    assert_eq!(e, Err(OperatorError::InvalidPayload));
    assert_eq!(d, Ok(()));
    assert_eq!(f, Err(AttentionError::MissingPage));
    assert_eq!(count, 0);
}
