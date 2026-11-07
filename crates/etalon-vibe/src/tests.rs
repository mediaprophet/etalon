//! Comprehensive unit and conformance tests for all 8 Etalon VibeScript providers.

use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, OntologicalContract, PrimitiveKind, StableFuncId,
    VibeTrap,
};

use crate::*;

fn encode_f32_slice<const N: usize>(vals: [f32; N], buf: &mut [u8]) -> BorrowedSliceView<'_> {
    for (i, v) in vals.iter().enumerate() {
        buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    BorrowedSliceView::from_raw_bytes(&buf[..N * 4], PrimitiveKind::F32, N).unwrap()
}

fn encode_f64_slice<const N: usize>(vals: [f64; N], buf: &mut [u8]) -> BorrowedSliceView<'_> {
    for (i, v) in vals.iter().enumerate() {
        buf[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
    }
    BorrowedSliceView::from_raw_bytes(&buf[..N * 8], PrimitiveKind::F64, N).unwrap()
}

fn encode_u32_slice<const N: usize>(vals: [u32; N], buf: &mut [u8]) -> BorrowedSliceView<'_> {
    for (i, v) in vals.iter().enumerate() {
        buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    BorrowedSliceView::from_raw_bytes(&buf[..N * 4], PrimitiveKind::U32, N).unwrap()
}

fn encode_u64_slice<const N: usize>(vals: [u64; N], buf: &mut [u8]) -> BorrowedSliceView<'_> {
    for (i, v) in vals.iter().enumerate() {
        buf[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
    }
    BorrowedSliceView::from_raw_bytes(&buf[..N * 8], PrimitiveKind::U64, N).unwrap()
}

#[test]
fn all_eight_providers_have_valid_descriptors_contracts_and_5field_ontology_quins() {
    assert_eq!(ETALON_VIBE_PROVIDERS.len(), 8);
    let mut total_funcs = 0usize;

    for (idx, &provider) in ETALON_VIBE_PROVIDERS.iter().enumerate() {
        let desc = provider.descriptor();
        assert!(desc.canonical_iri.is_valid());
        assert_eq!(desc.short_alias, ETALON_EXTENSION_ALIASES[idx]);
        assert_eq!(desc.canonical_iri.iri, ETALON_EXTENSION_IRIS[idx]);
        assert!(desc.versions.is_engine_compatible());
        assert!(provider_by_alias(desc.short_alias).is_some());
        assert!(provider_by_iri(desc.canonical_iri.iri).is_some());

        let quins = provider.static_ontology_nquins();
        assert_eq!(quins.len(), 3);
        assert!(quins.len() <= desc.max_ontology_quins as usize);
        for q in quins {
            assert!(
                verify_nquin48_5field(q),
                "provider {} ontology quin failed 5-field XOR parity",
                desc.short_alias
            );
        }

        for sub in desc.sub_namespaces {
            assert!(!sub.functions.is_empty());
            for func in sub.functions {
                total_funcs += 1;
                assert!(func.canonical_iri.is_valid());
                assert_eq!(func.stable_id.sub_namespace_id, sub.sub_id);
                assert_eq!(func.envelope.max_scratch_bytes, 0);
                assert!(func.contract.validate_spans(quins.len()));

                let expected_digest = OntologicalContract::canonical_digest(
                    func.canonical_iri.fnv1a_hash,
                    func.stable_id,
                    func.contract.schema_hash,
                    func.effects,
                    func.determinism,
                    func.reversibility,
                    func.envelope,
                    func.contract.precondition_quin_span,
                    func.contract.postcondition_quin_span,
                    func.contract.invariant_quin_span,
                );
                assert_eq!(func.contract.contract_digest, expected_digest);
            }
        }
    }
    assert_eq!(total_funcs, 30);
}

#[test]
fn gpu_compute_provider_matches_oracles_and_traps_on_invalid_inputs() {
    // (0, 0) affine_f32
    let mut b0 = [0u8; 16];
    let mut b1 = [0u8; 8];
    let in_vec = encode_f32_slice([1.0, -2.0, 3.5, 4.0], &mut b0);
    let sb = encode_f32_slice([2.0, 0.5], &mut b1);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    let ticks = GPU_PROVIDER
        .call(StableFuncId::new(0, 0), &[in_vec, sb], &mut out)
        .unwrap();
    assert!(ticks > 0);
    let out_view =
        BorrowedSliceView::from_raw_bytes(out.written_byte_slice(), PrimitiveKind::F32, 4).unwrap();
    let expected = oracle_affine_f32([1.0, -2.0, 3.5, 4.0], 2.0, 0.5).unwrap();
    for (i, &exp) in expected.iter().enumerate() {
        assert!((out_view.read_f32_le(i).unwrap() - exp).abs() < 1e-6);
    }

    // (0, 1) gemv4_f32
    let mut mat_b = [0u8; 64];
    let mut vec_b = [0u8; 16];
    let mat = [
        1.0, 0.0, 2.0, -1.0, // row 0
        0.0, 3.0, 1.0, 0.0, // row 1
        2.0, 1.0, 0.0, 4.0, // row 2
        -1.0, 0.0, 0.0, 2.0, // row 3
    ];
    let v4 = [2.0, -1.0, 3.0, 0.5];
    let mv = encode_f32_slice(mat, &mut mat_b);
    let vv = encode_f32_slice(v4, &mut vec_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    GPU_PROVIDER
        .call(StableFuncId::new(0, 1), &[mv, vv], &mut out)
        .unwrap();
    let out_view =
        BorrowedSliceView::from_raw_bytes(out.written_byte_slice(), PrimitiveKind::F32, 4).unwrap();
    assert!((out_view.read_f32_le(0).unwrap() - 7.5).abs() < 1e-6);
    assert!((out_view.read_f32_le(1).unwrap() - 0.0).abs() < 1e-6);
    assert!((out_view.read_f32_le(2).unwrap() - 5.0).abs() < 1e-6);
    assert!((out_view.read_f32_le(3).unwrap() - (-1.0)).abs() < 1e-6);

    // (0, 2) topk2_f32
    let mut sc_b = [0u8; 24];
    let sc = encode_f32_slice([0.4, 9.2, -3.1, 4.8, 9.5, 1.2], &mut sc_b);
    let mut out_raw = [0u8; 8];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    GPU_PROVIDER
        .call(StableFuncId::new(0, 2), &[sc], &mut out)
        .unwrap();
    let out_view =
        BorrowedSliceView::from_raw_bytes(out.written_byte_slice(), PrimitiveKind::F32, 2).unwrap();
    assert!((out_view.read_f32_le(0).unwrap() - 9.5).abs() < 1e-6);
    assert!((out_view.read_f32_le(1).unwrap() - 9.2).abs() < 1e-6);

    // (0, 3) rmsnorm_f32 & (0, 4) rope_f32
    let mut x_b = [0u8; 16];
    let mut w_b = [0u8; 16];
    let mut e_b = [0u8; 4];
    let xv = encode_f32_slice([2.0, 2.0, 2.0, 2.0], &mut x_b);
    let wv = encode_f32_slice([1.0, 2.0, 0.5, 1.5], &mut w_b);
    let ev = encode_f32_slice([1e-6], &mut e_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    GPU_PROVIDER
        .call(StableFuncId::new(0, 3), &[xv, wv, ev], &mut out)
        .unwrap();
    let out_view =
        BorrowedSliceView::from_raw_bytes(out.written_byte_slice(), PrimitiveKind::F32, 4).unwrap();
    assert!((out_view.read_f32_le(0).unwrap() - 1.0).abs() < 1e-5);
    assert!((out_view.read_f32_le(1).unwrap() - 2.0).abs() < 1e-5);

    let mut tp_b = [0u8; 8];
    let tpv = encode_f32_slice([10000.0, 0.0], &mut tp_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    GPU_PROVIDER
        .call(StableFuncId::new(0, 4), &[xv, tpv], &mut out)
        .unwrap();
    let out_view =
        BorrowedSliceView::from_raw_bytes(out.written_byte_slice(), PrimitiveKind::F32, 4).unwrap();
    assert!((out_view.read_f32_le(0).unwrap() - 2.0).abs() < 1e-6);

    // NaN input traps with NumericalDomainError
    assert_eq!(
        oracle_affine_f32([f32::NAN, 0.0, 0.0, 0.0], 1.0, 0.0),
        Err(VibeTrap::NumericalDomainError)
    );
}

#[test]
fn mechanics_and_forge_providers_match_oracles() {
    // Mechanics: verlet_step (0, 0), kinetic_energy (0, 1), spring_damper_force (0, 2), aabb_volume (1, 0), excavation_prism_volume (1, 1)
    let mut p_b = [0u8; 24];
    let mut v_b = [0u8; 24];
    let mut a_b = [0u8; 24];
    let mut dt_b = [0u8; 8];
    let pv = encode_f64_slice([0.0, 10.0, 0.0], &mut p_b);
    let vv = encode_f64_slice([2.0, 0.0, -1.0], &mut v_b);
    let av = encode_f64_slice([0.0, -9.8, 0.0], &mut a_b);
    let dtv = encode_f64_slice([0.5], &mut dt_b);
    let mut out_raw = [0u8; 48];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F64).unwrap();
    MECH_PROVIDER
        .call(StableFuncId::new(0, 0), &[pv, vv, av, dtv], &mut out)
        .unwrap();
    let out_view =
        BorrowedSliceView::from_raw_bytes(out.written_byte_slice(), PrimitiveKind::F64, 6).unwrap();
    assert!((out_view.read_f64_le(0).unwrap() - 1.0).abs() < 1e-12);
    assert!((out_view.read_f64_le(1).unwrap() - 8.775).abs() < 1e-12);
    assert!((out_view.read_f64_le(4).unwrap() - (-4.9)).abs() < 1e-12);

    assert!((oracle_kinetic_energy(4.0, [3.0, 4.0, 0.0]).unwrap() - 50.0).abs() < 1e-12);
    assert_eq!(
        oracle_spring_damper_force([1.0, -2.0, 0.5], [0.5, 1.0, -1.0], 10.0, 2.0).unwrap(),
        [-11.0, 18.0, -3.0]
    );
    assert_eq!(
        oracle_aabb_volume([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]).unwrap(),
        [24.0, 52.0]
    );
    assert!((oracle_excavation_prism_volume(12.0, 1.0, 2.0, 3.0).unwrap() - 24.0).abs() < 1e-12);
    assert_eq!(
        oracle_verlet_step([0.0; 3], [0.0; 3], [0.0; 3], 0.0),
        Err(VibeTrap::NumericalDomainError)
    );

    // Forge: tile_schedule_gemm (0, 0), iq_power_reference (0, 1), coop_matrix_tile_ops (0, 2)
    let mut mnk_b = [0u8; 12];
    let mut wg_b = [0u8; 12];
    let mnk_v = encode_u32_slice([64, 128, 32], &mut mnk_b);
    let wg_v = encode_u32_slice([16, 16, 16], &mut wg_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::U32).unwrap();
    FORGE_PROVIDER
        .call(StableFuncId::new(0, 0), &[mnk_v, wg_v], &mut out)
        .unwrap();
    let out_view =
        BorrowedSliceView::from_raw_bytes(out.written_byte_slice(), PrimitiveKind::U32, 4).unwrap();
    assert_eq!(out_view.read_u32_le(0).unwrap(), 8);
    assert_eq!(out_view.read_u32_le(1).unwrap(), 4);
    assert_eq!(out_view.read_u32_le(2).unwrap(), 2);
    assert_eq!(out_view.read_u32_le(3).unwrap(), 8 * 4 * 256);

    assert!((oracle_iq_power_reference(&[3.0, 4.0, 0.0, 5.0], 2.0).unwrap() - 50.0).abs() < 1e-5);
    assert_eq!(
        oracle_coop_matrix_tile_ops([16, 16, 16]).unwrap(),
        [4096, 2048]
    );
    assert_eq!(
        oracle_tile_schedule_gemm([0, 16, 16], [16, 16, 16]),
        Err(VibeTrap::NumericalDomainError)
    );
}

#[test]
fn render_signal_sense_model_and_nquin_providers_match_oracles() {
    // Render
    let p3 = oracle_project_10d_to_3d(
        &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0],
        &[0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0],
    )
    .unwrap();
    assert!((p3[0] - 9.5).abs() < 1e-6);
    assert!((p3[1] - 4.5).abs() < 1e-6);
    assert!((p3[2] - 5.5).abs() < 1e-6);
    assert!((oracle_sdf_torus_eval([3.0, 0.0, 0.0], 2.0, 0.5).unwrap() - 0.5).abs() < 1e-6);
    assert_eq!(
        oracle_skeletal_lerp_joint([0.0, 0.0, 0.0], [10.0, 20.0, 30.0], 0.25).unwrap(),
        [2.5, 5.0, 7.5]
    );
    // Cube topology: V=8, E=12, F=6 -> chi=2, 2g=0
    assert_eq!(oracle_brep_euler_characteristic(8, 12, 6).unwrap(), [2, 0]);

    // Signal
    let rms_pk = oracle_signal_rms_peak(&[3.0, -3.0, 3.0, -3.0]).unwrap();
    assert!((rms_pk[0] - 3.0).abs() < 1e-6);
    assert!((rms_pk[1] - 3.0).abs() < 1e-6);
    assert_eq!(
        oracle_cfar_cell_average(&[2.0, 4.0, 6.0, 8.0], 3.0).unwrap(),
        [5.0, 15.0]
    );
    assert!((oracle_biquad_df1_step([1.0, 0.5, 0.0], [0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0]).unwrap() - 0.75).abs() < 1e-6);
    assert_eq!(oracle_region_mean_u8(&[10, 20, 30, 40]).unwrap(), [25.0, 30.0]);

    // Sense
    assert_eq!(
        oracle_clock_stamp_drift_ns(1_000_000, 1_000_400, 500).unwrap(),
        [400, 1]
    );
    assert!((oracle_spectrum_bin_hz(1000.0, 200.0, 0, 4).unwrap() - 925.0).abs() < 1e-9);
    assert!((oracle_rssi_log_distance_m(-60.0, -40.0, 2.0).unwrap() - 10.0).abs() < 1e-9);

    // Model
    assert_eq!(oracle_p64_align_offset(100, 64).unwrap(), [128, 28]);
    assert_eq!(
        oracle_p64_align_offset(100, 32),
        Err(VibeTrap::NumericalDomainError)
    );
    let moe = oracle_moe_top2_softmax(&[1.0, 5.0, 5.0, -2.0]).unwrap();
    assert_eq!(moe[0], 1.0);
    assert_eq!(moe[2], 2.0);
    assert!((moe[1] - 0.5).abs() < 1e-6);
    assert!((moe[3] - 0.5).abs() < 1e-6);
    assert_eq!(
        oracle_residency_admit_check(1024, 512, 2048).unwrap(),
        [1, 512]
    );

    // NQuin
    let fields5 = [0x11u64, 0x22, 0x33, 0x44, 0x5500_0000_0001];
    let p = oracle_compute_wire_parity(fields5);
    assert_eq!(p, 0x11 ^ 0x22 ^ 0x33 ^ 0x44 ^ 0x5500_0000_0001);
    let mut f5_b = [0u8; 40];
    let f5_v = encode_u64_slice(fields5, &mut f5_b);
    let mut out_raw = [0u8; 8];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::U64).unwrap();
    NQUIN_PROVIDER
        .call(StableFuncId::new(0, 0), &[f5_v], &mut out)
        .unwrap();
    let out_view =
        BorrowedSliceView::from_raw_bytes(out.written_byte_slice(), PrimitiveKind::U64, 1).unwrap();
    assert_eq!(out_view.read_u64_le(0).unwrap(), p);
    assert_eq!(
        oracle_verify_wire_words([fields5[0], fields5[1], fields5[2], fields5[3], fields5[4], p]),
        [1, p]
    );
    assert_eq!(oracle_radix_bucket_byte(0xABCD_EF12_3456_789Au64, 8).unwrap(), 0x78);
    assert_eq!(
        oracle_radix_bucket_byte(0x1234, 7),
        Err(VibeTrap::NumericalDomainError)
    );
}

#[test]
fn all_remaining_provider_call_arms_and_traps_round_trip() {
    // Render (0,0), (0,1), (0,2), (0,3)
    let mut s10_b = [0u8; 40];
    let mut w7_b = [0u8; 28];
    let s10 = encode_f32_slice([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0], &mut s10_b);
    let w7 = encode_f32_slice([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0], &mut w7_b);
    let mut out_raw = [0u8; 12];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    RENDER_PROVIDER.call(StableFuncId::new(0, 0), &[s10, w7], &mut out).unwrap();

    let mut pt_b = [0u8; 12];
    let mut r_b = [0u8; 8];
    let pt = encode_f32_slice([3.0, 0.0, 0.0], &mut pt_b);
    let rad = encode_f32_slice([2.0, 0.5], &mut r_b);
    let mut out_raw = [0u8; 4];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    RENDER_PROVIDER.call(StableFuncId::new(0, 1), &[pt, rad], &mut out).unwrap();

    let mut jb_b = [0u8; 12];
    let mut t_b = [0u8; 4];
    let jb = encode_f32_slice([6.0, 4.0, 2.0], &mut jb_b);
    let tv = encode_f32_slice([0.5], &mut t_b);
    let mut out_raw = [0u8; 12];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    RENDER_PROVIDER.call(StableFuncId::new(0, 2), &[pt, jb, tv], &mut out).unwrap();

    let mut vef_b = [0u8; 12];
    let vef = encode_u32_slice([8, 12, 6], &mut vef_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::I64).unwrap();
    RENDER_PROVIDER.call(StableFuncId::new(0, 3), &[vef], &mut out).unwrap();

    // Signal (0,0), (0,1), (0,2), (1,0)
    let mut smp_b = [0u8; 16];
    let smp = encode_f32_slice([3.0, -3.0, 3.0, -3.0], &mut smp_b);
    let mut out_raw = [0u8; 8];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    SIGNAL_PROVIDER.call(StableFuncId::new(0, 0), &[smp], &mut out).unwrap();

    let mut out_raw = [0u8; 8];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    let mut al_b = [0u8; 4];
    let al = encode_f32_slice([2.5], &mut al_b);
    let pos_cells = encode_f32_slice([2.0, 4.0, 6.0, 8.0], &mut smp_b);
    SIGNAL_PROVIDER.call(StableFuncId::new(0, 1), &[pos_cells, al], &mut out).unwrap();

    let mut c5_b = [0u8; 20];
    let c5 = encode_f32_slice([0.5, 0.5, 0.0, 0.0, 0.0], &mut c5_b);
    let mut out_raw = [0u8; 4];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    SIGNAL_PROVIDER.call(StableFuncId::new(0, 2), &[pt, rad, c5], &mut out).unwrap();

    let px_raw = [10u8, 20, 30, 40];
    let px_v = BorrowedSliceView::from_raw_bytes(&px_raw, PrimitiveKind::U8, 4).unwrap();
    let mut out_raw = [0u8; 8];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    SIGNAL_PROVIDER.call(StableFuncId::new(1, 0), &[px_v], &mut out).unwrap();

    // Sense (0,0), (0,1), (0,2)
    let mut st_b = [0u8; 16];
    let mut u_b = [0u8; 8];
    let st = encode_u64_slice([1000, 1200], &mut st_b);
    let unc = encode_u64_slice([500], &mut u_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::U64).unwrap();
    SENSE_PROVIDER.call(StableFuncId::new(0, 0), &[st, unc], &mut out).unwrap();

    let mut cb_b = [0u8; 16];
    let mut bt_b = [0u8; 8];
    let cb = encode_f64_slice([1000.0, 200.0], &mut cb_b);
    let bt = encode_u32_slice([1, 4], &mut bt_b);
    let mut out_raw = [0u8; 8];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F64).unwrap();
    SENSE_PROVIDER.call(StableFuncId::new(0, 1), &[cb, bt], &mut out).unwrap();

    let mut rssi_b = [0u8; 24];
    let rssi = encode_f64_slice([-60.0, -40.0, 2.0], &mut rssi_b);
    let mut out_raw = [0u8; 8];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F64).unwrap();
    SENSE_PROVIDER.call(StableFuncId::new(0, 2), &[rssi], &mut out).unwrap();

    // Model (0,0), (0,1), (0,2)
    let mut oa_b = [0u8; 16];
    let oa = encode_u64_slice([100, 64], &mut oa_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::U64).unwrap();
    MODEL_PROVIDER.call(StableFuncId::new(0, 0), &[oa], &mut out).unwrap();

    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::F32).unwrap();
    MODEL_PROVIDER.call(StableFuncId::new(0, 1), &[pos_cells], &mut out).unwrap();

    let mut urc_b = [0u8; 24];
    let urc = encode_u64_slice([1024, 512, 2048], &mut urc_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::U64).unwrap();
    MODEL_PROVIDER.call(StableFuncId::new(0, 2), &[urc], &mut out).unwrap();

    // NQuin (0,1), (0,2) and trap checks
    let mut w6_b = [0u8; 48];
    let w6 = encode_u64_slice([1, 2, 3, 4, 5, 1 ^ 2 ^ 3 ^ 4 ^ 5], &mut w6_b);
    let mut out_raw = [0u8; 16];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::U64).unwrap();
    NQUIN_PROVIDER.call(StableFuncId::new(0, 1), &[w6], &mut out).unwrap();

    let mut k_b = [0u8; 8];
    let mut sh_b = [0u8; 4];
    let kv = encode_u64_slice([0x1234_5678], &mut k_b);
    let shv = encode_u32_slice([8], &mut sh_b);
    let mut out_raw = [0u8; 4];
    let mut out = CallerOutputSlice::new(&mut out_raw, PrimitiveKind::U32).unwrap();
    NQUIN_PROVIDER.call(StableFuncId::new(0, 2), &[kv, shv], &mut out).unwrap();

    assert_eq!(
        NQUIN_PROVIDER.call(StableFuncId::new(9, 9), &[], &mut out),
        Err(VibeTrap::FunctionNotFound)
    );
    assert_eq!(
        NQUIN_PROVIDER.call(StableFuncId::new(0, 2), &[kv], &mut out),
        Err(VibeTrap::ArityMismatch)
    );
}
