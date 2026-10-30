//! Comprehensive unit tests for `etalon-wgsl-forge`.

use crate::{
    certify_kernel, compile_hlsl_to_spirv, compile_hlsl_to_spirv_cached, compute_cache_key,
    dxc_cache_key, emit_cuda_c, emit_cuda_kernel, emit_hlsl, emit_hlsl_wave_kernel, emit_kernel,
    emit_msl, emit_msl_kernel, emit_shader, emit_wgsl, matmul_tc_f16_16x16_wgsl,
    matmul_tc_i8_16x16x32_wgsl, matmul_tc_wgsl, matmul_tc_wgsl_tiled, should_use_coopmat,
    validate_wgsl, AdapterConstraints, BuiltinKernel, CoopMatShape, Op, Schedule, ScheduleSpace,
    TargetBackend, TileConfig, WGPU_COOPMAT_BLOCKED,
};

#[test]
fn all_builtin_wgsl_kernels_emit_and_validate_with_naga() {
    for builtin in BuiltinKernel::ALL {
        if builtin == BuiltinKernel::RayProbe {
            let spec = builtin.spec();
            spec.validate().expect("RayProbe spec must validate");
            let shader = emit_wgsl(&spec, Schedule::default()).expect("RayProbe WGSL emit");
            assert!(shader.source.contains("enable etalon_ray_query;"));
            continue;
        }
        let spec = builtin.spec();
        spec.validate().expect("builtin spec must validate");
        let sched = Schedule::default();
        let shader = emit_wgsl(&spec, sched).expect("WGSL emission must succeed");
        assert!(
            shader.source.contains("@compute"),
            "WGSL for {} must contain @compute",
            spec.id
        );
        let report =
            validate_wgsl(&shader.source).expect("Naga validation must succeed for builtin WGSL");
        assert!(report.naga_validated);
        assert_eq!(report.entry_points, vec![spec.entry_point]);
    }
}

#[test]
fn emit_kernel_high_level_ops_wgsl_validates_with_naga() {
    let sched = Schedule {
        tile: TileConfig::new(1, 64, 64),
        ..Schedule::default()
    };
    let ops = [
        Op::MatMul {
            tc: false,
            transpose_b: false,
        },
        Op::MatMul {
            tc: true,
            transpose_b: false,
        },
        Op::Gemv { tc: false },
        Op::ElementwiseAdd,
        Op::ElementwiseMul,
        Op::Silu,
        Op::RmsNorm { eps: 1e-5 },
        Op::RoPE { theta: 10000.0 },
        Op::PrefixScan,
        Op::RadixHistogram { shift: 8 },
        Op::RadixScatter { shift: 8 },
    ];

    for op in &ops {
        let src = emit_kernel(op, &sched, TargetBackend::Wgsl).expect("WGSL kernel emit");
        assert!(
            !src.contains("// Stub"),
            "WGSL kernel must not be a stub: {op:?}"
        );
        certify_kernel(&src).expect("Naga certification must pass");
    }
}

#[test]
fn coopmat_m1_decode_bypass_and_tile_shapes() {
    const {
        assert!(
            !WGPU_COOPMAT_BLOCKED,
            "Runbook 01 Bug 1.5: WGPU_COOPMAT_BLOCKED must be false"
        )
    };

    // M <= 4 decode must bypass WMMA / CooperativeMatrix and use GEMV
    for m in 1..=4 {
        let shape = CoopMatShape::f16_16x16x16(m, 4096, 4096);
        assert!(
            !should_use_coopmat(&shape),
            "M={m} decode must NOT use coopmat (87.5%+ waste on 16x16 tile)"
        );
    }

    // M >= 8 with divisible dimensions and hardware support should use coopmat
    let batch_f16 = CoopMatShape::f16_16x16x16(16, 64, 64);
    assert!(should_use_coopmat(&batch_f16));

    let batch_i8 = CoopMatShape::i8_16x16x32(16, 64, 64);
    assert!(should_use_coopmat(&batch_i8));

    let batch_f32 = CoopMatShape::f32_8x8x8(8, 64, 64);
    assert!(should_use_coopmat(&batch_f32));

    // Verify all 4 cooperative matrix WGSL generators emit real, non-stub WGSL
    let tc8 = matmul_tc_wgsl();
    assert!(tc8.contains("enable etalon_cooperative_matrix;"));
    assert!(tc8.contains("coop_mat8x8<f32"));

    let tc8_tiled = matmul_tc_wgsl_tiled();
    assert!(tc8_tiled.contains("enable etalon_cooperative_matrix;"));
    assert!(tc8_tiled.contains("coopMultiplyAdd"));

    let tc16_f16 = matmul_tc_f16_16x16_wgsl();
    assert!(tc16_f16.contains("enable f16;"));
    assert!(tc16_f16.contains("coop_mat16x16<f16"));

    let tc16_i8 = matmul_tc_i8_16x16x32_wgsl();
    assert!(tc16_i8.contains("enable etalon_cooperative_matrix;"));
    assert!(tc16_i8.contains("coop_mat16x16<i32"));
}

#[test]
fn cuda_c_emits_real_kernels_and_routes_m1_to_dp4a_gemv() {
    // M=1 decode routes to Q4_K SoA __dp4a GEMV, NOT WMMA
    let decode_sched = Schedule {
        tile: TileConfig::new(1, 4096, 4096),
        ..Schedule::default()
    };
    let decode_op = Op::MatMul {
        tc: true,
        transpose_b: false,
    };
    let decode_cuda = emit_cuda_kernel(&decode_op, &decode_sched).expect("CUDA M=1 emit");
    assert!(decode_cuda.contains("extern \"C\" __global__"));
    assert!(decode_cuda.contains("__dp4a"));
    assert!(!decode_cuda.contains("wmma::mma_sync"));

    // M=32 prefill routes to WMMA Tensor Core GEMM
    let prefill_sched = Schedule {
        tile: TileConfig::new(32, 4096, 4096),
        ..Schedule::default()
    };
    let prefill_cuda = emit_cuda_kernel(&decode_op, &prefill_sched).expect("CUDA M=32 emit");
    assert!(prefill_cuda.contains("wmma::mma_sync"));

    // BuiltinKernel CUDA-C emission
    for builtin in [
        BuiltinKernel::AffineF32,
        BuiltinKernel::TopK,
        BuiltinKernel::FusedFfn,
        BuiltinKernel::P64Project,
        BuiltinKernel::TernaryGemv,
        BuiltinKernel::Gemm,
        BuiltinKernel::Gemv,
        BuiltinKernel::Fft,
    ] {
        let shader =
            emit_cuda_c(&builtin.spec(), Schedule::default()).expect("CUDA-C builtin emit");
        assert!(
            shader.source.contains("extern \"C\" __global__"),
            "CUDA-C for {} must contain extern \"C\" __global__",
            builtin.name()
        );
    }
}

#[test]
fn msl_emits_real_metal_kernels_and_simdgroup_matrix() {
    let sched = Schedule {
        tile: TileConfig::new(32, 64, 64),
        ..Schedule::default()
    };

    for builtin in [
        BuiltinKernel::AffineF32,
        BuiltinKernel::TopK,
        BuiltinKernel::FusedFfn,
        BuiltinKernel::P64Project,
        BuiltinKernel::TernaryGemv,
        BuiltinKernel::Gemm,
        BuiltinKernel::Gemv,
        BuiltinKernel::Fft,
    ] {
        let shader = emit_msl(&builtin.spec(), sched).expect("MSL builtin emit");
        assert!(
            shader.source.contains("#include <metal_stdlib>"),
            "MSL for {} must include metal_stdlib",
            builtin.name()
        );
        assert!(
            shader.source.contains("kernel void"),
            "MSL for {} must contain kernel void",
            builtin.name()
        );
    }

    // Cooperative matrix MSL path emits simdgroup_matrix<float, 8, 8>
    let gemm_op = Op::MatMul {
        tc: true,
        transpose_b: false,
    };
    let msl_coop = emit_msl_kernel(&gemm_op, &sched).expect("MSL coopmat emit");
    assert!(msl_coop.contains("simdgroup_matrix<float, 8, 8>"));
    assert!(msl_coop.contains("simdgroup_multiply_accumulate"));
}

#[test]
fn hlsl_wave_and_wavematrix_emit_real_sm6_intrinsics() {
    let sched = Schedule {
        tile: TileConfig::new(32, 64, 64),
        ..Schedule::default()
    };

    // Wave GEMV emits WaveActiveSum
    let gemv_shader = emit_hlsl(&BuiltinKernel::Gemv.spec(), sched).expect("HLSL gemv emit");
    assert!(gemv_shader.source.contains("WaveActiveSum"));
    assert!(gemv_shader.source.contains("WaveGetLaneCount"));

    // WaveMatrix GEMM/GEMV emits WaveMatrixMultiply when tc is true and M > 4
    let gemm_op = Op::MatMul {
        tc: true,
        transpose_b: false,
    };
    let hlsl_wm = emit_hlsl_wave_kernel(&gemm_op, &sched).expect("HLSL WaveMatrix emit");
    assert!(hlsl_wm.contains("WaveMatrixA<float16_t>"));
    assert!(hlsl_wm.contains("WaveMatrixB<float16_t>"));
    assert!(hlsl_wm.contains("WaveMatrixC<float>"));
    assert!(hlsl_wm.contains("WaveMatrixMultiply"));

    // M=1 decode bypasses WaveMatrix and uses WaveActiveSum
    let decode_sched = Schedule {
        tile: TileConfig::new(1, 64, 64),
        ..Schedule::default()
    };
    let hlsl_decode = emit_hlsl_wave_kernel(&gemm_op, &decode_sched).expect("HLSL M=1 decode emit");
    assert!(hlsl_decode.contains("WaveActiveSum"));
    assert!(!hlsl_decode.contains("WaveMatrixMultiply"));
}

#[test]
fn dxc_and_forge_cache_keys_are_length_delimited_and_collision_free() {
    // Runbook 01 Bug 1.4: ("ab", "c") vs ("a", "bc") must produce distinct keys
    let k1 = dxc_cache_key("ab", "c");
    let k2 = dxc_cache_key("a", "bc");
    assert_ne!(
        k1, k2,
        "DXC cache key must be length-delimited to prevent boundary collisions"
    );

    let fk1 = compute_cache_key("ab", "c", 42);
    let fk2 = compute_cache_key("a", "bc", 42);
    assert_ne!(
        fk1, fk2,
        "Forge cache key must be length-delimited to prevent boundary collisions"
    );

    let _ = compile_hlsl_to_spirv_cached("invalid hlsl", "main", compile_hlsl_to_spirv);
}

#[test]
fn schedule_space_and_emit_shader_round_trip() {
    let spec = BuiltinKernel::AffineF32.spec();
    let space = ScheduleSpace::default();
    let candidates = space.candidates(&spec, &AdapterConstraints::portable());
    assert!(!candidates.is_empty());

    let shader = emit_shader(&spec, candidates[0], TargetBackend::Wgsl).expect("emit_shader");
    assert_eq!(shader.kernel_id, "affine-f32");
}
