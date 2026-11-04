#[cfg(test)]
mod migration_tests {
    use super::migration_legacy_wgsl::{emit_fft_wgsl, emit_gemm_wgsl, emit_gemv_wgsl};
    use super::*;
    use crate::numerical::{BuiltinKernel, Schedule};

    #[test]
    fn emission_is_byte_deterministic() {
        let kernel = BuiltinKernel::AffineF32.spec();
        let first = emit_wgsl(&kernel, Schedule::default()).unwrap();
        let second = emit_wgsl(&kernel, Schedule::default()).unwrap();
        assert_eq!(first.source, second.source);
    }

    /// Phase-1 DAG-IR seam (docs/plans/dag-ir-forge.md §8.2): routing gemm/gemv/fft through
    /// `KernelSpec::to_graph` → `lower_graph` (serviced by `WgslDelegateLowerer`) must emit
    /// the EXACT body the legacy leaf emitter produces — byte-identical, since the certify
    /// cache key folds in the source_hash. This also guards the bridge: if `to_graph`
    /// produced the wrong op-class, the routed body would not match the right leaf.
    #[test]
    fn graph_routing_is_byte_equal_for_gemm_gemv_fft() {
        let sched = Schedule::default();
        let cases: [(&str, KernelSpec); 3] = [
            ("gemm", BuiltinKernel::Gemm.spec()),
            ("gemv", BuiltinKernel::Gemv.spec()),
            ("fft", BuiltinKernel::Fft.spec()),
        ];
        for (id, kernel) in cases {
            let routed = emit_wgsl(&kernel, sched).unwrap();
            let mut legacy_body = String::new();
            match id {
                "gemm" => emit_gemm_wgsl(&mut legacy_body, &kernel, sched).unwrap(),
                "gemv" => emit_gemv_wgsl(&mut legacy_body, &kernel, sched).unwrap(),
                "fft" => emit_fft_wgsl(&mut legacy_body, &kernel, sched).unwrap(),
                _ => unreachable!(),
            }
            assert!(
                routed.source.ends_with(&legacy_body),
                "{id}: graph-routed WGSL body must be byte-identical to the legacy leaf body"
            );
        }
    }

    /// Phase-2: a native `Reduce` graph node lowers through `emit_graph_wgsl` →
    /// `WgslGraphLowerer` to the `reduce_wgsl` template (no `KernelSpec` involved). This
    /// proves the native-template lowering path, distinct from Phase 1's delegation.
    #[test]
    fn graph_native_reduce_lowers_to_template() {
        use crate::numerical::graph_ops::reduce::{reduce_wgsl, REDUCE_ENTRY};
        use crate::numerical::ir::graph::{
            Axis, ComputeGraph, DType, OpNode, RedKind, Shape, TensorRef,
        };

        let sched = Schedule {
            workgroup_size: 256,
            ..Default::default()
        };
        let mut g = ComputeGraph::new();
        let inp = TensorRef::external(Shape::new(&[1024]), DType::F32);
        g.push(
            OpNode::Reduce {
                op: RedKind::Sum,
                axis: Axis::Last,
            },
            &[inp],
            Shape::new(&[1]),
            DType::F32,
            sched,
        )
        .unwrap();
        let shader = emit_graph_wgsl(&g, sched).unwrap();
        assert!(shader.source.contains(REDUCE_ENTRY));
        assert!(
            shader.source.ends_with(&reduce_wgsl(RedKind::Sum, 256)),
            "native graph emission must equal the reduce template"
        );
    }

    #[test]
    fn graph_native_slice_lowers_to_template() {
        use crate::numerical::graph_ops::slice::{slice_wgsl, SLICE_ENTRY};
        use crate::numerical::ir::graph::{ComputeGraph, DType, OpNode, Shape, TensorRef};

        let sched = Schedule {
            workgroup_size: 64,
            ..Default::default()
        };
        let mut g = ComputeGraph::new();
        let inp = TensorRef::external(Shape::new(&[1024]), DType::F32);
        g.push(
            OpNode::Slice {
                offset: 128,
                len: 64,
            },
            &[inp],
            Shape::new(&[64]),
            DType::F32,
            sched,
        )
        .unwrap();
        let shader = emit_graph_wgsl(&g, sched).unwrap();
        assert!(shader.source.contains(SLICE_ENTRY));
        assert!(
            shader.source.ends_with(&slice_wgsl(64)),
            "native graph emission must equal the slice template"
        );
    }

    #[test]
    fn graph_native_gather_dequant_lowers_to_template() {
        use crate::numerical::graph_ops::gather_dequant::{
            gather_dequant_ternary_wgsl, GATHER_DEQUANT_ENTRY,
        };
        use crate::numerical::ir::graph::{ComputeGraph, DType, OpNode, Shape, TensorRef};

        let sched = Schedule {
            workgroup_size: 128,
            ..Default::default()
        };
        let mut g = ComputeGraph::new();
        let inp = TensorRef::external(Shape::new(&[16, 64]), DType::F32);
        g.push(
            OpNode::GatherDequant {
                scheme: DType::F32,
                block: 16,
            },
            &[inp],
            Shape::new(&[16, 64]),
            DType::F32,
            sched,
        )
        .unwrap();
        let shader = emit_graph_wgsl(&g, sched).unwrap();
        assert!(shader.source.contains(GATHER_DEQUANT_ENTRY));
        assert!(
            shader.source.ends_with(&gather_dequant_ternary_wgsl(128)),
            "native graph emission must equal the gather_dequant template"
        );
    }

    #[test]
    fn graph_native_scatter_lowers_to_template() {
        use crate::numerical::graph_ops::scatter::{scatter_wgsl, SCATTER_ENTRY};
        use crate::numerical::ir::graph::{
            AccumKind, ComputeGraph, DType, OpNode, Shape, TensorRef,
        };

        let sched = Schedule {
            workgroup_size: 64,
            ..Default::default()
        };
        let mut g = ComputeGraph::new();
        let src = TensorRef::external(Shape::new(&[128]), DType::F32);
        let idx = TensorRef::external(Shape::new(&[128]), DType::U32);
        g.push(
            OpNode::ScatterAccum { op: AccumKind::Add },
            &[src, idx],
            Shape::new(&[256]),
            DType::F32,
            sched,
        )
        .unwrap();
        let shader = emit_graph_wgsl(&g, sched).unwrap();
        assert!(shader.source.contains(SCATTER_ENTRY));
        assert!(
            shader.source.ends_with(&scatter_wgsl(64)),
            "native graph emission must equal the scatter template"
        );
    }

    #[test]
    fn p64_kernel_lays_64bit_fields_as_paired_u32_words() {
        // The p64 kernel carries 64-bit P64 fields. WGSL has no portable native u64,
        // so the generated module must lay them out as packed `u32` words. Assert the
        // exact packed layout the emitter produces (plan §1 / §10): a `P64Words64`
        // struct of `array<vec4<u32>, 4>` lanes, bound as `array<P64Words64>`, and the
        // per-word `f32(...)` projection over those `u32` lanes.
        let kernel = BuiltinKernel::P64Project.spec();
        let generated = emit_wgsl(&kernel, Schedule::default()).unwrap();
        let source = &generated.source;

        assert!(
            source.contains("lanes: array<vec4<u32>, 4>"),
            "expected paired/packed vec4<u32> word layout, got:\n{source}"
        );
        assert!(
            source.contains("var<storage, read> input: array<P64Words64>"),
            "expected the P64 record buffer bound as packed u32 words, got:\n{source}"
        );
        // The 64-bit data is read as u32 words and projected via f32(word) — i.e. no
        // native 64-bit scalar appears in the kernel body.
        assert!(
            source.contains("f32(word)"),
            "expected per-u32-word projection, got:\n{source}"
        );
        assert!(
            !source.contains("u64") && !source.contains("f64"),
            "the portable p64 kernel must not emit any native 64-bit scalar, got:\n{source}"
        );
    }
}
