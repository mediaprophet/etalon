use super::*;
use crate::ir::{BufferElement, ForgeError, ScalarType};
use crate::production_ir::{ArithmeticModel, ProductionContract};
use crate::schedule::{AdapterConstraints, Schedule};

fn portable() -> AdapterConstraints {
    AdapterConstraints::portable()
}

fn binding_slots() -> [BindingDecl; MAX_BINDINGS] {
    std::array::from_fn(|_| BindingDecl::default())
}

fn fallback_slots() -> [FallbackRecord; MAX_FALLBACKS] {
    [FallbackRecord::default(); MAX_FALLBACKS]
}

fn geometry_contract() -> ProductionContract {
    ProductionContract::geometry_point_in_tetra(Schedule::default()).expect("geometry contract")
}

fn df64_reduction() -> ProductionContract {
    let mut contract =
        ProductionContract::reduction_sum(Schedule::default()).expect("reduction contract");
    contract.arithmetic = ArithmeticModel::Df64PairedF32;
    contract
        .validate()
        .expect("df64 paired-f32 reduction is legal");
    contract
}

#[test]
fn happy_path_emits_geometry_wgsl_with_explicit_decls() {
    let contract = geometry_contract();
    let mut bindings = binding_slots();
    let mut fallbacks = fallback_slots();
    let (lowered, admission) = lower_admitted(
        &contract,
        &portable(),
        &AdmissionOverride::from_contract(&contract),
        &mut bindings,
        &mut fallbacks,
    )
    .expect("geometry lowering");

    assert_eq!(lowered.emitter, EmitterKind::GeometryTetrahedronSpec);
    assert_eq!(lowered.layout, LayoutKind::GeometryPacked15);
    assert_eq!(lowered.workgroup.x, 64);
    assert_eq!(lowered.binding_count, 3);
    assert_eq!(bindings[0].name, "data");
    assert_eq!(bindings[0].wgsl_type, "array<f32>");
    assert_eq!(bindings[1].name, "result_out");
    assert_eq!(bindings[1].wgsl_type, "array<i32>");
    assert_eq!(bindings[2].name, "params");
    assert_eq!(bindings[2].wgsl_type, "Params");
    assert!(lowered.shader.source.contains("@group(0)"));
    assert!(lowered.shader.source.contains("@binding(0)"));
    assert!(lowered.shader.source.contains("@binding(1)"));
    assert!(lowered.shader.source.contains("@binding(2)"));
    assert!(lowered.shader.source.contains("@workgroup_size(64, 1, 1)"));
    assert!(lowered.shader.source.contains("fn point_in_tetra_f32"));
    assert!(lowered.shader.source.contains("Conservative f32 orient3d"));
    assert!(!lowered.shader.source.contains("f64"));
    assert_eq!(CLUSTER_NODE_STRIDE_SCALARS, 12);
    assert_eq!(GEOMETRY_PACKED_QUERY_STRIDE, 15);
    assert!(admission.accepted);
    assert!(!admission.occupancy.assumed);
    assert_eq!(admission.occupancy.note, OCCUPANCY_NOTE);
    assert_eq!(admission.shared_memory.declared_bytes, 0);
    match lowered.naga {
        NagaStep::ParsedAndValidated {
            ref entry_points, ..
        } => {
            assert!(
                entry_points
                    .iter()
                    .any(|entry| entry == "point_in_tetra_f32"),
                "{entry_points:?}"
            );
        }
        NagaStep::Unsupported => {
            assert!(
                !cfg!(feature = "naga-validate"),
                "naga-validate feature must record parse/validate, not unsupported"
            );
        }
    }
}

#[test]
fn type_binding_mismatch_fails_closed() {
    let mut contract = geometry_contract();
    contract.kernel.buffers[1].element = BufferElement::Scalar(ScalarType::F32);
    contract.tensors[1].element = BufferElement::Scalar(ScalarType::F32);
    contract
        .validate()
        .expect("mutated contract still self-consistent");
    let mut bindings = binding_slots();
    match lower_contract(&contract, &portable(), &mut bindings) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("type/binding mismatch"), "{message}");
            assert!(
                message.contains("result_out") || message.contains("array<i32>"),
                "{message}"
            );
        }
        other => panic!("expected fail-closed mismatch, got {other:?}"),
    }
}

#[test]
fn df64_is_not_native_wgsl_f64() {
    let contract = df64_reduction();
    assert_eq!(contract.arithmetic, ArithmeticModel::Df64PairedF32);
    assert_eq!(DF64_PAIR_STRIDE, 2);
    assert_eq!(layout_from_contract(&contract), LayoutKind::Df64PairedF32);

    let wrapped =
        crate::emit_wgsl(&contract.kernel, contract.schedule).expect("existing Forge emit wrap");
    assert!(!super::admission::contains_wgsl_f64_type(&wrapped.source));
    assert!(!wrapped.source.contains("<f64>"));
    assert!(!wrapped.source.contains("array<f64>"));

    let mut fallbacks = fallback_slots();
    let request = AdmissionRequest {
        contract: &contract,
        constraints: &portable(),
        precision: RequestedPrecision::NativeWgslF64,
        layout: LayoutKind::Df64PairedF32,
        occupancy: OccupancyAssumption::None,
        subgroup_required: false,
        assumed_subgroup_size: None,
    };
    let report = admit(&request, Some(&wrapped.source), &mut fallbacks)
        .expect("native f64 falls back to paired f32");
    assert!(report.accepted);
    assert_eq!(report.fallback_count, 1);
    assert_eq!(fallbacks[0].kind, FallbackKind::Precision);
    assert_eq!(fallbacks[0].requested, "native_wgsl_f64");
    assert_eq!(fallbacks[0].admitted, "df64_paired_f32");
    assert!(fallbacks[0].note.contains("paired f32"));
    match report.precision {
        PrecisionAdmission::Fallback { model } => {
            assert_eq!(model, ArithmeticModel::Df64PairedF32);
        }
        other => panic!("expected fallback, got {other:?}"),
    }

    let geometry = geometry_contract();
    let mut bindings = binding_slots();
    let lowered = lower_contract(&geometry, &portable(), &mut bindings).expect("geometry emit");
    assert!(!super::admission::contains_wgsl_f64_type(
        &lowered.shader.source
    ));
    fallbacks = fallback_slots();
    let geo_req = AdmissionRequest {
        contract: &geometry,
        constraints: &portable(),
        precision: RequestedPrecision::NativeWgslF64,
        layout: LayoutKind::GeometryPacked15,
        occupancy: OccupancyAssumption::None,
        subgroup_required: false,
        assumed_subgroup_size: None,
    };
    let geo = admit(&geo_req, Some(&lowered.shader.source), &mut fallbacks)
        .expect("geometry native-f64 request falls back");
    match geo.precision {
        PrecisionAdmission::Fallback { model } => {
            assert_eq!(model, ArithmeticModel::FilteredPredicate);
        }
        other => panic!("expected filtered fallback, got {other:?}"),
    }
}

#[test]
fn occupancy_undeclared_is_rejected() {
    let contract = geometry_contract();
    let mut bindings = binding_slots();
    let mut fallbacks = fallback_slots();
    let mut override_req = AdmissionOverride::from_contract(&contract);
    override_req.occupancy = OccupancyAssumption::Undeclared;
    match lower_admitted(
        &contract,
        &portable(),
        &override_req,
        &mut bindings,
        &mut fallbacks,
    ) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("undeclared occupancy"), "{message}");
            assert!(
                message.contains("not measured GPU timing") || message.contains("admission/report"),
                "{message}"
            );
        }
        other => panic!("expected undeclared occupancy rejection, got {other:?}"),
    }

    let request = AdmissionRequest {
        contract: &contract,
        constraints: &portable(),
        precision: RequestedPrecision::FilteredPredicate,
        layout: LayoutKind::ClusterNode12,
        occupancy: OccupancyAssumption::None,
        subgroup_required: false,
        assumed_subgroup_size: None,
    };
    match admit(&request, None, &mut fallbacks) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("layout mismatch"), "{message}");
        }
        other => panic!("expected 12-vs-15 layout rejection, got {other:?}"),
    }
}
