//! `DeterministicMechanicsProvider` — Zero-heap kinematics and spatial geometry kernels
//! (`verlet_step`, `kinetic_energy`, `spring_damper_force`, `aabb_volume`, `excavation_prism_volume`).

use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, make_func_descriptor, ontology_triad, prepare_output,
    read_f64_finite, read_f64_vec3,
};

pub const MECH_PROVIDER_IRI: &str = "https://qualiadb.org/ext/etalon/mechanics";
pub const MECH_SHORT_ALIAS: &str = "mech";

static MECH_ONTOLOGY: [NQuin48; 3] = ontology_triad(
    MECH_PROVIDER_IRI,
    "https://qualiadb.org/ontology/etalon/mechanics#PositiveTimestepAndMass",
    "https://qualiadb.org/ontology/etalon/mechanics#SymplecticStepState",
    "https://qualiadb.org/ontology/etalon/mechanics#MechanicalEnergyConservation",
);

const PARAMS_VERLET: [ParamSpec; 4] = [
    ParamSpec::slice("pos_xyz", PrimitiveKind::F64, 3, 3),
    ParamSpec::slice("vel_xyz", PrimitiveKind::F64, 3, 3),
    ParamSpec::slice("acc_xyz", PrimitiveKind::F64, 3, 3),
    ParamSpec::scalar("dt", PrimitiveKind::F64),
];
const PARAMS_KINETIC: [ParamSpec; 2] = [
    ParamSpec::scalar("mass", PrimitiveKind::F64),
    ParamSpec::slice("vel_xyz", PrimitiveKind::F64, 3, 3),
];
const PARAMS_SPRING: [ParamSpec; 3] = [
    ParamSpec::slice("disp_xyz", PrimitiveKind::F64, 3, 3),
    ParamSpec::slice("vel_xyz", PrimitiveKind::F64, 3, 3),
    ParamSpec::slice("k_c", PrimitiveKind::F64, 2, 2),
];
const PARAMS_AABB: [ParamSpec; 2] = [
    ParamSpec::slice("min_xyz", PrimitiveKind::F64, 3, 3),
    ParamSpec::slice("max_xyz", PrimitiveKind::F64, 3, 3),
];
const PARAMS_PRISM: [ParamSpec; 1] = [ParamSpec::slice("base_area_depths", PrimitiveKind::F64, 4, 4)];

const KINEMATICS_FUNCS: [FuncDescriptor; 3] = [
    make_func_descriptor(
        "verlet_step",
        "https://qualiadb.org/ext/etalon/mechanics/kinematics/verlet_step",
        StableFuncId::new(0, 0),
        &PARAMS_VERLET,
        PrimitiveKind::F64,
        6,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(64, 48),
        "(F64[3],F64[3],F64[3],F64[1])->F64[6]",
    ),
    make_func_descriptor(
        "kinetic_energy",
        "https://qualiadb.org/ext/etalon/mechanics/kinematics/kinetic_energy",
        StableFuncId::new(0, 1),
        &PARAMS_KINETIC,
        PrimitiveKind::F64,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 8),
        "(F64[1],F64[3])->F64[1]",
    ),
    make_func_descriptor(
        "spring_damper_force",
        "https://qualiadb.org/ext/etalon/mechanics/kinematics/spring_damper_force",
        StableFuncId::new(0, 2),
        &PARAMS_SPRING,
        PrimitiveKind::F64,
        3,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(48, 24),
        "(F64[3],F64[3],F64[2])->F64[3]",
    ),
];

const GEOMETRY_FUNCS: [FuncDescriptor; 2] = [
    make_func_descriptor(
        "aabb_volume",
        "https://qualiadb.org/ext/etalon/mechanics/geometry/aabb_volume",
        StableFuncId::new(1, 0),
        &PARAMS_AABB,
        PrimitiveKind::F64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 16),
        "(F64[3],F64[3])->F64[2]",
    ),
    make_func_descriptor(
        "excavation_prism_volume",
        "https://qualiadb.org/ext/etalon/mechanics/geometry/excavation_prism_volume",
        StableFuncId::new(1, 1),
        &PARAMS_PRISM,
        PrimitiveKind::F64,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 8),
        "(F64[4])->F64[1]",
    ),
];

const MECH_SUB_NAMESPACES: [SubNamespaceDescriptor; 2] = [
    SubNamespaceDescriptor {
        name: "kinematics",
        sub_id: 0,
        functions: &KINEMATICS_FUNCS,
    },
    SubNamespaceDescriptor {
        name: "geometry",
        sub_id: 1,
        functions: &GEOMETRY_FUNCS,
    },
];

const MECH_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    canonical_iri: CanonicalIri::new(MECH_PROVIDER_IRI),
    short_alias: MECH_SHORT_ALIAS,
    versions: ContractVersionAxes::v0_3_default(
        SemVer64::release(0, 4, 2),
        SemVer64::release(0, 4, 2),
    ),
    supported_targets: TargetProfileMask::ALL_CPU,
    sub_namespaces: &MECH_SUB_NAMESPACES,
    max_ontology_quins: 8,
};

/// Reference oracle for `(0, 0) verlet_step`.
#[inline]
pub fn oracle_verlet_step(
    pos: [f64; 3],
    vel: [f64; 3],
    acc: [f64; 3],
    dt: f64,
) -> Result<[f64; 6], VibeTrap> {
    if !dt.is_finite()
        || dt <= 0.0
        || pos.iter().any(|x| !x.is_finite())
        || vel.iter().any(|x| !x.is_finite())
        || acc.iter().any(|x| !x.is_finite())
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let half_dt2 = 0.5 * dt * dt;
    let out = [
        pos[0] + vel[0] * dt + acc[0] * half_dt2,
        pos[1] + vel[1] * dt + acc[1] * half_dt2,
        pos[2] + vel[2] * dt + acc[2] * half_dt2,
        vel[0] + acc[0] * dt,
        vel[1] + acc[1] * dt,
        vel[2] + acc[2] * dt,
    ];
    if out.iter().any(|v| !v.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(out)
}

/// Reference oracle for `(0, 1) kinetic_energy`: `0.5 * m * ||v||^2`.
#[inline]
pub fn oracle_kinetic_energy(mass: f64, vel: [f64; 3]) -> Result<f64, VibeTrap> {
    if !mass.is_finite() || mass <= 0.0 || vel.iter().any(|v| !v.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let v_sq = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
    let ke = 0.5 * mass * v_sq;
    if !ke.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(ke)
}

/// Reference oracle for `(0, 2) spring_damper_force`: `F = -k * disp - c * vel`.
#[inline]
pub fn oracle_spring_damper_force(
    disp: [f64; 3],
    vel: [f64; 3],
    k: f64,
    c: f64,
) -> Result<[f64; 3], VibeTrap> {
    if !k.is_finite()
        || !c.is_finite()
        || k < 0.0
        || c < 0.0
        || disp.iter().any(|x| !x.is_finite())
        || vel.iter().any(|v| !v.is_finite())
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let out = [
        -k * disp[0] - c * vel[0],
        -k * disp[1] - c * vel[1],
        -k * disp[2] - c * vel[2],
    ];
    if out.iter().any(|f| !f.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(out)
}

/// Reference oracle for `(1, 0) aabb_volume`: `[volume, surface_area]`.
#[inline]
pub fn oracle_aabb_volume(min_xyz: [f64; 3], max_xyz: [f64; 3]) -> Result<[f64; 2], VibeTrap> {
    if min_xyz.iter().any(|x| !x.is_finite()) || max_xyz.iter().any(|x| !x.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let dx = max_xyz[0] - min_xyz[0];
    let dy = max_xyz[1] - min_xyz[1];
    let dz = max_xyz[2] - min_xyz[2];
    if dx < 0.0 || dy < 0.0 || dz < 0.0 {
        return Err(VibeTrap::NumericalDomainError);
    }
    let vol = dx * dy * dz;
    let area = 2.0 * (dx * dy + dy * dz + dz * dx);
    if !vol.is_finite() || !area.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([vol, area])
}

/// Reference oracle for `(1, 1) excavation_prism_volume`: `base_area * (d0 + d1 + d2) / 3.0`.
#[inline]
pub fn oracle_excavation_prism_volume(
    base_area: f64,
    d0: f64,
    d1: f64,
    d2: f64,
) -> Result<f64, VibeTrap> {
    if !base_area.is_finite()
        || !d0.is_finite()
        || !d1.is_finite()
        || !d2.is_finite()
        || base_area <= 0.0
        || d0 < 0.0
        || d1 < 0.0
        || d2 < 0.0
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let vol = base_area * (d0 + d1 + d2) / 3.0;
    if !vol.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(vol)
}

/// Zero-heap Etalon deterministic mechanics provider (`https://qualiadb.org/ext/etalon/mechanics`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicMechanicsProvider;

pub static MECH_PROVIDER: DeterministicMechanicsProvider = DeterministicMechanicsProvider;

impl VibeNamespaceProvider for DeterministicMechanicsProvider {
    #[inline]
    fn descriptor(&self) -> &'static ProviderDescriptor {
        &MECH_DESCRIPTOR
    }

    #[inline]
    fn static_ontology_nquins(&self) -> &'static [NQuin48] {
        &MECH_ONTOLOGY
    }

    fn call(
        &self,
        stable_id: StableFuncId,
        inputs: &[BorrowedSliceView<'_>],
        output: &mut CallerOutputSlice<'_>,
    ) -> Result<u64, VibeTrap> {
        match (stable_id.sub_namespace_id, stable_id.func_id) {
            (0, 0) => {
                check_arity(inputs, 4)?;
                let pos = read_f64_vec3(&inputs[0])?;
                let vel = read_f64_vec3(&inputs[1])?;
                let acc = read_f64_vec3(&inputs[2])?;
                check_len_exact(&inputs[3], PrimitiveKind::F64, 1)?;
                let dt = read_f64_finite(&inputs[3], 0)?;
                prepare_output(output, PrimitiveKind::F64, 6)?;
                let step = oracle_verlet_step(pos, vel, acc, dt)?;
                for v in step {
                    output.push_f64_le(v)?;
                }
                Ok(18)
            }
            (0, 1) => {
                check_arity(inputs, 2)?;
                check_len_exact(&inputs[0], PrimitiveKind::F64, 1)?;
                let mass = read_f64_finite(&inputs[0], 0)?;
                let vel = read_f64_vec3(&inputs[1])?;
                prepare_output(output, PrimitiveKind::F64, 1)?;
                let ke = oracle_kinetic_energy(mass, vel)?;
                output.push_f64_le(ke)?;
                Ok(8)
            }
            (0, 2) => {
                check_arity(inputs, 3)?;
                let disp = read_f64_vec3(&inputs[0])?;
                let vel = read_f64_vec3(&inputs[1])?;
                check_len_exact(&inputs[2], PrimitiveKind::F64, 2)?;
                let k = read_f64_finite(&inputs[2], 0)?;
                let c = read_f64_finite(&inputs[2], 1)?;
                prepare_output(output, PrimitiveKind::F64, 3)?;
                let force = oracle_spring_damper_force(disp, vel, k, c)?;
                for f in force {
                    output.push_f64_le(f)?;
                }
                Ok(12)
            }
            (1, 0) => {
                check_arity(inputs, 2)?;
                let min_xyz = read_f64_vec3(&inputs[0])?;
                let max_xyz = read_f64_vec3(&inputs[1])?;
                prepare_output(output, PrimitiveKind::F64, 2)?;
                let res = oracle_aabb_volume(min_xyz, max_xyz)?;
                output.push_f64_le(res[0])?;
                output.push_f64_le(res[1])?;
                Ok(10)
            }
            (1, 1) => {
                check_arity(inputs, 1)?;
                check_len_exact(&inputs[0], PrimitiveKind::F64, 4)?;
                let area = read_f64_finite(&inputs[0], 0)?;
                let d0 = read_f64_finite(&inputs[0], 1)?;
                let d1 = read_f64_finite(&inputs[0], 2)?;
                let d2 = read_f64_finite(&inputs[0], 3)?;
                prepare_output(output, PrimitiveKind::F64, 1)?;
                let vol = oracle_excavation_prism_volume(area, d0, d1, d2)?;
                output.push_f64_le(vol)?;
                Ok(8)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
