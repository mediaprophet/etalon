//! `EtalonRenderProvider` — Zero-heap N-dimensional projection, SDF torus evaluation,
//! skeletal joint interpolation, and B-Rep topological Euler characteristic kernels.

use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, make_func_descriptor, ontology_triad, prepare_output,
    read_f32_finite, read_f32_vec3, read_f32_vec4,
};

pub const RENDER_PROVIDER_IRI: &str = "https://qualiadb.org/ext/etalon/render";
pub const RENDER_SHORT_ALIAS: &str = "render";

static RENDER_ONTOLOGY: [NQuin48; 3] = ontology_triad(
    RENDER_PROVIDER_IRI,
    "https://qualiadb.org/ontology/etalon/render#FiniteManifoldAndRadii",
    "https://qualiadb.org/ontology/etalon/render#MetricSignedDistanceField",
    "https://qualiadb.org/ontology/etalon/render#PoincareEulerTopologyInvariance",
);

const PARAMS_PROJ10D: [ParamSpec; 2] = [
    ParamSpec::slice("state10d", PrimitiveKind::F32, 10, 10),
    ParamSpec::slice("weights7d", PrimitiveKind::F32, 7, 7),
];
const PARAMS_SDF_TORUS: [ParamSpec; 2] = [
    ParamSpec::slice("point_xyz", PrimitiveKind::F32, 3, 3),
    ParamSpec::slice("radii_major_minor", PrimitiveKind::F32, 2, 2),
];
const PARAMS_SKELETAL_LERP: [ParamSpec; 3] = [
    ParamSpec::slice("joint_a", PrimitiveKind::F32, 3, 3),
    ParamSpec::slice("joint_b", PrimitiveKind::F32, 3, 3),
    ParamSpec::scalar("t", PrimitiveKind::F32),
];
const PARAMS_EULER: [ParamSpec; 1] = [ParamSpec::slice("v_e_f", PrimitiveKind::U32, 3, 3)];
const PARAMS_DIPOLE_PREVIEW: [ParamSpec; 3] = [
    ParamSpec::slice("dipole_moment_xyz", PrimitiveKind::F32, 3, 3),
    ParamSpec::slice("probe_offset_m", PrimitiveKind::F32, 3, 3),
    ParamSpec::scalar("gain", PrimitiveKind::F32),
];
const PARAMS_RADIATIVE_TRANSFER: [ParamSpec; 2] = [
    ParamSpec::slice("optical_params", PrimitiveKind::F32, 3, 3),
    ParamSpec::scalar("incident_radiance", PrimitiveKind::F32),
];
const PARAMS_SH_IRRADIANCE: [ParamSpec; 2] = [
    ParamSpec::slice("sh_l0_l1", PrimitiveKind::F32, 4, 4),
    ParamSpec::slice("normal_xyz", PrimitiveKind::F32, 3, 3),
];

const SPATIAL_FUNCS: [FuncDescriptor; 4] = [
    make_func_descriptor(
        "project_10d_to_3d",
        "https://qualiadb.org/ext/etalon/render/spatial/project_10d_to_3d",
        StableFuncId::new(0, 0),
        &PARAMS_PROJ10D,
        PrimitiveKind::F32,
        3,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(64, 12),
        "(F32[10],F32[7])->F32[3]",
    ),
    make_func_descriptor(
        "sdf_torus_eval",
        "https://qualiadb.org/ext/etalon/render/spatial/sdf_torus_eval",
        StableFuncId::new(0, 1),
        &PARAMS_SDF_TORUS,
        PrimitiveKind::F32,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(48, 4),
        "(F32[3],F32[2])->F32[1]",
    ),
    make_func_descriptor(
        "skeletal_lerp_joint",
        "https://qualiadb.org/ext/etalon/render/spatial/skeletal_lerp_joint",
        StableFuncId::new(0, 2),
        &PARAMS_SKELETAL_LERP,
        PrimitiveKind::F32,
        3,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 12),
        "(F32[3],F32[3],F32[1])->F32[3]",
    ),
    make_func_descriptor(
        "brep_euler_characteristic",
        "https://qualiadb.org/ext/etalon/render/spatial/brep_euler_characteristic",
        StableFuncId::new(0, 3),
        &PARAMS_EULER,
        PrimitiveKind::I64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(24, 16),
        "(U32[3])->I64[2]",
    ),
];

const EMF_FUNCS: [FuncDescriptor; 3] = [
    make_func_descriptor(
        "bioelectric_dipole_preview",
        "https://qualiadb.org/ext/etalon/render/emf/bioelectric_dipole_preview",
        StableFuncId::new(1, 0),
        &PARAMS_DIPOLE_PREVIEW,
        PrimitiveKind::F32,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(48, 8),
        "(F32[3],F32[3],F32[1])->F32[2]",
    ),
    make_func_descriptor(
        "radiative_transfer_slice",
        "https://qualiadb.org/ext/etalon/render/emf/radiative_transfer_slice",
        StableFuncId::new(1, 1),
        &PARAMS_RADIATIVE_TRANSFER,
        PrimitiveKind::F32,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(40, 8),
        "(F32[3],F32[1])->F32[2]",
    ),
    make_func_descriptor(
        "sh_irradiance_probe",
        "https://qualiadb.org/ext/etalon/render/emf/sh_irradiance_probe",
        StableFuncId::new(1, 2),
        &PARAMS_SH_IRRADIANCE,
        PrimitiveKind::F32,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(40, 8),
        "(F32[4],F32[3])->F32[2]",
    ),
];

const RENDER_SUB_NAMESPACES: [SubNamespaceDescriptor; 2] = [
    SubNamespaceDescriptor {
        name: "spatial",
        sub_id: 0,
        functions: &SPATIAL_FUNCS,
    },
    SubNamespaceDescriptor {
        name: "emf",
        sub_id: 1,
        functions: &EMF_FUNCS,
    },
];

const RENDER_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    canonical_iri: CanonicalIri::new(RENDER_PROVIDER_IRI),
    short_alias: RENDER_SHORT_ALIAS,
    versions: ContractVersionAxes::v0_3_default(
        SemVer64::release(0, 3, 0),
        SemVer64::release(0, 3, 0),
    ),
    supported_targets: TargetProfileMask(
        TargetProfileMask::ALL_CPU.0 | TargetProfileMask::GPU_COMPUTE.0,
    ),
    sub_namespaces: &RENDER_SUB_NAMESPACES,
    max_ontology_quins: 8,
};

/// Reference oracle for `(0, 0) project_10d_to_3d`: projects a 10D manifold state onto 3D coordinates.
#[inline]
pub fn oracle_project_10d_to_3d(
    state10d: &[f32; 10],
    weights7d: &[f32; 7],
) -> Result<[f32; 3], VibeTrap> {
    if state10d.iter().any(|x| !x.is_finite()) || weights7d.iter().any(|w| !w.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let mut acc = [state10d[0] as f64, state10d[1] as f64, state10d[2] as f64];
    for d in 0..7 {
        acc[d % 3] += (state10d[3 + d] as f64) * (weights7d[d] as f64);
    }
    let out = [acc[0] as f32, acc[1] as f32, acc[2] as f32];
    if out.iter().any(|v| !v.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(out)
}

/// Reference oracle for `(0, 1) sdf_torus_eval`: exact signed distance to a torus in the XZ plane.
#[inline]
pub fn oracle_sdf_torus_eval(
    point_xyz: [f32; 3],
    major_r: f32,
    minor_r: f32,
) -> Result<f32, VibeTrap> {
    if point_xyz.iter().any(|p| !p.is_finite())
        || !major_r.is_finite()
        || !minor_r.is_finite()
        || major_r <= 0.0
        || minor_r <= 0.0
        || minor_r >= major_r
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let px = point_xyz[0] as f64;
    let py = point_xyz[1] as f64;
    let pz = point_xyz[2] as f64;
    let qx = (px * px + pz * pz).sqrt() - (major_r as f64);
    let dist = ((qx * qx + py * py).sqrt() - (minor_r as f64)) as f32;
    if !dist.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(dist)
}

/// Reference oracle for `(0, 2) skeletal_lerp_joint`: clamped `[0, 1]` linear joint interpolation.
#[inline]
pub fn oracle_skeletal_lerp_joint(
    joint_a: [f32; 3],
    joint_b: [f32; 3],
    t: f32,
) -> Result<[f32; 3], VibeTrap> {
    if joint_a.iter().any(|x| !x.is_finite())
        || joint_b.iter().any(|x| !x.is_finite())
        || !t.is_finite()
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let tc = t.clamp(0.0, 1.0);
    let out = [
        joint_a[0] + (joint_b[0] - joint_a[0]) * tc,
        joint_a[1] + (joint_b[1] - joint_a[1]) * tc,
        joint_a[2] + (joint_b[2] - joint_a[2]) * tc,
    ];
    if out.iter().any(|v| !v.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(out)
}

/// Reference oracle for `(0, 3) brep_euler_characteristic`: `[chi = V - E + F, genus_times_2 = 2 - chi]`.
#[inline]
pub fn oracle_brep_euler_characteristic(v: u32, e: u32, f: u32) -> Result<[i64; 2], VibeTrap> {
    if v == 0 || e == 0 || f == 0 {
        return Err(VibeTrap::NumericalDomainError);
    }
    let chi = (v as i64) - (e as i64) + (f as i64);
    let genus_times_2 = 2 - chi;
    Ok([chi, genus_times_2])
}

/// Reference oracle for `(1, 0) bioelectric_dipole_preview`: `[potential_uv, field_mag_uv_per_m]`.
#[inline]
pub fn oracle_bioelectric_dipole_preview(
    dipole_moment_xyz: [f32; 3],
    probe_offset_m: [f32; 3],
    gain: f32,
) -> Result<[f32; 2], VibeTrap> {
    if dipole_moment_xyz.iter().any(|x| !x.is_finite())
        || probe_offset_m.iter().any(|x| !x.is_finite())
        || !gain.is_finite()
        || gain <= 0.0
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let rx = probe_offset_m[0] as f64;
    let ry = probe_offset_m[1] as f64;
    let rz = probe_offset_m[2] as f64;
    let r2 = (rx * rx) + (ry * ry) + (rz * rz);
    let r = r2.sqrt();
    if !r.is_finite() || r < 1e-6 {
        return Err(VibeTrap::NumericalDomainError);
    }
    let px = dipole_moment_xyz[0] as f64;
    let py = dipole_moment_xyz[1] as f64;
    let pz = dipole_moment_xyz[2] as f64;
    let p_dot_r = (px * rx) + (py * ry) + (pz * rz);
    let p_dot_r_hat = p_dot_r / r;
    let p2 = (px * px) + (py * py) + (pz * pz);
    let g = gain as f64;
    let inv_r3 = 1.0 / (r2 * r);
    let potential_uv = (g * p_dot_r * inv_r3) as f32;
    let field_mag_uv_per_m = (g * (p2 + 3.0 * p_dot_r_hat * p_dot_r_hat).sqrt() * inv_r3) as f32;
    if !potential_uv.is_finite() || !field_mag_uv_per_m.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([potential_uv, field_mag_uv_per_m])
}

/// Reference oracle for `(1, 1) radiative_transfer_slice`: `[transmitted_radiance, single_scatter_radiance]`.
#[inline]
pub fn oracle_radiative_transfer_slice(
    optical_params: [f32; 3],
    incident_radiance: f32,
) -> Result<[f32; 2], VibeTrap> {
    if optical_params.iter().any(|x| !x.is_finite())
        || !incident_radiance.is_finite()
        || incident_radiance <= 0.0
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let optical_depth = optical_params[0] as f64;
    let albedo = optical_params[1] as f64;
    let anisotropy_g = optical_params[2] as f64;
    if optical_depth < 0.0
        || !(0.0..=1.0).contains(&albedo)
        || !(-0.99..=0.99).contains(&anisotropy_g)
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let l0 = incident_radiance as f64;
    let transmittance = (-optical_depth).exp();
    let transmitted = (l0 * transmittance) as f32;
    let forward_fraction = 0.5 * (1.0 + anisotropy_g);
    let single_scatter = (l0 * albedo * (1.0 - transmittance) * forward_fraction) as f32;
    if !transmitted.is_finite() || !single_scatter.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([transmitted, single_scatter])
}

/// Reference oracle for `(1, 2) sh_irradiance_probe`: `[irradiance, directional_contrast]`.
#[inline]
pub fn oracle_sh_irradiance_probe(
    sh_l0_l1: [f32; 4],
    normal_xyz: [f32; 3],
) -> Result<[f32; 2], VibeTrap> {
    if sh_l0_l1.iter().any(|x| !x.is_finite()) || normal_xyz.iter().any(|x| !x.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let nx = normal_xyz[0] as f64;
    let ny = normal_xyz[1] as f64;
    let nz = normal_xyz[2] as f64;
    let norm = ((nx * nx) + (ny * ny) + (nz * nz)).sqrt();
    if !norm.is_finite() || norm <= 1e-6 {
        return Err(VibeTrap::NumericalDomainError);
    }
    let inv_n = 1.0 / norm;
    let ux = nx * inv_n;
    let uy = ny * inv_n;
    let uz = nz * inv_n;
    // Zonal harmonic cosine-lobe weights: pi*Y00 = sqrt(pi)/2, (2*pi/3)*Y1m = sqrt(pi/3)*n_i.
    let e0 = (core::f64::consts::PI.sqrt() * 0.5) * (sh_l0_l1[0] as f64);
    let e1 = (core::f64::consts::PI / 3.0).sqrt()
        * ((sh_l0_l1[1] as f64) * uy + (sh_l0_l1[2] as f64) * uz + (sh_l0_l1[3] as f64) * ux);
    let irradiance = (e0 + e1).max(0.0) as f32;
    let contrast = (e1 / (e0.abs() + e1.abs() + 1e-6)) as f32;
    if !irradiance.is_finite() || !contrast.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([irradiance, contrast])
}

/// Zero-heap Etalon spatial & volumetric render provider (`https://qualiadb.org/ext/etalon/render`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EtalonRenderProvider;

pub static RENDER_PROVIDER: EtalonRenderProvider = EtalonRenderProvider;

impl VibeNamespaceProvider for EtalonRenderProvider {
    #[inline]
    fn descriptor(&self) -> &'static ProviderDescriptor {
        &RENDER_DESCRIPTOR
    }

    #[inline]
    fn static_ontology_nquins(&self) -> &'static [NQuin48] {
        &RENDER_ONTOLOGY
    }

    fn call(
        &self,
        stable_id: StableFuncId,
        inputs: &[BorrowedSliceView<'_>],
        output: &mut CallerOutputSlice<'_>,
    ) -> Result<u64, VibeTrap> {
        match (stable_id.sub_namespace_id, stable_id.func_id) {
            (0, 0) => {
                check_arity(inputs, 2)?;
                check_len_exact(&inputs[0], PrimitiveKind::F32, 10)?;
                check_len_exact(&inputs[1], PrimitiveKind::F32, 7)?;
                let mut s10 = [0.0f32; 10];
                for (i, slot) in s10.iter_mut().enumerate() {
                    *slot = read_f32_finite(&inputs[0], i)?;
                }
                let mut w7 = [0.0f32; 7];
                for (i, slot) in w7.iter_mut().enumerate() {
                    *slot = read_f32_finite(&inputs[1], i)?;
                }
                prepare_output(output, PrimitiveKind::F32, 3)?;
                let p3 = oracle_project_10d_to_3d(&s10, &w7)?;
                for v in p3 {
                    output.push_f32_le(v)?;
                }
                Ok(14)
            }
            (0, 1) => {
                check_arity(inputs, 2)?;
                let pt = read_f32_vec3(&inputs[0])?;
                check_len_exact(&inputs[1], PrimitiveKind::F32, 2)?;
                let major_r = read_f32_finite(&inputs[1], 0)?;
                let minor_r = read_f32_finite(&inputs[1], 1)?;
                prepare_output(output, PrimitiveKind::F32, 1)?;
                let d = oracle_sdf_torus_eval(pt, major_r, minor_r)?;
                output.push_f32_le(d)?;
                Ok(10)
            }
            (0, 2) => {
                check_arity(inputs, 3)?;
                let a = read_f32_vec3(&inputs[0])?;
                let b = read_f32_vec3(&inputs[1])?;
                check_len_exact(&inputs[2], PrimitiveKind::F32, 1)?;
                let t = read_f32_finite(&inputs[2], 0)?;
                prepare_output(output, PrimitiveKind::F32, 3)?;
                let j = oracle_skeletal_lerp_joint(a, b, t)?;
                for v in j {
                    output.push_f32_le(v)?;
                }
                Ok(8)
            }
            (0, 3) => {
                check_arity(inputs, 1)?;
                check_len_exact(&inputs[0], PrimitiveKind::U32, 3)?;
                let v = inputs[0].read_u32_le(0)?;
                let e = inputs[0].read_u32_le(1)?;
                let f = inputs[0].read_u32_le(2)?;
                prepare_output(output, PrimitiveKind::I64, 2)?;
                let res = oracle_brep_euler_characteristic(v, e, f)?;
                output.push_i64_le(res[0])?;
                output.push_i64_le(res[1])?;
                Ok(6)
            }
            (1, 0) => {
                check_arity(inputs, 3)?;
                let moment = read_f32_vec3(&inputs[0])?;
                let offset = read_f32_vec3(&inputs[1])?;
                check_len_exact(&inputs[2], PrimitiveKind::F32, 1)?;
                let gain = read_f32_finite(&inputs[2], 0)?;
                prepare_output(output, PrimitiveKind::F32, 2)?;
                let res = oracle_bioelectric_dipole_preview(moment, offset, gain)?;
                output.push_f32_le(res[0])?;
                output.push_f32_le(res[1])?;
                Ok(12)
            }
            (1, 1) => {
                check_arity(inputs, 2)?;
                let optical = read_f32_vec3(&inputs[0])?;
                check_len_exact(&inputs[1], PrimitiveKind::F32, 1)?;
                let incident = read_f32_finite(&inputs[1], 0)?;
                prepare_output(output, PrimitiveKind::F32, 2)?;
                let res = oracle_radiative_transfer_slice(optical, incident)?;
                output.push_f32_le(res[0])?;
                output.push_f32_le(res[1])?;
                Ok(10)
            }
            (1, 2) => {
                check_arity(inputs, 2)?;
                let sh = read_f32_vec4(&inputs[0])?;
                let normal = read_f32_vec3(&inputs[1])?;
                prepare_output(output, PrimitiveKind::F32, 2)?;
                let res = oracle_sh_irradiance_probe(sh, normal)?;
                output.push_f32_le(res[0])?;
                output.push_f32_le(res[1])?;
                Ok(10)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
