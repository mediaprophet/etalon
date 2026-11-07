//! `EtalonForgeProvider` — Zero-heap WGSL Forge schedule & reference oracle provider
//! (`tile_schedule_gemm`, `iq_power_reference`, `coop_matrix_tile_ops`).

use etalon_signal::iq::power_into;
use etalon_wgsl_forge::{CoopMatShape, TileConfig, WorkgroupSize};
use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, check_len_range, make_func_descriptor, ontology_triad,
    prepare_output, push_u32_le, read_f32_finite,
};

pub const FORGE_PROVIDER_IRI: &str = "https://qualiadb.org/ext/etalon/forge";
pub const FORGE_SHORT_ALIAS: &str = "forge";

static FORGE_ONTOLOGY: [NQuin48; 3] = ontology_triad(
    FORGE_PROVIDER_IRI,
    "https://qualiadb.org/ontology/etalon/forge#PositiveWorkgroupTileGeometry",
    "https://qualiadb.org/ontology/etalon/forge#CertifiedDispatchSchedule",
    "https://qualiadb.org/ontology/etalon/forge#DeterministicTileInvocationCover",
);

const PARAMS_TILE_GEMM: [ParamSpec; 2] = [
    ParamSpec::slice("mnk", PrimitiveKind::U32, 3, 3),
    ParamSpec::slice("wg_mnk", PrimitiveKind::U32, 3, 3),
];
const PARAMS_IQ_POWER: [ParamSpec; 2] = [
    ParamSpec::slice("iq_interleaved", PrimitiveKind::F32, 2, 64),
    ParamSpec::scalar("scale", PrimitiveKind::F32),
];
const PARAMS_COOP_MAT: [ParamSpec; 1] = [ParamSpec::slice("mnk", PrimitiveKind::U32, 3, 3)];

const SHADER_FUNCS: [FuncDescriptor; 3] = [
    make_func_descriptor(
        "tile_schedule_gemm",
        "https://qualiadb.org/ext/etalon/forge/shader/tile_schedule_gemm",
        StableFuncId::new(0, 0),
        &PARAMS_TILE_GEMM,
        PrimitiveKind::U32,
        4,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 16),
        "(U32[3],U32[3])->U32[4]",
    ),
    make_func_descriptor(
        "iq_power_reference",
        "https://qualiadb.org/ext/etalon/forge/shader/iq_power_reference",
        StableFuncId::new(0, 1),
        &PARAMS_IQ_POWER,
        PrimitiveKind::F32,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(128, 4),
        "(F32[2..64],F32[1])->F32[1]",
    ),
    make_func_descriptor(
        "coop_matrix_tile_ops",
        "https://qualiadb.org/ext/etalon/forge/shader/coop_matrix_tile_ops",
        StableFuncId::new(0, 2),
        &PARAMS_COOP_MAT,
        PrimitiveKind::U64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 16),
        "(U32[3])->U64[2]",
    ),
];

const FORGE_SUB_NAMESPACES: [SubNamespaceDescriptor; 1] = [SubNamespaceDescriptor {
    name: "shader",
    sub_id: 0,
    functions: &SHADER_FUNCS,
}];

const FORGE_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    canonical_iri: CanonicalIri::new(FORGE_PROVIDER_IRI),
    short_alias: FORGE_SHORT_ALIAS,
    versions: ContractVersionAxes::v0_3_default(
        SemVer64::release(0, 3, 0),
        SemVer64::release(0, 3, 0),
    ),
    supported_targets: TargetProfileMask(
        TargetProfileMask::ALL_CPU.0 | TargetProfileMask::GPU_COMPUTE.0,
    ),
    sub_namespaces: &FORGE_SUB_NAMESPACES,
    max_ontology_quins: 8,
};

/// Reference oracle for `(0, 0) tile_schedule_gemm` using `etalon_wgsl_forge::{TileConfig, WorkgroupSize}`.
#[inline]
pub fn oracle_tile_schedule_gemm(mnk: [u32; 3], wg_mnk: [u32; 3]) -> Result<[u32; 4], VibeTrap> {
    if mnk.contains(&0) || wg_mnk.contains(&0) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let tile = TileConfig::new(wg_mnk[0], wg_mnk[1], wg_mnk[2]);
    let wg = WorkgroupSize::new(wg_mnk[0], wg_mnk[1], 1);
    let grid_x = mnk[1].div_ceil(tile.n);
    let grid_y = mnk[0].div_ceil(tile.m);
    let tiles_k = mnk[2].div_ceil(tile.k);
    let total_invocations = grid_x
        .checked_mul(grid_y)
        .and_then(|g| g.checked_mul(wg.flat_threads()))
        .ok_or(VibeTrap::NumericalDomainError)?;
    Ok([grid_x, grid_y, tiles_k, total_invocations])
}

/// Reference oracle for `(0, 1) iq_power_reference` using `etalon_signal::iq::power_into`.
#[inline]
pub fn oracle_iq_power_reference(iq_interleaved: &[f32], scale: f32) -> Result<f32, VibeTrap> {
    if iq_interleaved.len() < 2 || iq_interleaved.len() > 64 || !iq_interleaved.len().is_multiple_of(2) {
        return Err(VibeTrap::SliceLengthMismatch);
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(VibeTrap::NumericalDomainError);
    }
    let mut powers = [0.0f64; 32];
    let pair_count = power_into(iq_interleaved, &mut powers).map_err(|_| VibeTrap::NumericalDomainError)?;
    let mut sum = 0.0f64;
    for &p in &powers[..pair_count] {
        sum += p * (scale as f64);
    }
    let mean = (sum / (pair_count as f64)) as f32;
    if !mean.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(mean)
}

/// Reference oracle for `(0, 2) coop_matrix_tile_ops` using `etalon_wgsl_forge::CoopMatShape::f16_16x16x16`.
#[inline]
pub fn oracle_coop_matrix_tile_ops(mnk: [u32; 3]) -> Result<[u64; 2], VibeTrap> {
    if mnk.contains(&0) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let shape = CoopMatShape::f16_16x16x16(mnk[0], mnk[1], mnk[2]);
    let tm = shape.tile_m as u64;
    let tn = shape.tile_n as u64;
    let tk = shape.tile_k as u64;
    let tiles_m = (shape.m as u64).div_ceil(tm);
    let tiles_n = (shape.n as u64).div_ceil(tn);
    let tiles_k = (shape.k as u64).div_ceil(tk);
    let mn_tiles = tiles_m
        .checked_mul(tiles_n)
        .ok_or(VibeTrap::NumericalDomainError)?;
    let total_tiles = mn_tiles
        .checked_mul(tiles_k)
        .ok_or(VibeTrap::NumericalDomainError)?;
    let fmas_per_tile = tm * tn * tk; // 4096 FMAs per 16x16x16 tile
    let total_fmas = total_tiles
        .checked_mul(fmas_per_tile)
        .ok_or(VibeTrap::NumericalDomainError)?;
    let bytes_ab = total_tiles
        .checked_mul(1024)
        .ok_or(VibeTrap::NumericalDomainError)?;
    let bytes_c = mn_tiles
        .checked_mul(1024)
        .ok_or(VibeTrap::NumericalDomainError)?;
    let bytes_transferred = bytes_ab
        .checked_add(bytes_c)
        .ok_or(VibeTrap::NumericalDomainError)?;
    Ok([total_fmas, bytes_transferred])
}

/// Zero-heap Etalon WGSL Forge provider (`https://qualiadb.org/ext/etalon/forge`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EtalonForgeProvider;

pub static FORGE_PROVIDER: EtalonForgeProvider = EtalonForgeProvider;

impl VibeNamespaceProvider for EtalonForgeProvider {
    #[inline]
    fn descriptor(&self) -> &'static ProviderDescriptor {
        &FORGE_DESCRIPTOR
    }

    #[inline]
    fn static_ontology_nquins(&self) -> &'static [NQuin48] {
        &FORGE_ONTOLOGY
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
                check_len_exact(&inputs[0], PrimitiveKind::U32, 3)?;
                check_len_exact(&inputs[1], PrimitiveKind::U32, 3)?;
                let mnk = [
                    inputs[0].read_u32_le(0)?,
                    inputs[0].read_u32_le(1)?,
                    inputs[0].read_u32_le(2)?,
                ];
                let wg = [
                    inputs[1].read_u32_le(0)?,
                    inputs[1].read_u32_le(1)?,
                    inputs[1].read_u32_le(2)?,
                ];
                prepare_output(output, PrimitiveKind::U32, 4)?;
                let res = oracle_tile_schedule_gemm(mnk, wg)?;
                for v in res {
                    push_u32_le(output, v)?;
                }
                Ok(8)
            }
            (0, 1) => {
                check_arity(inputs, 2)?;
                let n = check_len_range(&inputs[0], PrimitiveKind::F32, 2, 64)?;
                if !n.is_multiple_of(2) {
                    return Err(VibeTrap::SliceLengthMismatch);
                }
                let mut iq = [0.0f32; 64];
                for i in 0..n {
                    iq[i] = read_f32_finite(&inputs[0], i)?;
                }
                check_len_exact(&inputs[1], PrimitiveKind::F32, 1)?;
                let scale = read_f32_finite(&inputs[1], 0)?;
                prepare_output(output, PrimitiveKind::F32, 1)?;
                let p = oracle_iq_power_reference(&iq[..n], scale)?;
                output.push_f32_le(p)?;
                Ok((n as u64) + 4)
            }
            (0, 2) => {
                check_arity(inputs, 1)?;
                check_len_exact(&inputs[0], PrimitiveKind::U32, 3)?;
                let mnk = [
                    inputs[0].read_u32_le(0)?,
                    inputs[0].read_u32_le(1)?,
                    inputs[0].read_u32_le(2)?,
                ];
                prepare_output(output, PrimitiveKind::U64, 2)?;
                let res = oracle_coop_matrix_tile_ops(mnk)?;
                output.push_u64_le(res[0])?;
                output.push_u64_le(res[1])?;
                Ok(8)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
