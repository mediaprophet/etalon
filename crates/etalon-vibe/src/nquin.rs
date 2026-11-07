//! `EtalonNQuinProvider` — Zero-heap 48-byte `NQuinWire` 5-field XOR parity verification
//! and GPU LSD radix sort bucket extraction backed by `etalon-nquin`.

use etalon_nquin::NQuinWire;
use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, make_func_descriptor, ontology_triad, prepare_output,
    push_u32_le,
};

pub const NQUIN_PROVIDER_IRI: &str = "https://qualiadb.org/ext/etalon/nquin";
pub const NQUIN_SHORT_ALIAS: &str = "nquin";

static NQUIN_ONTOLOGY: [NQuin48; 3] = ontology_triad(
    NQUIN_PROVIDER_IRI,
    "https://qualiadb.org/ontology/etalon/nquin#Aligned48ByteSuperQuin",
    "https://qualiadb.org/ontology/etalon/nquin#FiveFieldXorParityVerified",
    "https://qualiadb.org/ontology/etalon/nquin#StableRadixKeyPartitionInvariance",
);

const PARAMS_COMPUTE_PARITY: [ParamSpec; 1] = [ParamSpec::slice("fields5", PrimitiveKind::U64, 5, 5)];
const PARAMS_VERIFY_WIRE: [ParamSpec; 1] = [ParamSpec::slice("words6", PrimitiveKind::U64, 6, 6)];
const PARAMS_RADIX_BYTE: [ParamSpec; 2] = [
    ParamSpec::scalar("key", PrimitiveKind::U64),
    ParamSpec::scalar("pass_shift", PrimitiveKind::U32),
];

const WIRE_FUNCS: [FuncDescriptor; 3] = [
    make_func_descriptor(
        "compute_wire_parity",
        "https://qualiadb.org/ext/etalon/nquin/wire/compute_wire_parity",
        StableFuncId::new(0, 0),
        &PARAMS_COMPUTE_PARITY,
        PrimitiveKind::U64,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(16, 8),
        "(U64[5])->U64[1]",
    ),
    make_func_descriptor(
        "verify_wire_words",
        "https://qualiadb.org/ext/etalon/nquin/wire/verify_wire_words",
        StableFuncId::new(0, 1),
        &PARAMS_VERIFY_WIRE,
        PrimitiveKind::U64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(16, 16),
        "(U64[6])->U64[2]",
    ),
    make_func_descriptor(
        "radix_bucket_byte",
        "https://qualiadb.org/ext/etalon/nquin/wire/radix_bucket_byte",
        StableFuncId::new(0, 2),
        &PARAMS_RADIX_BYTE,
        PrimitiveKind::U32,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(16, 4),
        "(U64[1],U32[1])->U32[1]",
    ),
];

const NQUIN_SUB_NAMESPACES: [SubNamespaceDescriptor; 1] = [SubNamespaceDescriptor {
    name: "wire",
    sub_id: 0,
    functions: &WIRE_FUNCS,
}];

const NQUIN_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    canonical_iri: CanonicalIri::new(NQUIN_PROVIDER_IRI),
    short_alias: NQUIN_SHORT_ALIAS,
    versions: ContractVersionAxes::v0_3_default(
        SemVer64::release(0, 3, 0),
        SemVer64::release(0, 3, 0),
    ),
    supported_targets: TargetProfileMask(
        TargetProfileMask::ALL_CPU.0 | TargetProfileMask::GPU_COMPUTE.0,
    ),
    sub_namespaces: &NQUIN_SUB_NAMESPACES,
    max_ontology_quins: 8,
};

/// Reference oracle for `(0, 0) compute_wire_parity` using `etalon_nquin::NQuinWire::from_parts`.
#[inline]
pub const fn oracle_compute_wire_parity(fields5: [u64; 5]) -> u64 {
    NQuinWire::from_parts(fields5[0], fields5[1], fields5[2], fields5[3], fields5[4]).parity
}

/// Reference oracle for `(0, 1) verify_wire_words` using `etalon_nquin::NQuinWire`.
#[inline]
pub const fn oracle_verify_wire_words(words6: [u64; 6]) -> [u64; 2] {
    let wire = NQuinWire::new(
        words6[0], words6[1], words6[2], words6[3], words6[4], words6[5],
    );
    let valid = if wire.is_parity_valid() { 1 } else { 0 };
    [valid, wire.compute_parity()]
}

/// Reference oracle for `(0, 2) radix_bucket_byte`: extracts `(key >> pass_shift) & 0xFF`
/// for byte-aligned LSD radix sort passes (`0, 8, 16, 24, 32, 40, 48, 56`).
#[inline]
pub const fn oracle_radix_bucket_byte(key: u64, pass_shift: u32) -> Result<u32, VibeTrap> {
    if pass_shift > 56 || !pass_shift.is_multiple_of(8) {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(((key >> pass_shift) & 0xFF) as u32)
}

/// Zero-heap Etalon 48-byte `NQuinWire` & radix sort provider (`https://qualiadb.org/ext/etalon/nquin`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EtalonNQuinProvider;

pub static NQUIN_PROVIDER: EtalonNQuinProvider = EtalonNQuinProvider;

impl VibeNamespaceProvider for EtalonNQuinProvider {
    #[inline]
    fn descriptor(&self) -> &'static ProviderDescriptor {
        &NQUIN_DESCRIPTOR
    }

    #[inline]
    fn static_ontology_nquins(&self) -> &'static [NQuin48] {
        &NQUIN_ONTOLOGY
    }

    fn call(
        &self,
        stable_id: StableFuncId,
        inputs: &[BorrowedSliceView<'_>],
        output: &mut CallerOutputSlice<'_>,
    ) -> Result<u64, VibeTrap> {
        match (stable_id.sub_namespace_id, stable_id.func_id) {
            (0, 0) => {
                check_arity(inputs, 1)?;
                check_len_exact(&inputs[0], PrimitiveKind::U64, 5)?;
                let fields5 = [
                    inputs[0].read_u64_le(0)?,
                    inputs[0].read_u64_le(1)?,
                    inputs[0].read_u64_le(2)?,
                    inputs[0].read_u64_le(3)?,
                    inputs[0].read_u64_le(4)?,
                ];
                prepare_output(output, PrimitiveKind::U64, 1)?;
                let parity = oracle_compute_wire_parity(fields5);
                output.push_u64_le(parity)?;
                Ok(4)
            }
            (0, 1) => {
                check_arity(inputs, 1)?;
                check_len_exact(&inputs[0], PrimitiveKind::U64, 6)?;
                let words6 = [
                    inputs[0].read_u64_le(0)?,
                    inputs[0].read_u64_le(1)?,
                    inputs[0].read_u64_le(2)?,
                    inputs[0].read_u64_le(3)?,
                    inputs[0].read_u64_le(4)?,
                    inputs[0].read_u64_le(5)?,
                ];
                prepare_output(output, PrimitiveKind::U64, 2)?;
                let res = oracle_verify_wire_words(words6);
                output.push_u64_le(res[0])?;
                output.push_u64_le(res[1])?;
                Ok(5)
            }
            (0, 2) => {
                check_arity(inputs, 2)?;
                check_len_exact(&inputs[0], PrimitiveKind::U64, 1)?;
                check_len_exact(&inputs[1], PrimitiveKind::U32, 1)?;
                let key = inputs[0].read_u64_le(0)?;
                let pass_shift = inputs[1].read_u32_le(0)?;
                prepare_output(output, PrimitiveKind::U32, 1)?;
                let bucket = oracle_radix_bucket_byte(key, pass_shift)?;
                push_u32_le(output, bucket)?;
                Ok(4)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
