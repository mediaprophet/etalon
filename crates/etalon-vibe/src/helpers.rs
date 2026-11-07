//! Shared compile-time ontological helpers, 5-field `NQuin48` sealing, and
//! zero-heap finite slice validators for Etalon VibeScript providers.

use vibescript_abi::{
    q_hash_str, BorrowedSliceView, CallerOutputSlice, CanonicalIri, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, OntologicalContract, ParamSpec, PrimitiveKind,
    ResourceEnvelope, ReversibilityKind, StableFuncId, VibeTrap,
};

/// Constructs a 48-byte `NQuin48` with canonical 5-field XOR parity (`s ^ p ^ o ^ c ^ m`).
#[inline]
pub const fn seal_nquin48(s: u64, p: u64, o: u64, c: u64, m: u64) -> NQuin48 {
    NQuin48([s, p, o, c, m, s ^ p ^ o ^ c ^ m])
}

/// Verifies canonical 5-field XOR parity (`subject ^ predicate ^ object ^ context ^ metadata`).
#[inline]
pub const fn verify_nquin48_5field(quin: &NQuin48) -> bool {
    quin.0[5] == (quin.0[0] ^ quin.0[1] ^ quin.0[2] ^ quin.0[3] ^ quin.0[4])
}

/// Builds a 3-Quin static ontology triad (`[precondition, postcondition, invariant]`)
/// with canonical 5-field XOR parity.
#[inline]
pub const fn ontology_triad(
    provider_iri: &str,
    pre_iri: &str,
    post_iri: &str,
    inv_iri: &str,
) -> [NQuin48; 3] {
    let prov = q_hash_str(provider_iri);
    let ctx = q_hash_str("https://qualiadb.org/ontology/vibescript/v0.3");
    [
        seal_nquin48(
            prov,
            q_hash_str("https://qualiadb.org/ontology/vibescript#precondition"),
            q_hash_str(pre_iri),
            ctx,
            1,
        ),
        seal_nquin48(
            prov,
            q_hash_str("https://qualiadb.org/ontology/vibescript#postcondition"),
            q_hash_str(post_iri),
            ctx,
            2,
        ),
        seal_nquin48(
            prov,
            q_hash_str("https://qualiadb.org/ontology/vibescript#invariant"),
            q_hash_str(inv_iri),
            ctx,
            3,
        ),
    ]
}

/// Builds a static `FuncDescriptor` and computes its compile-time `OntologicalContract::canonical_digest`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub const fn make_func_descriptor(
    name: &'static str,
    func_iri: &'static str,
    stable_id: StableFuncId,
    params: &'static [ParamSpec],
    output_kind: PrimitiveKind,
    required_output_elems: u32,
    effects: EffectSet,
    determinism: DeterminismGuarantee,
    reversibility: ReversibilityKind,
    envelope: ResourceEnvelope,
    schema_sig: &'static str,
) -> FuncDescriptor {
    let canonical_iri = CanonicalIri::new(func_iri);
    let schema_hash = q_hash_str(schema_sig);
    let pre_span = (0, 1);
    let post_span = (1, 1);
    let inv_span = (2, 1);
    let contract_digest = OntologicalContract::canonical_digest(
        canonical_iri.fnv1a_hash,
        stable_id,
        schema_hash,
        effects,
        determinism,
        reversibility,
        envelope,
        pre_span,
        post_span,
        inv_span,
    );
    FuncDescriptor {
        name,
        canonical_iri,
        stable_id,
        params,
        output_kind,
        required_output_elems,
        effects,
        determinism,
        reversibility,
        envelope,
        contract: OntologicalContract {
            contract_digest,
            schema_hash,
            precondition_quin_span: pre_span,
            postcondition_quin_span: post_span,
            invariant_quin_span: inv_span,
        },
    }
}

/// Validates exact argument count.
#[inline]
pub const fn check_arity(inputs: &[BorrowedSliceView<'_>], expected: usize) -> Result<(), VibeTrap> {
    if inputs.len() != expected {
        return Err(VibeTrap::ArityMismatch);
    }
    Ok(())
}

/// Validates exact element kind and element count.
#[inline]
pub fn check_len_exact(
    view: &BorrowedSliceView<'_>,
    expected_kind: PrimitiveKind,
    expected_len: usize,
) -> Result<(), VibeTrap> {
    if view.kind() != expected_kind {
        return Err(VibeTrap::TypeMismatch);
    }
    if view.len() != expected_len {
        return Err(VibeTrap::SliceLengthMismatch);
    }
    Ok(())
}

/// Validates element kind and inclusive length range `min_len..=max_len`.
#[inline]
pub fn check_len_range(
    view: &BorrowedSliceView<'_>,
    expected_kind: PrimitiveKind,
    min_len: usize,
    max_len: usize,
) -> Result<usize, VibeTrap> {
    if view.kind() != expected_kind {
        return Err(VibeTrap::TypeMismatch);
    }
    let n = view.len();
    if n < min_len || n > max_len {
        return Err(VibeTrap::SliceLengthMismatch);
    }
    Ok(n)
}

/// Validates output element kind and capacity before writing.
#[inline]
pub fn prepare_output(
    output: &mut CallerOutputSlice<'_>,
    expected_kind: PrimitiveKind,
    required_elems: usize,
) -> Result<(), VibeTrap> {
    if output.kind() != expected_kind {
        return Err(VibeTrap::TypeMismatch);
    }
    output.ensure_elem_capacity(required_elems)
}

/// Appends one `u32` (little-endian) into `CallerOutputSlice`.
#[inline]
pub fn push_u32_le(output: &mut CallerOutputSlice<'_>, value: u32) -> Result<(), VibeTrap> {
    if output.kind() != PrimitiveKind::U32 {
        return Err(VibeTrap::TypeMismatch);
    }
    output.write_all(&value.to_le_bytes(), 1)
}

/// Reads a finite `f64` at `index`, returning `Err(VibeTrap::NumericalDomainError)` on `NaN`/`Inf`.
#[inline]
pub fn read_f64_finite(view: &BorrowedSliceView<'_>, index: usize) -> Result<f64, VibeTrap> {
    let v = view.read_f64_le(index)?;
    if !v.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(v)
}

/// Reads a finite `f32` at `index`, returning `Err(VibeTrap::NumericalDomainError)` on `NaN`/`Inf`.
#[inline]
pub fn read_f32_finite(view: &BorrowedSliceView<'_>, index: usize) -> Result<f32, VibeTrap> {
    let v = view.read_f32_le(index)?;
    if !v.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(v)
}

/// Reads a fixed 3-element finite `f64` vector.
#[inline]
pub fn read_f64_vec3(view: &BorrowedSliceView<'_>) -> Result<[f64; 3], VibeTrap> {
    check_len_exact(view, PrimitiveKind::F64, 3)?;
    Ok([
        read_f64_finite(view, 0)?,
        read_f64_finite(view, 1)?,
        read_f64_finite(view, 2)?,
    ])
}

/// Reads a fixed 3-element finite `f32` vector.
#[inline]
pub fn read_f32_vec3(view: &BorrowedSliceView<'_>) -> Result<[f32; 3], VibeTrap> {
    check_len_exact(view, PrimitiveKind::F32, 3)?;
    Ok([
        read_f32_finite(view, 0)?,
        read_f32_finite(view, 1)?,
        read_f32_finite(view, 2)?,
    ])
}

/// Reads a fixed 4-element finite `f32` vector.
#[inline]
pub fn read_f32_vec4(view: &BorrowedSliceView<'_>) -> Result<[f32; 4], VibeTrap> {
    check_len_exact(view, PrimitiveKind::F32, 4)?;
    Ok([
        read_f32_finite(view, 0)?,
        read_f32_finite(view, 1)?,
        read_f32_finite(view, 2)?,
        read_f32_finite(view, 3)?,
    ])
}
