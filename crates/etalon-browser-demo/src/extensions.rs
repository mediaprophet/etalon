use serde_json::{json, Value};

use crate::browser::failure;

pub(crate) fn nquin() -> Result<Value, Value> {
    use etalon_nquin::{radix_sort_u64_cpu, sieve_eq_cpu, NQuinWire, QuinField};
    let mut quins = [NQuinWire::new(0, 0, 0, 0, 0, 0); 6];
    for (i, q) in quins.iter_mut().enumerate() {
        *q = NQuinWire::new(
            i as u64,
            if i % 2 == 0 { 42 } else { 7 },
            100 + i as u64,
            3,
            0,
            0,
        );
        q.parity = q.compute_parity();
    }
    if core::mem::size_of::<NQuinWire>() != 48
        || core::mem::align_of::<NQuinWire>() != 16
        || quins.iter().any(|q| !q.is_parity_valid())
    {
        return Err(failure(
            "nquin-abi-parity",
            "48-byte / 16-byte alignment or parity invariant failed",
        ));
    }
    let mut corrupt = quins[0];
    corrupt.object ^= 1;
    if corrupt.is_parity_valid() {
        return Err(failure(
            "nquin-corruption",
            "Changed object did not invalidate parity",
        ));
    }
    let mut out = [quins[0]; 6];
    let count = sieve_eq_cpu(&quins, QuinField::Predicate, 42, &mut out);
    if count != 3 || out[..count] != [quins[0], quins[2], quins[4]] {
        return Err(failure(
            "nquin-sieve",
            "Predicate sieve differs from expected stable matches",
        ));
    }
    let mut keys = [u64::MAX, 42, 0, 42, 1u64 << 40, 7];
    let mut indices = [0, 1, 2, 3, 4, 5];
    radix_sort_u64_cpu(&mut keys, &mut indices);
    if keys != [0, 7, 42, 42, 1u64 << 40, u64::MAX] || indices != [2, 5, 1, 3, 4, 0] {
        return Err(failure(
            "nquin-sort",
            "64-bit stable radix sort differs from expected keys and indices",
        ));
    }
    Ok(
        json!({"execution": "WASM CPU", "wire_bytes": 48, "wire_alignment": 16,
        "parity_checked": 6, "corruption_detected": true, "sieve_matches": count,
        "sorted_keys": keys.map(|k| k.to_string()), "stable_indices": indices}),
    )
}

pub(crate) fn forge() -> Result<Value, Value> {
    use etalon_wgsl_forge::{emit_wgsl, validate_wgsl, BuiltinKernel, Schedule};
    let spec = BuiltinKernel::AffineF32.spec();
    let schedule = Schedule::from_1d(64, 1, 1);
    let first =
        emit_wgsl(&spec, schedule).map_err(|e| failure("forge-emission", &e.to_string()))?;
    let second =
        emit_wgsl(&spec, schedule).map_err(|e| failure("forge-emission", &e.to_string()))?;
    if first != second || first.source.len() > 32768 {
        return Err(failure(
            "forge-determinism",
            "Repeated emission differs or exceeds source bound",
        ));
    }
    let validation =
        validate_wgsl(&first.source).map_err(|e| failure("forge-validation", &e.to_string()))?;
    if !validation.naga_validated
        || validation.binding_count != 3
        || validation.entry_points != [spec.entry_point]
    {
        return Err(failure(
            "forge-validation",
            "Generated affine kernel validation or reflected bindings differ",
        ));
    }
    if validate_wgsl("@compute @workgroup_size(1) fn broken( {").is_ok() {
        return Err(failure(
            "forge-negative-validation",
            "Malformed WGSL unexpectedly accepted",
        ));
    }
    Ok(
        json!({"execution": "WASM CPU", "kernel": first.kernel_id, "deterministic": true,
        "source_bytes": first.source.len(), "source_hash": first.source_hash,
        "semantic_hash": first.semantic_hash, "naga_validated": true,
        "bindings": validation.binding_count, "entry_points": validation.entry_points,
        "malformed_shader_rejected": true}),
    )
}
