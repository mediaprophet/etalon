//! `etalon-vibe` — Comprehensive, zero-heap VibeScript extension provider suite for Etalon.
//!
//! Exports 8 self-describing `VibeNamespaceProvider` implementations backed by Etalon's
//! deterministic simulation, GPU compute, WGSL Forge, N-dimensional rendering, DSP/vision
//! signal processing, sensor telemetry, P64 model runtime, and 48-byte `NQuinWire` crates.

pub mod forge;
pub mod gpu;
pub mod helpers;
pub mod mechanics;
pub mod model;
pub mod nquin;
pub mod render;
pub mod sense;
pub mod signal;

#[cfg(test)]
mod tests;

pub use forge::{
    oracle_coop_matrix_tile_ops, oracle_iq_power_reference, oracle_tile_schedule_gemm,
    EtalonForgeProvider, FORGE_PROVIDER, FORGE_PROVIDER_IRI, FORGE_SHORT_ALIAS,
};
pub use gpu::{
    oracle_affine_f32, oracle_gemv4_f32, oracle_rmsnorm_f32, oracle_rope_f32, oracle_topk2_f32,
    EtalonGpuComputeProvider, GPU_PROVIDER, GPU_PROVIDER_IRI, GPU_SHORT_ALIAS,
};
pub use helpers::{
    make_func_descriptor, ontology_triad, seal_nquin48, verify_nquin48_5field,
};
pub use mechanics::{
    oracle_aabb_volume, oracle_excavation_prism_volume, oracle_kinetic_energy,
    oracle_spring_damper_force, oracle_verlet_step, DeterministicMechanicsProvider, MECH_PROVIDER,
    MECH_PROVIDER_IRI, MECH_SHORT_ALIAS,
};
pub use model::{
    oracle_moe_top2_softmax, oracle_p64_align_offset, oracle_residency_admit_check,
    EtalonModelProvider, MODEL_PROVIDER, MODEL_PROVIDER_IRI, MODEL_SHORT_ALIAS,
};
pub use nquin::{
    oracle_compute_wire_parity, oracle_radix_bucket_byte, oracle_verify_wire_words,
    EtalonNQuinProvider, NQUIN_PROVIDER, NQUIN_PROVIDER_IRI, NQUIN_SHORT_ALIAS,
};
pub use render::{
    oracle_brep_euler_characteristic, oracle_project_10d_to_3d, oracle_sdf_torus_eval,
    oracle_skeletal_lerp_joint, EtalonRenderProvider, RENDER_PROVIDER, RENDER_PROVIDER_IRI,
    RENDER_SHORT_ALIAS,
};
pub use sense::{
    oracle_clock_stamp_drift_ns, oracle_rssi_log_distance_m, oracle_spectrum_bin_hz,
    EtalonSenseProvider, SENSE_PROVIDER, SENSE_PROVIDER_IRI, SENSE_SHORT_ALIAS,
};
pub use signal::{
    oracle_biquad_df1_step, oracle_cfar_cell_average, oracle_region_mean_u8,
    oracle_signal_rms_peak, EtalonSignalProvider, SIGNAL_PROVIDER, SIGNAL_PROVIDER_IRI,
    SIGNAL_SHORT_ALIAS,
};
pub use vibescript_abi::VibeNamespaceProvider;

/// Static catalog of all 8 Etalon VibeScript extension providers.
pub static ETALON_VIBE_PROVIDERS: [&dyn VibeNamespaceProvider; 8] = [
    &GPU_PROVIDER,
    &MECH_PROVIDER,
    &FORGE_PROVIDER,
    &RENDER_PROVIDER,
    &SIGNAL_PROVIDER,
    &SENSE_PROVIDER,
    &MODEL_PROVIDER,
    &NQUIN_PROVIDER,
];

/// Canonical short aliases for all 8 Etalon VibeScript extension providers.
pub const ETALON_EXTENSION_ALIASES: [&str; 8] = [
    GPU_SHORT_ALIAS,
    MECH_SHORT_ALIAS,
    FORGE_SHORT_ALIAS,
    RENDER_SHORT_ALIAS,
    SIGNAL_SHORT_ALIAS,
    SENSE_SHORT_ALIAS,
    MODEL_SHORT_ALIAS,
    NQUIN_SHORT_ALIAS,
];

/// Canonical namespace IRIs for all 8 Etalon VibeScript extension providers.
pub const ETALON_EXTENSION_IRIS: [&str; 8] = [
    GPU_PROVIDER_IRI,
    MECH_PROVIDER_IRI,
    FORGE_PROVIDER_IRI,
    RENDER_PROVIDER_IRI,
    SIGNAL_PROVIDER_IRI,
    SENSE_PROVIDER_IRI,
    MODEL_PROVIDER_IRI,
    NQUIN_PROVIDER_IRI,
];

/// Looks up an Etalon VibeScript provider by its short alias (e.g. `"gpu"`, `"mech"`).
#[inline]
pub fn provider_by_alias(alias: &str) -> Option<&'static dyn VibeNamespaceProvider> {
    for &p in &ETALON_VIBE_PROVIDERS {
        if p.descriptor().short_alias == alias {
            return Some(p);
        }
    }
    None
}

/// Looks up an Etalon VibeScript provider by its canonical namespace IRI.
#[inline]
pub fn provider_by_iri(iri: &str) -> Option<&'static dyn VibeNamespaceProvider> {
    for &p in &ETALON_VIBE_PROVIDERS {
        if p.descriptor().canonical_iri.iri == iri {
            return Some(p);
        }
    }
    None
}
