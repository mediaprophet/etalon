//! Semantic numerical IR overlay and evidence-stage cache adapters over existing Forge.
//!
//! This is not a second compiler. Public `KernelSpec` / `Schedule` / emit APIs stay
//! unchanged; this module annotates them, guards rewrites, and versions receipts.

pub mod contract;
pub mod evidence;
pub mod rewrites;

#[cfg(test)]
mod tests;

pub use contract::{
    AliasingDecl, AliasingKind, ArithmeticModel, Assumption, AssumptionKind, Discretization,
    Effects, GeometryKind, MathFamily, MathOp, PhysicalUnit, ProductionContract, ReductionKind,
    ShapeAnnotation, SparseKind, TensorRole, PRODUCTION_IR_SCHEMA_V1, PRODUCTION_IR_SCHEMA_VERSION,
};
pub use evidence::{
    invalidate_descendants, Digest, DigestAlgorithm, EvidenceStage, MeasurementMethod,
    PromotionEvidence, ReplayReport, StageOptions, StageReceipt, EVIDENCE_SCHEMA_V1,
    EVIDENCE_SCHEMA_VERSION,
};
pub use rewrites::{
    apply_guarded, AppliedRewrite, RejectedRewrite, RewriteKind, RewriteRequest, RewriteTrace,
    MAX_REWRITE_REQUESTS,
};

pub(crate) fn delimited_blake3(schema: &[u8], parts: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(schema);
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().to_hex().to_string()
}

pub(crate) fn hex64(hex: &str) -> bool {
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())
}
