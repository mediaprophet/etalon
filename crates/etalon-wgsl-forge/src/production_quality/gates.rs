//! Frozen quality/resource thresholds. CRC is a checksum, not a signature.

use crate::certify::ComparisonReport;
use crate::ir::ForgeError;
use crate::production_ir::Digest;
use crate::production_lowering::PORTABLE_MAX_SHARED_BYTES;
use crate::schedule::AdapterConstraints;

use super::PRODUCTION_QUALITY_SCHEMA_V1;

pub const FROZEN_MAX_ORACLE_MISMATCHES: usize = 0;
pub const FROZEN_MAX_ABSOLUTE_ERROR: f32 = 1.0e-6;
pub const FROZEN_MAX_RELATIVE_ERROR: f32 = 1.0e-5;
pub const FROZEN_MAX_SHARED_BYTES: u32 = PORTABLE_MAX_SHARED_BYTES;
pub const FROZEN_MAX_WORKGROUP_THREADS: u32 = 256;
pub const LEARNED_SEARCH_EXCLUDED_NOTE: &str =
    "optional .p64 learned search is excluded from core; equal-budget deterministic search is the baseline";

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QualityThresholds {
    pub max_oracle_mismatches: usize,
    pub max_absolute_error: f32,
    pub max_relative_error: f32,
    pub max_shared_bytes: u32,
    pub max_workgroup_threads: u32,
}

impl QualityThresholds {
    pub const fn production() -> Self {
        Self {
            max_oracle_mismatches: FROZEN_MAX_ORACLE_MISMATCHES,
            max_absolute_error: FROZEN_MAX_ABSOLUTE_ERROR,
            max_relative_error: FROZEN_MAX_RELATIVE_ERROR,
            max_shared_bytes: FROZEN_MAX_SHARED_BYTES,
            max_workgroup_threads: FROZEN_MAX_WORKGROUP_THREADS,
        }
    }

    fn same_as_production(&self) -> bool {
        let production = Self::production();
        self.max_oracle_mismatches == production.max_oracle_mismatches
            && self.max_absolute_error.to_bits() == production.max_absolute_error.to_bits()
            && self.max_relative_error.to_bits() == production.max_relative_error.to_bits()
            && self.max_shared_bytes == production.max_shared_bytes
            && self.max_workgroup_threads == production.max_workgroup_threads
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FrozenQualityGates {
    thresholds: QualityThresholds,
    digest_hex: String,
}

impl FrozenQualityGates {
    pub fn production() -> Self {
        Self::freeze(QualityThresholds::production()).expect("production quality thresholds")
    }

    /// Only the production constants freeze. Loosened gates are tampering.
    pub fn freeze(thresholds: QualityThresholds) -> Result<Self, ForgeError> {
        if !thresholds.same_as_production() {
            return Err(tamper_error());
        }
        if thresholds.max_workgroup_threads
            != AdapterConstraints::portable().max_invocations_per_workgroup
        {
            return Err(tamper_error());
        }
        Ok(Self {
            digest_hex: thresholds_digest(&thresholds)?,
            thresholds,
        })
    }

    pub fn thresholds(&self) -> QualityThresholds {
        self.thresholds
    }

    pub fn digest_hex(&self) -> &str {
        &self.digest_hex
    }

    pub fn validate(&self) -> Result<(), ForgeError> {
        if !self.thresholds.same_as_production() {
            return Err(tamper_error());
        }
        let expected = thresholds_digest(&self.thresholds)?;
        if self.digest_hex != expected {
            return Err(tamper_error());
        }
        Ok(())
    }

    pub fn attempt_retune(&self, proposed: QualityThresholds) -> Result<(), ForgeError> {
        self.validate()?;
        if !proposed.same_as_production() {
            return Err(tamper_error());
        }
        Ok(())
    }
}

pub fn admit_oracle(
    gates: &FrozenQualityGates,
    oracle: &ComparisonReport,
) -> Result<(), ForgeError> {
    gates.validate()?;
    if oracle.mismatch_count > gates.thresholds.max_oracle_mismatches || !oracle.passed() {
        return Err(ForgeError::Validation(
            "invalid fast candidate cannot win: oracle failed frozen quality gates".to_string(),
        ));
    }
    if !oracle.max_absolute_error.is_finite()
        || !oracle.max_relative_error.is_finite()
        || oracle.max_absolute_error > gates.thresholds.max_absolute_error
        || oracle.max_relative_error > gates.thresholds.max_relative_error
    {
        return Err(ForgeError::Validation(
            "invalid fast candidate cannot win: error exceeds frozen quality thresholds"
                .to_string(),
        ));
    }
    Ok(())
}

/// CRC detects corruption. It is not a content identity or an authentication signature.
pub fn parse_content_digest(algorithm: &str, hex: &str) -> Result<Digest, ForgeError> {
    crc_is_not_a_signature(algorithm)?;
    Digest::parse(algorithm, hex)
}

pub fn crc_is_not_a_signature(algorithm: &str) -> Result<(), ForgeError> {
    let normalized = algorithm.trim().to_ascii_lowercase();
    if normalized == "crc" || normalized.starts_with("crc") {
        return Err(ForgeError::Validation(
            "CRC is a checksum, not a signature".to_string(),
        ));
    }
    Ok(())
}

fn thresholds_digest(thresholds: &QualityThresholds) -> Result<String, ForgeError> {
    let mismatches = (thresholds.max_oracle_mismatches as u64).to_le_bytes();
    let absolute = thresholds.max_absolute_error.to_bits().to_le_bytes();
    let relative = thresholds.max_relative_error.to_bits().to_le_bytes();
    let shared = thresholds.max_shared_bytes.to_le_bytes();
    let threads = thresholds.max_workgroup_threads.to_le_bytes();
    Ok(crate::production_ir::delimited_blake3(
        PRODUCTION_QUALITY_SCHEMA_V1,
        &[
            mismatches.as_slice(),
            absolute.as_slice(),
            relative.as_slice(),
            shared.as_slice(),
            threads.as_slice(),
        ],
    ))
}

fn tamper_error() -> ForgeError {
    ForgeError::Validation("immutable quality thresholds reject tampering".to_string())
}
