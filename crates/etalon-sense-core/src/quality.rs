//! Explicit source, calibration, and measurement-quality declarations.
//!
//! These descriptors preserve caller-supplied state; they do not authenticate
//! a source or prove that calibration occurred. In particular, this API has no
//! way to construct a verified/trusted source. Verification belongs to a
//! separate authority/provider contract.

use crate::ContractError;

/// How the sample was obtained. This describes provenance class, not trust.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Synthetic,
    Recorded,
    Live,
}

/// Trust status available in this initial contract slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrustState {
    /// No authority/provider verification has been supplied.
    Unverified,
}

/// A source declaration. All declarations remain unverified, including `Live`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceDescriptor {
    source_id: u64,
    kind: SourceKind,
}

impl SourceDescriptor {
    pub const fn new(source_id: u64, kind: SourceKind) -> Result<Self, ContractError> {
        if source_id == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self { source_id, kind })
    }

    pub const fn source_id(self) -> u64 {
        self.source_id
    }

    pub const fn kind(self) -> SourceKind {
        self.kind
    }

    pub const fn trust_state(self) -> TrustState {
        TrustState::Unverified
    }
}

/// Opaque nonzero identifier for a calibration artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CalibrationArtifactId([u8; 32]);

impl CalibrationArtifactId {
    pub const fn new(bytes: [u8; 32]) -> Result<Self, ContractError> {
        let mut i = 0;
        let mut nonzero = false;
        while i < bytes.len() {
            if bytes[i] != 0 {
                nonzero = true;
            }
            i += 1;
        }
        if !nonzero {
            return Err(ContractError::InvalidCalibration);
        }
        Ok(Self(bytes))
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Calibration status. The calibrated variant requires a validated artifact
/// identifier, but the identifier itself is not cryptographic proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationState {
    Missing,
    Expired,
    OutOfRange,
    Uncalibrated,
    Calibrated { artifact_id: CalibrationArtifactId },
}

/// Validated calibration declaration attached to a sample/stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CalibrationRecord {
    state: CalibrationState,
}

impl CalibrationRecord {
    pub const fn new(state: CalibrationState) -> Result<Self, ContractError> {
        if let CalibrationState::Calibrated { artifact_id } = state {
            let bytes = artifact_id.bytes();
            let mut i = 0;
            let mut nonzero = false;
            while i < bytes.len() {
                if bytes[i] != 0 {
                    nonzero = true;
                }
                i += 1;
            }
            if !nonzero {
                return Err(ContractError::InvalidCalibration);
            }
        }
        Ok(Self { state })
    }

    pub const fn state(self) -> CalibrationState {
        self.state
    }
}

/// Finite normalized quality metrics. Both values are in the closed range 0–1;
/// uncertainty is a normalized fraction of the declared measurand range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityMetrics {
    confidence: f32,
    relative_uncertainty: f32,
}

impl QualityMetrics {
    pub fn new(confidence: f32, relative_uncertainty: f32) -> Result<Self, ContractError> {
        if !confidence.is_finite()
            || !relative_uncertainty.is_finite()
            || !(0.0..=1.0).contains(&confidence)
            || !(0.0..=1.0).contains(&relative_uncertainty)
        {
            return Err(ContractError::InvalidQuality);
        }
        Ok(Self {
            confidence,
            relative_uncertainty,
        })
    }

    pub const fn confidence(self) -> f32 {
        self.confidence
    }

    pub const fn relative_uncertainty(self) -> f32 {
        self.relative_uncertainty
    }
}

/// Measurement quality is independent of source trust and calibration.
/// Unknown is explicit; there is intentionally no default success value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QualityRecord {
    Unknown,
    Known(QualityMetrics),
}

impl QualityRecord {
    pub const fn unknown() -> Self {
        Self::Unknown
    }

    pub const fn known(metrics: QualityMetrics) -> Self {
        Self::Known(metrics)
    }
}

#[cfg(test)]
#[path = "quality_tests.rs"]
mod tests;
