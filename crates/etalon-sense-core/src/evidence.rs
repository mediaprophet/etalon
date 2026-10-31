use crate::{
    quality::{CalibrationRecord, CalibrationState, SourceDescriptor},
    time::ClockStamp,
    ContractError,
};

/// Hash algorithm named by an artifact digest. This contract stores digests;
/// it does not compute them or establish authenticity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DigestAlgorithm {
    Sha256,
    Blake3,
}

/// Fixed-size artifact digest used for payload, configuration, or signature
/// references. A digest is an integrity identifier, not an authority proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactDigest {
    algorithm: DigestAlgorithm,
    bytes: [u8; 32],
}

impl ArtifactDigest {
    pub const fn new(algorithm: DigestAlgorithm, bytes: [u8; 32]) -> Self {
        Self { algorithm, bytes }
    }

    pub const fn algorithm(self) -> DigestAlgorithm {
        self.algorithm
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.bytes
    }
}

/// Nonzero authority and verification-key identifiers. Their presence alone
/// does not authenticate either identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AuthorityKeyRef {
    issuer_id: u64,
    key_id: u64,
}

impl AuthorityKeyRef {
    pub const fn new(issuer_id: u64, key_id: u64) -> Result<Self, ContractError> {
        if issuer_id == 0 || key_id == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self { issuer_id, key_id })
    }

    pub const fn issuer_id(self) -> u64 {
        self.issuer_id
    }

    pub const fn key_id(self) -> u64 {
        self.key_id
    }
}

/// Evidence origin, with caller assertions kept distinct from successful
/// cryptographic verification. Unknown provenance remains an explicit state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceOrigin {
    Unknown,
    Asserted {
        source: SourceDescriptor,
        assertion_id: u64,
    },
    SignaturePresented {
        source: SourceDescriptor,
        authority: AuthorityKeyRef,
        signed_digest: ArtifactDigest,
        signature_digest: ArtifactDigest,
    },
    Verified(VerifiedOrigin),
}

impl EvidenceOrigin {
    pub fn unknown() -> Self {
        Self::Unknown
    }

    pub const fn asserted(
        source: SourceDescriptor,
        assertion_id: u64,
    ) -> Result<Self, ContractError> {
        if assertion_id == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self::Asserted {
            source,
            assertion_id,
        })
    }

    /// Record that signature material was supplied. This does not verify it.
    pub const fn signature_presented(
        source: SourceDescriptor,
        authority: AuthorityKeyRef,
        signed_digest: ArtifactDigest,
        signature_digest: ArtifactDigest,
    ) -> Self {
        Self::SignaturePresented {
            source,
            authority,
            signed_digest,
            signature_digest,
        }
    }
}

/// Verification result created by an in-crate verifier only after successful
/// cryptographic verification. Identifiers and digest references are not
/// independently authenticated by this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedOrigin {
    source: SourceDescriptor,
    authority: AuthorityKeyRef,
    signed_digest: ArtifactDigest,
    signature_digest: ArtifactDigest,
}

impl VerifiedOrigin {
    /// Call only after a verifier has checked the signature and authority policy.
    pub(crate) const fn after_cryptographic_verification(
        source: SourceDescriptor,
        authority: AuthorityKeyRef,
        signed_digest: ArtifactDigest,
        signature_digest: ArtifactDigest,
    ) -> Self {
        Self {
            source,
            authority,
            signed_digest,
            signature_digest,
        }
    }

    pub const fn source(self) -> SourceDescriptor {
        self.source
    }

    pub const fn authority(self) -> AuthorityKeyRef {
        self.authority
    }

    pub const fn signed_digest(self) -> ArtifactDigest {
        self.signed_digest
    }

    pub const fn signature_digest(self) -> ArtifactDigest {
        self.signature_digest
    }
}

/// Processing provenance for a derived sample. A provider ID is attribution,
/// not proof that the provider executed the operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessingEvidence {
    algorithm_id: u64,
    configuration_digest: Option<ArtifactDigest>,
    provider_id: Option<u64>,
}

impl ProcessingEvidence {
    pub const fn new(
        algorithm_id: u64,
        configuration_digest: Option<ArtifactDigest>,
        provider_id: Option<u64>,
    ) -> Result<Self, ContractError> {
        if algorithm_id == 0 || matches!(provider_id, Some(0)) {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self {
            algorithm_id,
            configuration_digest,
            provider_id,
        })
    }

    pub const fn algorithm_id(self) -> u64 {
        self.algorithm_id
    }

    pub const fn configuration_digest(self) -> Option<ArtifactDigest> {
        self.configuration_digest
    }

    pub const fn provider_id(self) -> Option<u64> {
        self.provider_id
    }
}

/// Fixed-size acquisition and processing receipt. Optional fields preserve
/// unavailable provenance instead of synthesizing identifiers or timestamps.
/// This is a native contract, not a serialized ABI or cryptographic verifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvidenceRecord {
    origin: EvidenceOrigin,
    stream_id: Option<u64>,
    sequence: Option<u64>,
    timestamp: Option<ClockStamp>,
    payload_digest: Option<ArtifactDigest>,
    calibration: CalibrationRecord,
    processing: Option<ProcessingEvidence>,
}

impl EvidenceRecord {
    pub const fn new(
        origin: EvidenceOrigin,
        stream_id: Option<u64>,
        sequence: Option<u64>,
        timestamp: Option<ClockStamp>,
        payload_digest: Option<ArtifactDigest>,
        calibration: CalibrationRecord,
        processing: Option<ProcessingEvidence>,
    ) -> Result<Self, ContractError> {
        if matches!(stream_id, Some(0)) {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self {
            origin,
            stream_id,
            sequence,
            timestamp,
            payload_digest,
            calibration,
            processing,
        })
    }

    /// Receipt with no source, time, digest, or processing claims.
    pub const fn unknown() -> Self {
        Self {
            origin: EvidenceOrigin::Unknown,
            stream_id: None,
            sequence: None,
            timestamp: None,
            payload_digest: None,
            calibration: CalibrationRecord::new(CalibrationState::Missing)
                .expect("Missing calibration is valid"),
            processing: None,
        }
    }

    pub const fn origin(self) -> EvidenceOrigin {
        self.origin
    }

    pub const fn stream_id(self) -> Option<u64> {
        self.stream_id
    }

    pub const fn sequence(self) -> Option<u64> {
        self.sequence
    }

    pub const fn timestamp(self) -> Option<ClockStamp> {
        self.timestamp
    }

    pub const fn payload_digest(self) -> Option<ArtifactDigest> {
        self.payload_digest
    }

    pub const fn calibration(self) -> CalibrationRecord {
        self.calibration
    }

    pub const fn processing(self) -> Option<ProcessingEvidence> {
        self.processing
    }
}

#[cfg(test)]
#[path = "evidence_tests.rs"]
mod tests;
