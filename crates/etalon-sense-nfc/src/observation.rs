//! One-event host-fed NFC input contract.

use etalon_sense_core::time::ClockStamp;

/// Fixed UID capacity; the actual UID length is always carried separately.
pub const MAX_NFC_UID_BYTES: usize = 32;

/// Protocol reported by the host/provider. `HostDeclared*` variants are not
/// independently negotiated or verified by this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NfcProtocol {
    Unknown,
    Iso14443A,
    Iso14443B,
    Felica,
    Iso15693,
    NfcDep,
    HostDeclared(u16),
}

/// Tag/device category reported by the provider, not inferred from UID bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NfcTagKind {
    Unknown,
    Type1,
    Type2,
    Type3,
    Type4,
    Type5,
    Peer,
    HostDeclared(u16),
}

/// Reader-reported presence only. This is not a measured distance or proof of
/// a person's/object's identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NfcProximity {
    Unknown,
    HostReportedPresent,
    HostReportedAbsent,
}

/// Metadata supplied by the host for the next UID written by the provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostNfcMetadata {
    pub source_id: u64,
    pub stream_id: u64,
    pub generation: u32,
    pub sequence: u64,
    /// Unknown gap count stays `None`; it is never repaired here.
    pub gap_before: Option<u64>,
    /// Number of UID bytes the provider wrote to the supplied fixed buffer.
    pub uid_len: u8,
    pub protocol: NfcProtocol,
    pub tag_kind: NfcTagKind,
    pub proximity: NfcProximity,
    /// A mapped Etalon clock stamp, if the host already has one. `None` means
    /// time is unknown; raw platform ticks must not be placed here unchecked.
    pub timestamp: Option<ClockStamp>,
}

/// One validated host declaration. Validation establishes bounds and required
/// identifiers only; it does not authenticate bytes, source, protocol, time,
/// tag identity, permissions, or physical acquisition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NfcObservation {
    metadata: HostNfcMetadata,
    uid: [u8; MAX_NFC_UID_BYTES],
}

impl NfcObservation {
    /// Build a bounded host declaration. This does not authenticate its data.
    pub fn new(metadata: HostNfcMetadata, uid: &[u8]) -> Result<Self, NfcError> {
        if metadata.source_id == 0 {
            return Err(NfcError::InvalidSource);
        }
        if metadata.stream_id == 0 {
            return Err(NfcError::InvalidStream);
        }
        if metadata.generation == 0 {
            return Err(NfcError::InvalidGeneration);
        }
        if metadata.uid_len == 0 || metadata.uid_len as usize > MAX_NFC_UID_BYTES {
            return Err(NfcError::InvalidUidLength(metadata.uid_len));
        }
        if uid.len() != metadata.uid_len as usize {
            return Err(NfcError::UidLengthMismatch);
        }
        let mut bytes = [0; MAX_NFC_UID_BYTES];
        bytes[..uid.len()].copy_from_slice(uid);
        Ok(Self {
            metadata,
            uid: bytes,
        })
    }

    pub const fn metadata(self) -> HostNfcMetadata {
        self.metadata
    }
    pub const fn uid_len(self) -> u8 {
        self.metadata.uid_len
    }

    /// Return only the declared UID prefix. UID remains an opaque label and may
    /// be random, rotating, duplicated, spoofed, or privacy-sensitive.
    pub fn uid(&self) -> &[u8] {
        &self.uid[..self.metadata.uid_len as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NfcOperation {
    ReadOneObservation,
}

/// Optional host policy boundary. It must make its own current decision for
/// this operation. The adapter itself does not prescribe a human-subject,
/// guardian, scientific, or public-data policy model. Passing no gate leaves
/// policy to the host composition and does not assert that access was allowed.
pub trait NfcPolicyGate {
    fn approve(&mut self, source_id: u64, operation: NfcOperation) -> Result<(), NfcError>;

    /// Recheck this exact result. Hosts with identifier-specific rules must
    /// override this method and apply them before the observation is exposed.
    fn approve_observation(&mut self, observation: &NfcObservation) -> Result<(), NfcError> {
        self.approve(
            observation.metadata().source_id,
            NfcOperation::ReadOneObservation,
        )
    }
}

/// Host-defined input callback. It must write at most the provided fixed UID
/// buffer and return the number of meaningful bytes in `HostNfcMetadata`.
/// This trait is an adapter seam, not a hardware implementation.
pub trait NfcProvider {
    fn read_next(
        &mut self,
        uid_output: &mut [u8; MAX_NFC_UID_BYTES],
    ) -> Result<HostNfcMetadata, NfcProviderError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NfcProviderError {
    Unavailable,
    EndOfStream,
    Fault(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NfcError {
    PolicyDenied,
    Provider(NfcProviderError),
    InvalidSource,
    InvalidStream,
    InvalidGeneration,
    InvalidUidLength(u8),
    UidLengthMismatch,
}

/// Check optional host policy, request one event, validate it, recheck policy,
/// then replace
/// `output` atomically. Provider writes are bounded by a fixed stack buffer;
/// malformed metadata or provider failure leaves the prior output unchanged.
/// No capability, hardware provenance, or trusted-policy evidence is minted.
/// A blocking provider cannot be interrupted by a policy change in flight.
pub fn capture_next<G: NfcPolicyGate, P: NfcProvider>(
    mut gate: Option<&mut G>,
    provider: &mut P,
    source_id: u64,
    output: &mut NfcObservation,
) -> Result<(), NfcError> {
    if source_id == 0 {
        return Err(NfcError::InvalidSource);
    }
    if let Some(gate) = gate.as_deref_mut() {
        gate.approve(source_id, NfcOperation::ReadOneObservation)?;
    }

    let mut uid = [0u8; MAX_NFC_UID_BYTES];
    let metadata = provider.read_next(&mut uid).map_err(NfcError::Provider)?;
    if metadata.source_id != source_id {
        return Err(NfcError::InvalidSource);
    }
    if metadata.uid_len == 0 || metadata.uid_len as usize > MAX_NFC_UID_BYTES {
        return Err(NfcError::InvalidUidLength(metadata.uid_len));
    }
    let observation = NfcObservation::new(metadata, &uid[..metadata.uid_len as usize])?;
    if let Some(gate) = gate {
        gate.approve_observation(&observation)?;
    }
    *output = observation;
    Ok(())
}
