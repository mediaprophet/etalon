use super::*;
use etalon_sense_core::time::ClockStamp;

#[derive(Clone, Copy)]
struct Provider {
    metadata: HostNfcMetadata,
    bytes: [u8; MAX_NFC_UID_BYTES],
    fail: bool,
    calls: u32,
}

impl NfcProvider for Provider {
    fn read_next(
        &mut self,
        out: &mut [u8; MAX_NFC_UID_BYTES],
    ) -> Result<HostNfcMetadata, NfcProviderError> {
        self.calls += 1;
        out.copy_from_slice(&self.bytes);
        if self.fail {
            return Err(NfcProviderError::Unavailable);
        }
        Ok(self.metadata)
    }
}

struct Gate {
    allowed: bool,
    calls: u32,
}
impl NfcPolicyGate for Gate {
    fn approve(&mut self, _: u64, operation: NfcOperation) -> Result<(), NfcError> {
        self.calls += 1;
        assert_eq!(operation, NfcOperation::ReadOneObservation);
        if self.allowed {
            Ok(())
        } else {
            Err(NfcError::PolicyDenied)
        }
    }
}

fn metadata() -> HostNfcMetadata {
    HostNfcMetadata {
        source_id: 9,
        stream_id: 4,
        generation: 2,
        sequence: 17,
        gap_before: Some(1),
        uid_len: 7,
        protocol: NfcProtocol::Iso14443A,
        tag_kind: NfcTagKind::Type4,
        proximity: NfcProximity::HostReportedPresent,
        timestamp: Some(ClockStamp::new(8, 1, 100, 3).unwrap()),
    }
}

fn provider() -> Provider {
    let mut bytes = [0u8; MAX_NFC_UID_BYTES];
    bytes[..7].copy_from_slice(&[4, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66]);
    Provider {
        metadata: metadata(),
        bytes,
        fail: false,
        calls: 0,
    }
}

fn sentinel() -> NfcObservation {
    let mut p = provider();
    let mut g = Gate {
        allowed: true,
        calls: 0,
    };
    let mut out = NfcObservation::new(metadata(), &[0xab; 7]).unwrap();
    p.metadata.uid_len = 1;
    let _ = capture_next(Some(&mut g), &mut p, 9, &mut out);
    out
}

#[test]
fn accepted_record_preserves_uid_metadata_gaps_and_mapped_time() {
    let mut p = provider();
    let mut g = Gate {
        allowed: true,
        calls: 0,
    };
    let mut out = sentinel();
    capture_next(Some(&mut g), &mut p, 9, &mut out).unwrap();
    assert_eq!(out.uid(), &[4, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66]);
    assert_eq!(out.metadata().sequence, 17);
    assert_eq!(out.metadata().gap_before, Some(1));
    assert_eq!(out.metadata().protocol, NfcProtocol::Iso14443A);
    assert_eq!(
        out.metadata().timestamp,
        Some(ClockStamp::new(8, 1, 100, 3).unwrap())
    );
    assert_eq!(g.calls, 2);
}

#[test]
fn denied_gate_does_not_call_provider_or_change_output() {
    let mut p = provider();
    let mut g = Gate {
        allowed: false,
        calls: 0,
    };
    let mut out = sentinel();
    let before = out;
    assert_eq!(
        capture_next(Some(&mut g), &mut p, 9, &mut out),
        Err(NfcError::PolicyDenied)
    );
    assert_eq!(p.calls, 0);
    assert_eq!(out, before);
}

#[test]
fn invalid_length_or_source_mismatch_is_transactional() {
    let mut p = provider();
    let mut g = Gate {
        allowed: true,
        calls: 0,
    };
    let mut out = sentinel();
    let before = out;
    p.metadata.uid_len = (MAX_NFC_UID_BYTES + 1) as u8;
    assert_eq!(
        capture_next(Some(&mut g), &mut p, 9, &mut out),
        Err(NfcError::InvalidUidLength(33))
    );
    assert_eq!(out, before);
    p.metadata = metadata();
    p.metadata.source_id = 10;
    assert_eq!(
        capture_next(Some(&mut g), &mut p, 9, &mut out),
        Err(NfcError::InvalidSource)
    );
    assert_eq!(out, before);
}

#[test]
fn unknown_time_and_proximity_remain_unknown() {
    let mut p = provider();
    p.metadata.timestamp = None;
    p.metadata.proximity = NfcProximity::Unknown;
    let mut out = sentinel();
    capture_next::<Gate, _>(None, &mut p, 9, &mut out).unwrap();
    assert_eq!(out.metadata().timestamp, None);
    assert_eq!(out.metadata().proximity, NfcProximity::Unknown);
}

#[test]
fn observation_constructor_rejects_inconsistent_lengths() {
    assert_eq!(
        NfcObservation::new(metadata(), &[1, 2]),
        Err(NfcError::UidLengthMismatch)
    );
    let mut invalid = metadata();
    invalid.stream_id = 0;
    assert_eq!(
        NfcObservation::new(invalid, &[0; 7]),
        Err(NfcError::InvalidStream)
    );
}

struct RevokingGate {
    calls: u8,
}
impl NfcPolicyGate for RevokingGate {
    fn approve(&mut self, _: u64, _: NfcOperation) -> Result<(), NfcError> {
        self.calls += 1;
        if self.calls == 1 {
            Ok(())
        } else {
            Err(NfcError::PolicyDenied)
        }
    }
}

#[test]
fn policy_revocation_during_read_does_not_publish_uid() {
    let mut p = provider();
    let mut gate = RevokingGate { calls: 0 };
    let mut out = sentinel();
    let before = out;
    assert_eq!(
        capture_next(Some(&mut gate), &mut p, 9, &mut out),
        Err(NfcError::PolicyDenied)
    );
    assert_eq!(gate.calls, 2);
    assert_eq!(p.calls, 1);
    assert_eq!(out, before);
}

struct IdentifierGate;
impl NfcPolicyGate for IdentifierGate {
    fn approve(&mut self, _: u64, _: NfcOperation) -> Result<(), NfcError> {
        Ok(())
    }
    fn approve_observation(&mut self, observation: &NfcObservation) -> Result<(), NfcError> {
        if observation.uid().first() == Some(&4) {
            Err(NfcError::PolicyDenied)
        } else {
            Ok(())
        }
    }
}

#[test]
fn identifier_specific_gate_can_reject_after_read() {
    let mut p = provider();
    let mut out = sentinel();
    let before = out;
    assert_eq!(
        capture_next(Some(&mut IdentifierGate), &mut p, 9, &mut out),
        Err(NfcError::PolicyDenied)
    );
    assert_eq!(out, before);
}
