use super::*;
use crate::{
    adapter::{
        AdapterDescriptor, AdapterId, AuthorizedCaptureBuffer, CaptureBuffer, CapturePermit,
        DiscoveredSource,
    },
    buffer::{BufferToken, SampleLayout, SampleRegion},
    capability::Modality,
    policy::{EvidenceCommitment, PolicyOutcome, RevocationSnapshot},
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind},
    quantity::{PhysicalQuantity, SampleEncoding, Unit},
    time::ClockStamp,
};
use core::cell::Cell;

fn id(value: u64) -> PolicyId { PolicyId::new(value).unwrap() }

fn capability() -> SensorCapability {
    SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexI16,
        [2, 0, 0, 0],
        None,
    ).unwrap()
}

fn stamp(ticks: u64) -> ClockStamp { ClockStamp::new(7, 2, ticks, 0).unwrap() }

fn policy_request(operation: u64, purpose: u64) -> PolicyRequest {
    let mut evidence = [0u8; 32];
    evidence[0] = 1;
    PolicyRequest::new(
        None, None, id(301), id(operation), id(99), id(101), id(400 + purpose), id(90),
        EvidenceCommitment::new(evidence).unwrap(),
    )
}

fn binding() -> CapturePolicyBinding {
    CapturePolicyBinding::new(
        15,
        id(301),
        CaptureCapability::BuiltIn(capability()),
        4,
        policy_request(10, 4),
        policy_request(11, 4),
    ).unwrap()
}

fn authorization() -> AuthorizedCapture {
    let source = DiscoveredSource::new(15, capability()).unwrap();
    let permit = CapturePermit::after_authorization(15, 4, capability(), stamp(100)).unwrap();
    AuthorizedCapture::new(source, permit, stamp(5)).unwrap()
}

fn commitment() -> DecisionCommitment {
    let mut bytes = [0u8; 32];
    bytes[0] = 5;
    DecisionCommitment::new(bytes).unwrap()
}

struct PolicyHost { epoch: Cell<u64>, advance_epoch: bool, deny: bool }

impl PolicyEvaluator for PolicyHost {
    fn evaluate(&self, _request: &PolicyRequest) -> PolicyOutcome {
        if self.deny { PolicyOutcome::Deny } else { PolicyOutcome::Permit(commitment()) }
    }
}

impl PolicyEpochProvider for PolicyHost {
    fn current_policy_epoch(&self, _request: &PolicyRequest) -> Option<u64> {
        let current = self.epoch.get();
        if self.advance_epoch { self.epoch.set(current + 1); }
        Some(current)
    }
}

impl RevocationProvider for PolicyHost {
    fn current_revocation(&self, _request: &PolicyRequest) -> Option<RevocationSnapshot> {
        Some(RevocationSnapshot::new(id(90), 7, false))
    }
}

struct Adapter { starts: usize, captures: usize, stop_calls: usize, stop_failures_remaining: usize }

impl SensorAdapter for Adapter {
    fn descriptor(&self) -> AdapterDescriptor {
        AdapterDescriptor::new(AdapterId::new(3).unwrap(), Modality::Wifi)
    }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> { Ok(stamp(6)) }

    fn discover_backend(&mut self, out: &mut [DiscoveredSource]) -> Result<usize, AdapterError> {
        if out.is_empty() { return Err(AdapterError::DiscoveryBufferTooSmall); }
        out[0] = DiscoveredSource::new(15, capability()).unwrap();
        Ok(1)
    }

    fn start_backend(&mut self, request: AuthorizedCapture) -> Result<CaptureSession, AdapterError> {
        self.starts += 1;
        CaptureSession::new(self.descriptor().id(), request.source(), 4, 2)
    }

    fn capture_backend<'a>(
        &mut self,
        mut output: AuthorizedCaptureBuffer<'a>,
    ) -> Result<crate::adapter::CapturedSamples<'a>, AdapterError> {
        self.captures += 1;
        output.writable_bytes().fill(0);
        Ok(output.complete())
    }

    fn stop_backend(&mut self, _session: CaptureSession) -> Result<(), AdapterError> {
        self.stop_calls += 1;
        if self.stop_failures_remaining > 0 {
            self.stop_failures_remaining -= 1;
            return Err(AdapterError::BackendFailure);
        }
        Ok(())
    }
}

#[test]
fn guarded_start_rechecks_policy_state_immediately_before_adapter_start() {
    let mut adapter = Adapter { starts: 0, captures: 0, stop_calls: 0, stop_failures_remaining: 0 };
    let host = PolicyHost { epoch: Cell::new(4), advance_epoch: false, deny: false };
    let session = guarded_start_builtin(
        &mut adapter,
        authorization(),
        binding(),
        policy_request(10, 4),
        &host,
        &host,
        &host,
    ).unwrap();
    assert_eq!(session.source_id(), 15);
    assert_eq!(session.generation(), 2);
    assert_eq!(adapter.starts, 1);
}

#[test]
fn policy_epoch_change_blocks_start_before_backend_is_called() {
    let mut adapter = Adapter { starts: 0, captures: 0, stop_calls: 0, stop_failures_remaining: 0 };
    let host = PolicyHost { epoch: Cell::new(4), advance_epoch: true, deny: false };
    let result = guarded_start_builtin(
        &mut adapter,
        authorization(),
        binding(),
        policy_request(10, 4),
        &host,
        &host,
        &host,
    );
    assert_eq!(result, Err(GuardedCaptureError::Policy(PolicyError::StateChanged)));
    assert_eq!(adapter.starts, 0);
}

#[test]
fn request_scope_mismatch_fails_before_policy_or_adapter_calls() {
    let mut adapter = Adapter { starts: 0, captures: 0, stop_calls: 0, stop_failures_remaining: 0 };
    let host = PolicyHost { epoch: Cell::new(4), advance_epoch: false, deny: false };
    let result = guarded_start_builtin(
        &mut adapter,
        authorization(),
        binding(),
        policy_request(10, 5),
        &host,
        &host,
        &host,
    );
    assert_eq!(result, Err(GuardedCaptureError::Policy(PolicyError::RequestMismatch)));
    assert_eq!(adapter.starts, 0);
}

fn capture_layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::ComplexI16).unwrap()
}

fn capture_metadata(sequence: u64) -> IngestMetadata {
    IngestMetadata::new(
        SampleRegion::new(BufferToken::new(100, 1).unwrap(), 0, 8, 8, 2).unwrap(),
        SourceDescriptor::new(15, SourceKind::Live).unwrap(),
        stamp(1_000 + sequence),
        None,
        CalibrationRecord::new(CalibrationState::Missing).unwrap(),
        QualityRecord::Unknown,
        sequence,
    )
}

#[test]
fn guarded_capture_returns_frame_with_exact_policy_commitment() {
    let mut adapter = Adapter { starts: 0, captures: 0, stop_calls: 0, stop_failures_remaining: 0 };
    let host = PolicyHost { epoch: Cell::new(4), advance_epoch: false, deny: false };
    let session = guarded_start_builtin(
        &mut adapter, authorization(), binding(), policy_request(10, 4), &host, &host, &host,
    ).unwrap();
    let mut bytes = [0u8; 8];
    let buffer = CaptureBuffer::new(&mut bytes, capture_layout(), capability()).unwrap();
    let mut stream = IngestStream::new(15, 25, session.generation()).unwrap();

    let guarded = guarded_capture_and_ingest(
        &mut adapter,
        &session,
        authorization(),
        buffer,
        &mut stream,
        capture_metadata(1),
        &host,
        &host,
        &host,
    ).unwrap();

    assert_eq!(guarded.frame().payload(), &[0; 8]);
    assert_eq!(guarded.policy_request(), &policy_request(11, 4));
    assert_eq!(guarded.decision_commitment(), &commitment());
    assert_eq!(guarded.frame().measurement().source().trust_state(), crate::quality::TrustState::Unverified);
    assert_eq!(adapter.captures, 1);
}

#[test]
fn denied_guarded_capture_does_not_call_backend() {
    let mut adapter = Adapter { starts: 0, captures: 0, stop_calls: 0, stop_failures_remaining: 0 };
    let permit_host = PolicyHost { epoch: Cell::new(4), advance_epoch: false, deny: false };
    let session = guarded_start_builtin(
        &mut adapter, authorization(), binding(), policy_request(10, 4),
        &permit_host, &permit_host, &permit_host,
    ).unwrap();
    let denied_host = PolicyHost { epoch: Cell::new(4), advance_epoch: false, deny: true };
    let mut bytes = [0u8; 8];
    let buffer = CaptureBuffer::new(&mut bytes, capture_layout(), capability()).unwrap();
    let mut stream = IngestStream::new(15, 25, session.generation()).unwrap();

    let result = guarded_capture_and_ingest(
        &mut adapter,
        &session,
        authorization(),
        buffer,
        &mut stream,
        capture_metadata(1),
        &denied_host,
        &denied_host,
        &denied_host,
    );

    assert_eq!(result, Err(GuardedCaptureError::Policy(PolicyError::Denied)));
    assert_eq!(adapter.captures, 0);
}

#[test]
fn guarded_stop_retains_session_wrapper_for_retry_after_backend_failure() {
    let mut adapter = Adapter {
        starts: 0,
        captures: 0,
        stop_calls: 0,
        stop_failures_remaining: 1,
    };
    let host = PolicyHost { epoch: Cell::new(4), advance_epoch: false, deny: false };
    let session = guarded_start_builtin(
        &mut adapter,
        authorization(),
        binding(),
        policy_request(10, 4),
        &host,
        &host,
        &host,
    ).unwrap();

    assert_eq!(guarded_stop_builtin(&mut adapter, &session), Err(GuardedCaptureError::Adapter(AdapterError::BackendFailure)));
    assert_eq!(session.source_id(), 15);
    assert_eq!(guarded_stop_builtin(&mut adapter, &session), Ok(()));
    assert_eq!(adapter.stop_calls, 2);
}
