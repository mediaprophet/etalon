use super::*;
use crate::{
    adapter::{
        AdapterDescriptor, AdapterId, AuthorizedCapture, AuthorizedExtendedCapture, CapturePermit,
        DiscoveredSource, ExtendedAdapterDescriptor, ExtendedCapturePermit,
        ExtendedDiscoveredSource,
    },
    buffer::BufferToken,
    capability::Modality,
    extended_capability::ExtendedCapability,
    quality::{CalibrationState, SourceKind, TrustState},
    quantity::{PhysicalQuantity, SampleEncoding, SampleRate, Unit},
};

fn builtin_capability() -> SensorCapability {
    SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexF32,
        [4, 0, 0, 0],
        Some(SampleRate::new(100.0).unwrap()),
    )
    .unwrap()
}

fn builtin_layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::ComplexF32).unwrap()
}

fn extended_capability() -> ExtendedCapability {
    ExtendedCapability::new(
        801,
        802,
        None,
        SampleEncoding::I16,
        [4, 0, 0, 0],
        Some(SampleRate::new(20.0).unwrap()),
    )
    .unwrap()
}

fn extended_layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [2, 0, 0, 0], SampleEncoding::I16).unwrap()
}

fn stamp(ticks: u64) -> ClockStamp {
    ClockStamp::new(5, 2, ticks, 0).unwrap()
}

fn calibration() -> CalibrationRecord {
    CalibrationRecord::new(CalibrationState::Missing).unwrap()
}

fn metadata(source_id: u64, region: SampleRegion, sequence: u64) -> IngestMetadata {
    IngestMetadata::new(
        region,
        SourceDescriptor::new(source_id, SourceKind::Live).unwrap(),
        stamp(1_000 + sequence),
        None,
        calibration(),
        QualityRecord::Unknown,
        sequence,
    )
}

fn untimed_metadata(source_id: u64, region: SampleRegion, sequence: u64) -> IngestMetadata {
    IngestMetadata::new_with_optional_timestamp(
        region,
        SourceDescriptor::new(source_id, SourceKind::Live).unwrap(),
        None,
        None,
        calibration(),
        QualityRecord::Unknown,
        sequence,
    )
}

fn builtin_region() -> SampleRegion {
    SampleRegion::new(BufferToken::new(10, 1).unwrap(), 0, 16, 16, 4).unwrap()
}

fn extended_region() -> SampleRegion {
    SampleRegion::new(BufferToken::new(11, 1).unwrap(), 0, 4, 4, 2).unwrap()
}

struct BuiltinAdapter {
    calls: u32,
}

impl SensorAdapter for BuiltinAdapter {
    fn descriptor(&self) -> AdapterDescriptor {
        AdapterDescriptor::new(AdapterId::new(1).unwrap(), Modality::Wifi)
    }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        Ok(stamp(20))
    }

    fn discover_backend(&mut self, _out: &mut [DiscoveredSource]) -> Result<usize, AdapterError> {
        Ok(0)
    }

    fn start_backend(
        &mut self,
        _request: AuthorizedCapture,
    ) -> Result<CaptureSession, AdapterError> {
        Err(AdapterError::Unavailable)
    }

    fn capture_backend<'a>(
        &mut self,
        mut output: AuthorizedCaptureBuffer<'a>,
    ) -> Result<CapturedSamples<'a>, AdapterError> {
        self.calls += 1;
        output.writable_bytes().fill(0xA5);
        Ok(output.complete())
    }

    fn stop_backend(&mut self, _session: CaptureSession) -> Result<(), AdapterError> {
        Ok(())
    }
}

struct RegisteredAdapter {
    calls: u32,
}

impl ExtendedSensorAdapter for RegisteredAdapter {
    fn descriptor(&self) -> ExtendedAdapterDescriptor {
        ExtendedAdapterDescriptor::new(AdapterId::new(2).unwrap(), 801).unwrap()
    }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        Ok(stamp(20))
    }

    fn discover_backend(
        &mut self,
        _out: &mut [ExtendedDiscoveredSource],
    ) -> Result<usize, AdapterError> {
        Ok(0)
    }

    fn start_backend(
        &mut self,
        _request: AuthorizedExtendedCapture,
    ) -> Result<ExtendedCaptureSession, AdapterError> {
        Err(AdapterError::Unavailable)
    }

    fn capture_backend<'a>(
        &mut self,
        mut output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<ExtendedCapturedSamples<'a>, AdapterError> {
        self.calls += 1;
        output.writable_bytes().fill(0x5A);
        Ok(output.complete())
    }

    fn stop_backend(&mut self, _session: ExtendedCaptureSession) -> Result<(), AdapterError> {
        Ok(())
    }
}

fn builtin_request() -> (DiscoveredSource, AuthorizedCapture) {
    let capability = builtin_capability();
    let source = DiscoveredSource::new(17, capability).unwrap();
    let permit = CapturePermit::after_authorization(17, 99, capability, stamp(100)).unwrap();
    let request = AuthorizedCapture::new(source, permit, stamp(10)).unwrap();
    (source, request)
}

fn extended_request() -> (ExtendedDiscoveredSource, AuthorizedExtendedCapture) {
    let capability = extended_capability();
    let source = ExtendedDiscoveredSource::new(31, capability).unwrap();
    let permit =
        ExtendedCapturePermit::after_authorization(31, 99, capability, stamp(100)).unwrap();
    let request = AuthorizedExtendedCapture::new(source, permit, stamp(10)).unwrap();
    (source, request)
}

#[test]
fn builtin_authorized_capture_flows_into_ingest_without_trust_upgrade() {
    let (source, request) = builtin_request();
    let session = CaptureSession::new(AdapterId::new(1).unwrap(), source, 50, 3).unwrap();
    let mut bytes = [0u8; 16];
    let buffer = CaptureBuffer::new(&mut bytes, builtin_layout(), builtin_capability()).unwrap();
    let mut adapter = BuiltinAdapter { calls: 0 };
    let mut stream = IngestStream::new(17, 25, 3).unwrap();

    let frame = capture_and_ingest(
        &mut adapter,
        session,
        request,
        buffer,
        &mut stream,
        metadata(17, builtin_region(), 1),
    )
    .unwrap();

    assert_eq!(frame.payload(), &[0xA5; 16]);
    assert_eq!(
        frame.measurement().source().trust_state(),
        TrustState::Unverified
    );
    assert_eq!(frame.measurement().sample_rate(), None);
    assert_eq!(frame.measurement().quality(), QualityRecord::Unknown);
    assert_eq!(adapter.calls, 1);
    assert_eq!(stream.last_sequence(), Some((1, stamp(1_001))));
}

#[test]
fn authorized_capture_can_preserve_an_unknown_timestamp() {
    let (source, request) = builtin_request();
    let session = CaptureSession::new(AdapterId::new(1).unwrap(), source, 50, 3).unwrap();
    let mut bytes = [0u8; 16];
    let buffer = CaptureBuffer::new(&mut bytes, builtin_layout(), builtin_capability()).unwrap();
    let mut adapter = BuiltinAdapter { calls: 0 };
    let mut stream = IngestStream::new(17, 25, 3).unwrap();
    let frame = capture_and_ingest(
        &mut adapter,
        session,
        request,
        buffer,
        &mut stream,
        untimed_metadata(17, builtin_region(), 1),
    )
    .unwrap();

    assert_eq!(frame.measurement().timestamp(), None);
    assert_eq!(stream.last_observation(), Some((1, None)));
    assert_eq!(stream.last_known_timestamp(), None);
}

#[test]
fn capture_preflight_enforces_sequence_without_a_timestamp() {
    assert_eq!(preflight_sequence(None, None, 4, None), Ok(()));
    assert_eq!(preflight_sequence(Some((4, None)), None, 5, None), Ok(()));
    assert_eq!(
        preflight_sequence(Some((4, None)), None, 4, Some(stamp(2_000))),
        Err(CaptureIngestError::Contract(ContractError::InvalidSource)),
    );
    assert_eq!(
        preflight_sequence(Some((4, None)), Some(stamp(2_000)), 5, Some(stamp(1_999))),
        Err(CaptureIngestError::Contract(ContractError::InvalidTime)),
    );
    assert_eq!(
        preflight_sequence(
            Some((4, None)),
            Some(stamp(2_000)),
            5,
            Some(ClockStamp::new(6, 2, 2_001, 0).unwrap()),
        ),
        Err(CaptureIngestError::Contract(
            ContractError::ClockDomainMismatch
        )),
    );
}

#[test]
fn registered_capture_flows_into_extended_ingest_and_keeps_unknown_unit() {
    let (source, request) = extended_request();
    let session = ExtendedCaptureSession::new(AdapterId::new(2).unwrap(), source, 70, 5).unwrap();
    let mut bytes = [0u8; 4];
    let buffer =
        ExtendedCaptureBuffer::new(&mut bytes, extended_layout(), extended_capability()).unwrap();
    let mut adapter = RegisteredAdapter { calls: 0 };
    let mut stream = ExtendedIngestStream::new(31, 61, 5, extended_capability()).unwrap();

    let frame = capture_extended_and_ingest(
        &mut adapter,
        session,
        request,
        buffer,
        &mut stream,
        metadata(31, extended_region(), 4),
    )
    .unwrap();

    assert_eq!(frame.payload(), &[0x5A; 4]);
    assert_eq!(frame.measurement().capability().unit_id(), None);
    assert_eq!(
        frame.measurement().source().trust_state(),
        TrustState::Unverified
    );
    assert_eq!(adapter.calls, 1);
    assert_eq!(stream.last_sequence(), Some((4, stamp(1_004))));
}

#[test]
fn invalid_source_or_sequence_is_rejected_before_adapter_capture() {
    let (source, request) = builtin_request();
    let session = CaptureSession::new(AdapterId::new(1).unwrap(), source, 50, 3).unwrap();
    let mut bytes = [0u8; 16];
    let buffer = CaptureBuffer::new(&mut bytes, builtin_layout(), builtin_capability()).unwrap();
    let mut adapter = BuiltinAdapter { calls: 0 };
    let mut stream = IngestStream::new(17, 25, 3).unwrap();
    let first = [0u8; 16];
    stream
        .ingest(
            &first,
            builtin_layout(),
            builtin_region(),
            SourceDescriptor::new(17, SourceKind::Live).unwrap(),
            stamp(1_001),
            None,
            calibration(),
            QualityRecord::Unknown,
            1,
        )
        .unwrap();

    assert_eq!(
        capture_and_ingest(
            &mut adapter,
            session,
            request,
            buffer,
            &mut stream,
            metadata(18, builtin_region(), 2),
        ),
        Err(CaptureIngestError::SourceMismatch),
    );
    assert_eq!(adapter.calls, 0);
    assert_eq!(stream.last_sequence(), Some((1, stamp(1_001))));
}

#[test]
fn generation_mismatch_is_rejected_without_calling_backend() {
    let (source, request) = builtin_request();
    let session = CaptureSession::new(AdapterId::new(1).unwrap(), source, 50, 4).unwrap();
    let mut bytes = [0u8; 16];
    let buffer = CaptureBuffer::new(&mut bytes, builtin_layout(), builtin_capability()).unwrap();
    let mut adapter = BuiltinAdapter { calls: 0 };
    let mut stream = IngestStream::new(17, 25, 3).unwrap();
    assert_eq!(
        capture_and_ingest(
            &mut adapter,
            session,
            request,
            buffer,
            &mut stream,
            metadata(17, builtin_region(), 1),
        ),
        Err(CaptureIngestError::GenerationMismatch),
    );
    assert_eq!(adapter.calls, 0);
}

#[test]
fn duplicate_sequence_is_rejected_before_backend_capture() {
    let (source, request) = builtin_request();
    let session = CaptureSession::new(AdapterId::new(1).unwrap(), source, 50, 3).unwrap();
    let mut bytes = [0u8; 16];
    let buffer = CaptureBuffer::new(&mut bytes, builtin_layout(), builtin_capability()).unwrap();
    let mut adapter = BuiltinAdapter { calls: 0 };
    let mut stream = IngestStream::new(17, 25, 3).unwrap();
    stream
        .ingest(
            &[0u8; 16],
            builtin_layout(),
            builtin_region(),
            SourceDescriptor::new(17, SourceKind::Live).unwrap(),
            stamp(1_001),
            None,
            calibration(),
            QualityRecord::Unknown,
            1,
        )
        .unwrap();
    assert_eq!(
        capture_and_ingest(
            &mut adapter,
            session,
            request,
            buffer,
            &mut stream,
            metadata(17, builtin_region(), 1),
        ),
        Err(CaptureIngestError::Contract(ContractError::InvalidSource)),
    );
    assert_eq!(adapter.calls, 0);
    assert_eq!(stream.last_sequence(), Some((1, stamp(1_001))));
}
