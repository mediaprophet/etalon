use super::{
    AdapterDescriptor, AdapterError, AdapterId, AuthorizedCapture, AuthorizedCaptureBuffer,
    CaptureBuffer, CapturePermit, CaptureSession, DiscoveredSource, SensorAdapter,
};
use crate::{
    buffer::SampleLayout,
    capability::{Modality, SensorCapability},
    quantity::{PhysicalQuantity, SampleEncoding, Unit},
    time::ClockStamp,
    ContractError,
};

fn capability(modality: Modality) -> SensorCapability {
    let (quantity, unit, encoding) = match modality {
        Modality::Microphone => (PhysicalQuantity::AcousticPressure, Unit::Pascal, SampleEncoding::F32),
        _ => (PhysicalQuantity::RfChannelResponse, Unit::Unitless, SampleEncoding::ComplexI16),
    };
    SensorCapability::new(modality, quantity, unit, encoding, [8, 0, 0, 0], None).unwrap()
}

fn stamp(clock: u64, epoch: u32, ticks: u64) -> ClockStamp {
    ClockStamp::new(clock, epoch, ticks, 0).unwrap()
}

fn source(modality: Modality) -> DiscoveredSource {
    DiscoveredSource::new(15, capability(modality)).unwrap()
}

fn permit(source: DiscoveredSource, expires: ClockStamp) -> CapturePermit {
    CapturePermit::after_authorization(15, 4, source.capability(), expires).unwrap()
}

fn audio_layout(extent: u32) -> SampleLayout {
    SampleLayout::new(1, [extent, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::F32).unwrap()
}

struct ContractFixture {
    descriptor: AdapterDescriptor,
    source: DiscoveredSource,
    trusted_clock: ClockStamp,
}

impl SensorAdapter for ContractFixture {
    fn descriptor(&self) -> AdapterDescriptor {
        self.descriptor
    }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        Ok(self.trusted_clock)
    }

    fn discover_backend(&mut self, out: &mut [DiscoveredSource]) -> Result<usize, AdapterError> {
        if out.is_empty() {
            return Err(AdapterError::DiscoveryBufferTooSmall);
        }
        out[0] = self.source;
        Ok(1)
    }

    fn start_backend(&mut self, request: AuthorizedCapture) -> Result<CaptureSession, AdapterError> {
        CaptureSession::new(self.descriptor.id(), request.source(), 21, 2)
    }

    fn capture_backend<'a>(
        &mut self,
        mut output: AuthorizedCaptureBuffer<'a>,
    ) -> Result<super::CapturedSamples<'a>, AdapterError> {
        output.writable_bytes().fill(0x5a);
        Ok(output.complete())
    }

    fn stop_backend(&mut self, _session: CaptureSession) -> Result<(), AdapterError> {
        Ok(())
    }
}

#[test]
fn adapter_and_source_ids_reject_zero() {
    assert_eq!(AdapterId::new(0), Err(AdapterError::InvalidIdentifier));
    assert_eq!(
        DiscoveredSource::new(0, capability(Modality::Microphone)),
        Err(AdapterError::InvalidIdentifier)
    );
    assert_eq!(
        CaptureSession::new(
            AdapterId::new(1).unwrap(),
            source(Modality::Microphone),
            0,
            1,
        ),
        Err(AdapterError::InvalidIdentifier)
    );
}

#[test]
fn capture_buffer_rejects_bad_shape_encoding_and_capacity_before_writes() {
    let cap = capability(Modality::Microphone);
    let mut short = [0u8; 7];
    assert_eq!(
        CaptureBuffer::new(&mut short, audio_layout(2), cap).err(),
        Some(AdapterError::Buffer(ContractError::OutOfBounds))
    );

    let mut enough = [0u8; 32];
    let too_many = SampleLayout::new(1, [9, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::F32).unwrap();
    assert_eq!(
        CaptureBuffer::new(&mut enough, too_many, cap).err(),
        Some(AdapterError::Buffer(ContractError::InvalidExtent))
    );
    let wrong_encoding = SampleLayout::new(1, [2, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::I16).unwrap();
    assert_eq!(
        CaptureBuffer::new(&mut enough, wrong_encoding, cap).err(),
        Some(AdapterError::Buffer(ContractError::IncompatibleEncoding))
    );

    let rank_two_capability = SensorCapability::new(
        Modality::Microphone,
        PhysicalQuantity::AcousticPressure,
        Unit::Pascal,
        SampleEncoding::F32,
        [8, 8, 0, 0],
        None,
    )
    .unwrap();
    assert_eq!(
        CaptureBuffer::new(&mut enough, audio_layout(2), rank_two_capability).err(),
        Some(AdapterError::Buffer(ContractError::InvalidExtent))
    );
}

#[test]
fn authorization_binds_source_capability_purpose_and_expiry_clock() {
    let declared = source(Modality::Microphone);
    let request = AuthorizedCapture::new(
        declared,
        permit(declared, stamp(5, 1, 500)),
        stamp(5, 1, 100),
    )
    .unwrap();
    assert_eq!(request.source(), declared);
    assert_eq!(request.purpose_id(), 4);
    assert_eq!(request.authorized_at(), stamp(5, 1, 100));

    assert_eq!(
        AuthorizedCapture::new(
            declared,
            permit(declared, stamp(5, 1, 100)),
            stamp(5, 1, 100),
        ),
        Err(AdapterError::AuthorizationExpired)
    );
    assert_eq!(
        AuthorizedCapture::new(
            declared,
            permit(declared, stamp(6, 1, 500)),
            stamp(5, 1, 100),
        ),
        Err(AdapterError::ClockDomainMismatch)
    );
    let different = DiscoveredSource::new(16, declared.capability()).unwrap();
    assert_eq!(
        AuthorizedCapture::new(different, permit(declared, stamp(5, 1, 500)), stamp(5, 1, 100)),
        Err(AdapterError::AuthorizationDenied)
    );
}

#[test]
fn capture_request_binds_adapter_session_and_validated_buffer() {
    let declared = source(Modality::Microphone);
    let descriptor = AdapterDescriptor::new(AdapterId::new(3).unwrap(), Modality::Microphone);
    let session = CaptureSession::new(descriptor.id(), declared, 11, 1).unwrap();
    let authorized = AuthorizedCapture::new(
        declared,
        permit(declared, stamp(1, 0, 500)),
        stamp(1, 0, 100),
    )
    .unwrap();
    let mut bytes = [0u8; 32];
    let output = CaptureBuffer::new(&mut bytes, audio_layout(2), declared.capability()).unwrap();
    let authorized_output =
        AuthorizedCaptureBuffer::new(descriptor, session, authorized, output).unwrap();
    assert_eq!(authorized_output.source(), declared);
    assert_eq!(authorized_output.purpose_id(), 4);

    let wrong_adapter = AdapterDescriptor::new(AdapterId::new(4).unwrap(), Modality::Microphone);
    let authorized = AuthorizedCapture::new(
        declared,
        permit(declared, stamp(1, 0, 500)),
        stamp(1, 0, 100),
    )
    .unwrap();
    let mut bytes = [0u8; 32];
    let output = CaptureBuffer::new(&mut bytes, audio_layout(2), declared.capability()).unwrap();
    assert_eq!(
        AuthorizedCaptureBuffer::new(wrong_adapter, session, authorized, output).err(),
        Some(AdapterError::InvalidSession)
    );
}

#[test]
fn checked_trait_wrappers_cover_discovery_start_capture_and_stop() {
    let declared = source(Modality::Microphone);
    let descriptor = AdapterDescriptor::new(AdapterId::new(3).unwrap(), Modality::Microphone);
    let mut adapter = ContractFixture {
        descriptor,
        source: declared,
        trusted_clock: stamp(1, 0, 100),
    };
    let mut discovered = [declared; 1];
    assert_eq!(adapter.discover(&mut discovered), Ok(1));

    let request = AuthorizedCapture::new(
        declared,
        permit(declared, stamp(1, 0, 500)),
        stamp(1, 0, 100),
    )
    .unwrap();
    let session = adapter.start(request).unwrap();
    adapter.trusted_clock = stamp(1, 0, 200);

    let request = AuthorizedCapture::new(
        declared,
        permit(declared, stamp(1, 0, 500)),
        stamp(1, 0, 200),
    )
    .unwrap();
    let mut bytes = [0u8; 32];
    let buffer = CaptureBuffer::new(&mut bytes, audio_layout(2), declared.capability()).unwrap();
    let authorized = AuthorizedCaptureBuffer::new(descriptor, session, request, buffer).unwrap();
    let captured = adapter.capture(authorized).unwrap();
    assert_eq!(captured.bytes(), &[0x5a; 8]);
    assert_eq!(adapter.stop(session), Ok(()));
}

#[test]
fn discovery_rejects_truncation_and_wrong_modality() {
    let declared = source(Modality::Microphone);
    let descriptor = AdapterDescriptor::new(AdapterId::new(3).unwrap(), Modality::Microphone);
    let mut adapter = ContractFixture {
        descriptor,
        source: declared,
        trusted_clock: stamp(1, 0, 100),
    };
    let mut empty = [];
    assert_eq!(
        adapter.discover(&mut empty),
        Err(AdapterError::DiscoveryBufferTooSmall)
    );

    adapter.source = source(Modality::Wifi);
    let mut output = [declared; 1];
    assert_eq!(adapter.discover(&mut output), Err(AdapterError::Unsupported));
}

#[test]
fn trusted_adapter_clock_rejects_request_expired_after_authorization() {
    let declared = source(Modality::Microphone);
    let descriptor = AdapterDescriptor::new(AdapterId::new(3).unwrap(), Modality::Microphone);
    let mut adapter = ContractFixture {
        descriptor,
        source: declared,
        trusted_clock: stamp(1, 0, 500),
    };
    let request = AuthorizedCapture::new(
        declared,
        permit(declared, stamp(1, 0, 500)),
        stamp(1, 0, 100),
    )
    .unwrap();
    assert_eq!(adapter.start(request), Err(AdapterError::AuthorizationExpired));
}
