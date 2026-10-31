//! Clear caller samples on every unsuccessful capture, including unwind.
use core::ops::{Deref, DerefMut};
use etalon_sense_core::adapter::{AuthorizedExtendedCaptureBuffer, ExtendedCapturedSamples};

pub(super) struct CaptureOutput<'a> {
    buffer: Option<AuthorizedExtendedCaptureBuffer<'a>>,
}

impl<'a> CaptureOutput<'a> {
    pub(super) fn new(buffer: AuthorizedExtendedCaptureBuffer<'a>) -> Self {
        Self { buffer: Some(buffer) }
    }

    pub(super) fn finish(mut self) -> ExtendedCapturedSamples<'a> {
        self.buffer.take().expect("live capture buffer").complete()
    }
}

impl<'a> Deref for CaptureOutput<'a> {
    type Target = AuthorizedExtendedCaptureBuffer<'a>;
    fn deref(&self) -> &Self::Target { self.buffer.as_ref().expect("live capture buffer") }
}

impl DerefMut for CaptureOutput<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target { self.buffer.as_mut().expect("live capture buffer") }
}

impl Drop for CaptureOutput<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = &mut self.buffer { buffer.writable_bytes().fill(0); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use etalon_sense_core::{
        adapter::{AdapterError, AdapterId, AuthorizedExtendedCaptureBuffer,
            ExtendedAdapterDescriptor, ExtendedCaptureBuffer, ExtendedCaptureSession,
            ExtendedDiscoveredSource, ExtendedSensorAdapter},
        authority::{AuthorizedCaptureRequest, CaptureAuthority, CaptureAuthorizationRequest,
            CapturePolicyDecision, CaptureSource, HostCapturePolicy, TrustedCaptureClock},
        buffer::SampleLayout,
        extended_capability::ExtendedCapability,
        quantity::SampleEncoding,
        time::ClockStamp,
    };

    struct Policy;
    impl HostCapturePolicy for Policy {
        fn decide(&self, _: &CaptureAuthorizationRequest, _: ClockStamp) -> CapturePolicyDecision {
            CapturePolicyDecision::Permit { expires_at: ClockStamp::new(1, 1, 100, 0).unwrap() }
        }
    }
    struct Clock;
    impl TrustedCaptureClock for Clock {
        fn now(&mut self) -> Result<ClockStamp, AdapterError> { Ok(ClockStamp::new(1, 1, 1, 0).unwrap()) }
    }

    fn authorized(bytes: &mut [u8]) -> AuthorizedExtendedCaptureBuffer<'_> {
        let capability = ExtendedCapability::new(1, 2, None, SampleEncoding::I16,
            [2, 1, 0, 0], None).unwrap();
        let source = ExtendedDiscoveredSource::new(3, capability).unwrap();
        let descriptor = ExtendedAdapterDescriptor::new(AdapterId::new(4).unwrap(), 1).unwrap();
        let session = ExtendedCaptureSession::new(descriptor.id(), source, 5, 1).unwrap();
        let request = CaptureAuthorizationRequest::new(CaptureSource::Extended(source), None,
            None, 6, 7).unwrap();
        let request = match CaptureAuthority::issue(request, &Policy, &mut Clock).unwrap().into_capture_request() {
            AuthorizedCaptureRequest::Extended(request) => request,
            _ => unreachable!(),
        };
        let layout = SampleLayout::new(2, [2, 1, 0, 0], [2, 2, 0, 0], SampleEncoding::I16).unwrap();
        let buffer = ExtendedCaptureBuffer::new(bytes, layout, capability).unwrap();
        AuthorizedExtendedCaptureBuffer::new(descriptor, session, request, buffer).unwrap()
    }

    #[test]
    fn unwind_wipes_only_authorized_span_and_preserves_tail() {
        let mut bytes = [0x55; 8];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = CaptureOutput::new(authorized(&mut bytes));
            guard.writable_bytes().fill(0xa5);
            panic!("capture clock or backend unwind");
        }));
        assert!(result.is_err());
        assert_eq!(bytes, [0, 0, 0, 0, 0x55, 0x55, 0x55, 0x55]);
    }

    #[test]
    fn successful_completion_retains_samples_without_wiping() {
        let mut bytes = [0x55; 8];
        let mut guard = CaptureOutput::new(authorized(&mut bytes));
        guard.writable_bytes().fill(0xa5);
        let completed = guard.finish();
        assert_eq!(completed.bytes(), &[0xa5; 4]);
    }

    struct FailingClock { panic: bool }
    impl TrustedCaptureClock for FailingClock {
        fn now(&mut self) -> Result<ClockStamp, AdapterError> {
            if self.panic { panic!("clock failure"); }
            Err(AdapterError::BackendFailure)
        }
    }

    #[test]
    fn outer_capture_clock_failure_clears_output_and_remainder() {
        for panic in [false, true] {
            let mut bytes = [0x55; 8];
            let buffer = authorized(&mut bytes);
            let config = crate::WasapiConfig::new(AdapterId::new(4).unwrap(), 1,
                buffer.source().capability(), 4, 10, 50).unwrap();
            let mut scratch = [0u8; 8];
            {
                let mut adapter = crate::WasapiCaptureAdapter::with_remainder(
                    config, FailingClock { panic }, &mut scratch, 8).unwrap();
                adapter.source = Some(buffer.source());
                adapter.active = Some(buffer.session());
                adapter.remainder.stage(&[7, 8], crate::remainder::PacketMetadata {
                    device_position: Some(0), qpc_position: None, silent: false,
                    discontinuity: false, timestamp_error: false,
                }).unwrap();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| adapter.capture(buffer)));
                if panic { assert!(result.is_err()); } else { assert!(result.unwrap().is_err()); }
                assert!(adapter.faulted);
                assert!(adapter.remainder.is_empty());
            }
            assert_eq!(scratch, [0; 8]);
            assert_eq!(bytes, [0, 0, 0, 0, 0x55, 0x55, 0x55, 0x55]);
        }
    }
}
