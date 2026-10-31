//! Bounded adapter for host-supplied byte streams.
//!
//! A `Read` value may be a file or an OS-provided device stream, but this
//! adapter does not authenticate or classify it as hardware. It consumes one
//! exact frame per capture into the caller's buffer. Authorization expiry is
//! checked before every read and after a read returns. A blocking `Read::read`
//! cannot be interrupted here. If authorization expires while it is blocked,
//! the adapter clears the caller buffer and rejects the frame, but cannot undo
//! an external device read that already occurred. The host must arrange
//! policy/revocation checks and cancellation around queued or delayed work.

use std::io::{self, Read};

use etalon_sense_core::{
    adapter::{
        AdapterError, AdapterId, ExtendedAdapterDescriptor, ExtendedCaptureSession,
        ExtendedCapturedSamples, ExtendedDiscoveredSource, ExtendedSensorAdapter,
    },
    authority::TrustedCaptureClock,
    buffer::SampleLayout,
    time::ClockStamp,
};

fn zero_capture_buffer(
    output: &mut etalon_sense_core::adapter::AuthorizedExtendedCaptureBuffer<'_>,
) {
    output.writable_bytes().fill(0);
}

/// Terminal/readiness state for the underlying sequential source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamState {
    Ready,
    Eof,
    Faulted,
}

/// Host-fed, exact-frame adapter. The source declaration and layout are
/// supplied by the host and are checked against every authorized request.
/// This object owns neither the input nor capture memory for callers.
pub struct StreamAdapter<R, C> {
    descriptor: ExtendedAdapterDescriptor,
    source: ExtendedDiscoveredSource,
    layout: SampleLayout,
    max_frame_bytes: usize,
    reader: R,
    clock: C,
    active: Option<ExtendedCaptureSession>,
    next_session_id: u64,
    generation: u32,
    completed_frames: u64,
    state: StreamState,
}

impl<R, C> StreamAdapter<R, C>
where
    R: Read,
    C: TrustedCaptureClock,
{
    /// Bind a reader to one declared extended source and one exact layout.
    /// This checks structural compatibility only, not registry meaning or
    /// whether `reader` corresponds to a physical sensor.
    pub fn new(
        adapter_id: AdapterId,
        source: ExtendedDiscoveredSource,
        layout: SampleLayout,
        max_frame_bytes: usize,
        reader: R,
        clock: C,
    ) -> Result<Self, AdapterError> {
        let descriptor =
            ExtendedAdapterDescriptor::new(adapter_id, source.capability().modality_id())?;
        source.capability().validate_layout(&layout, None)?;
        let frame_bytes = usize::try_from(layout.required_byte_span()?).map_err(|_| {
            AdapterError::Buffer(etalon_sense_core::ContractError::ArithmeticOverflow)
        })?;
        if frame_bytes == 0 || frame_bytes > max_frame_bytes {
            return Err(AdapterError::Buffer(
                etalon_sense_core::ContractError::OutOfBounds,
            ));
        }
        Ok(Self {
            descriptor,
            source,
            layout,
            max_frame_bytes,
            reader,
            clock,
            active: None,
            next_session_id: 1,
            generation: 1,
            completed_frames: 0,
            state: StreamState::Ready,
        })
    }

    pub const fn state(&self) -> StreamState {
        self.state
    }
    pub const fn completed_frames(&self) -> u64 {
        self.completed_frames
    }
    pub const fn max_frame_bytes(&self) -> usize {
        self.max_frame_bytes
    }

    /// Access the wrapped source for host-side control, if its concrete type
    /// exposes appropriate cancellation or device lifecycle operations.
    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    fn validate_fresh_request(
        &mut self,
        output: &etalon_sense_core::adapter::AuthorizedExtendedCaptureBuffer<'_>,
    ) -> Result<(), AdapterError> {
        let now = self.trusted_now()?;
        output.validate_at(now)
    }

    fn fail_stream(&mut self) {
        self.state = StreamState::Faulted;
    }
}

impl<R, C> ExtendedSensorAdapter for StreamAdapter<R, C>
where
    R: Read,
    C: TrustedCaptureClock,
{
    fn descriptor(&self) -> ExtendedAdapterDescriptor {
        self.descriptor
    }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        self.clock.now()
    }

    fn discover_backend(
        &mut self,
        out: &mut [ExtendedDiscoveredSource],
    ) -> Result<usize, AdapterError> {
        if out.is_empty() {
            return Err(AdapterError::DiscoveryBufferTooSmall);
        }
        out[0] = self.source;
        Ok(1)
    }

    fn start_backend(
        &mut self,
        request: etalon_sense_core::adapter::AuthorizedExtendedCapture,
    ) -> Result<ExtendedCaptureSession, AdapterError> {
        if request.source() != self.source {
            return Err(AdapterError::AuthorizationDenied);
        }
        if self.active.is_some() {
            return Err(AdapterError::InvalidSession);
        }
        if self.state != StreamState::Ready {
            return Err(AdapterError::Unavailable);
        }
        request.validate_at(self.trusted_now()?)?;
        let id = self.next_session_id;
        self.next_session_id = id.checked_add(1).ok_or(AdapterError::BackendFailure)?;
        let session =
            ExtendedCaptureSession::new(self.descriptor.id(), self.source, id, self.generation)?;
        self.active = Some(session);
        Ok(session)
    }

    fn capture_backend<'a>(
        &mut self,
        mut output: etalon_sense_core::adapter::AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<ExtendedCapturedSamples<'a>, AdapterError> {
        let session = output.session();
        if self.active != Some(session) || !session.matches(self.descriptor, self.source) {
            return Err(AdapterError::InvalidSession);
        }
        if output.source() != self.source || output.layout() != &self.layout {
            return Err(AdapterError::Unsupported);
        }
        if self.state != StreamState::Ready {
            return Err(AdapterError::Unavailable);
        }
        let frame_bytes = usize::try_from(self.layout.required_byte_span()?)
            .map_err(|_| AdapterError::BackendFailure)?;
        if frame_bytes == 0 || frame_bytes > self.max_frame_bytes {
            return Err(AdapterError::BackendFailure);
        }

        let mut filled = 0usize;
        while filled < frame_bytes {
            if let Err(error) = self.validate_fresh_request(&output) {
                if filled > 0 {
                    zero_capture_buffer(&mut output);
                    self.fail_stream();
                }
                return Err(error);
            }
            let result = {
                let bytes = output.writable_bytes();
                self.reader.read(&mut bytes[filled..frame_bytes])
            };
            match result {
                Ok(0) => {
                    self.state = StreamState::Eof;
                    zero_capture_buffer(&mut output);
                    return Err(if filled == 0 {
                        AdapterError::Unavailable
                    } else {
                        AdapterError::BackendFailure
                    });
                }
                Ok(count) if count <= frame_bytes - filled => {
                    filled += count;
                    // A read may block until after expiry. Do not return its
                    // bytes as a capture if the one-shot request expired while
                    // the host reader was blocked.
                    if let Err(error) = self.validate_fresh_request(&output) {
                        zero_capture_buffer(&mut output);
                        self.fail_stream();
                        return Err(error);
                    }
                }
                Ok(_) => {
                    // A nonconforming reader may have written anywhere in the
                    // supplied frame before returning an impossible count.
                    zero_capture_buffer(&mut output);
                    self.fail_stream();
                    return Err(AdapterError::BackendFailure);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    // `Read` gives no byte count alongside an error. A
                    // caller-provided reader may have modified the offered
                    // tail before failing, so clear the whole frame span.
                    zero_capture_buffer(&mut output);
                    self.fail_stream();
                    return Err(AdapterError::BackendFailure);
                }
            }
        }

        let Some(completed_frames) = self.completed_frames.checked_add(1) else {
            zero_capture_buffer(&mut output);
            self.fail_stream();
            return Err(AdapterError::BackendFailure);
        };
        self.completed_frames = completed_frames;
        Ok(output.complete())
    }

    fn stop_backend(&mut self, session: ExtendedCaptureSession) -> Result<(), AdapterError> {
        if self.active != Some(session) {
            return Err(AdapterError::InvalidSession);
        }
        self.active = None;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(AdapterError::BackendFailure)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
