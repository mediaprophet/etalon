use crate::{
    buffer::SampleLayout,
    capability::{Modality, SensorCapability},
    time::ClockStamp,
    ContractError,
};

#[path = "extended_adapter.rs"]
mod extended;
pub use extended::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterId(u64);

impl AdapterId {
    pub const fn new(value: u64) -> Result<Self, AdapterError> {
        if value == 0 {
            return Err(AdapterError::InvalidIdentifier);
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterDescriptor {
    id: AdapterId,
    modality: Modality,
}

impl AdapterDescriptor {
    pub const fn new(id: AdapterId, modality: Modality) -> Self {
        Self { id, modality }
    }

    pub const fn id(self) -> AdapterId {
        self.id
    }

    pub const fn modality(self) -> Modality {
        self.modality
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscoveredSource {
    source_id: u64,
    capability: SensorCapability,
}

impl DiscoveredSource {
    pub const fn new(
        source_id: u64,
        capability: SensorCapability,
    ) -> Result<Self, AdapterError> {
        if source_id == 0 {
            return Err(AdapterError::InvalidIdentifier);
        }
        Ok(Self {
            source_id,
            capability,
        })
    }

    pub const fn source_id(self) -> u64 {
        self.source_id
    }

    pub const fn capability(self) -> SensorCapability {
        self.capability
    }
}

/// Capture authorization; only trusted crate code can issue a permit.
#[derive(Debug, PartialEq)]
pub struct CapturePermit {
    source_id: u64,
    purpose_id: u64,
    capability: SensorCapability,
    expires_at: ClockStamp,
}

impl CapturePermit {
    pub(crate) const fn after_authorization(
        source_id: u64,
        purpose_id: u64,
        capability: SensorCapability,
        expires_at: ClockStamp,
    ) -> Result<Self, AdapterError> {
        if source_id == 0 || purpose_id == 0 {
            return Err(AdapterError::InvalidIdentifier);
        }
        Ok(Self {
            source_id,
            purpose_id,
            capability,
            expires_at,
        })
    }

    pub const fn source_id(&self) -> u64 {
        self.source_id
    }

    pub const fn purpose_id(&self) -> u64 {
        self.purpose_id
    }

    pub const fn expires_at(&self) -> ClockStamp {
        self.expires_at
    }

    fn validate_for(
        &self,
        source: DiscoveredSource,
        now: ClockStamp,
    ) -> Result<(), AdapterError> {
        if self.source_id != source.source_id || self.capability != source.capability {
            return Err(AdapterError::AuthorizationDenied);
        }
        if now.clock_id() != self.expires_at.clock_id()
            || now.restart_epoch() != self.expires_at.restart_epoch()
        {
            return Err(AdapterError::ClockDomainMismatch);
        }
        let current = now.interval();
        let expiry = self.expires_at.interval();
        if current.upper_ns() < expiry.lower_ns() {
            Ok(())
        } else {
            Err(AdapterError::AuthorizationExpired)
        }
    }
}

/// Crate-issued capture preconditions. Each operation needs a fresh request;
/// `authorization_clock_reading` must come from the trusted authority.
#[derive(Debug, PartialEq)]
pub struct AuthorizedCapture {
    source: DiscoveredSource,
    purpose_id: u64,
    authorized_at: ClockStamp,
    expires_at: ClockStamp,
}

impl AuthorizedCapture {
    pub fn new(
        source: DiscoveredSource,
        permit: CapturePermit,
        authorization_clock_reading: ClockStamp,
    ) -> Result<Self, AdapterError> {
        permit.validate_for(source, authorization_clock_reading)?;
        Ok(Self {
            source,
            purpose_id: permit.purpose_id,
            authorized_at: authorization_clock_reading,
            expires_at: permit.expires_at,
        })
    }

    pub const fn source(&self) -> DiscoveredSource {
        self.source
    }

    pub const fn purpose_id(&self) -> u64 {
        self.purpose_id
    }

    pub const fn authorized_at(&self) -> ClockStamp {
        self.authorized_at
    }

    pub const fn expires_at(&self) -> ClockStamp {
        self.expires_at
    }

    fn validate_at(&self, trusted_now: ClockStamp) -> Result<(), AdapterError> {
        if trusted_now.clock_id() != self.expires_at.clock_id()
            || trusted_now.restart_epoch() != self.expires_at.restart_epoch()
        {
            return Err(AdapterError::ClockDomainMismatch);
        }
        let current = trusted_now.interval();
        if current.lower_ns() < self.authorized_at.interval().upper_ns() {
            return Err(AdapterError::AuthorizationDenied);
        }
        if current.upper_ns() < self.expires_at.interval().lower_ns() {
            Ok(())
        } else {
            Err(AdapterError::AuthorizationExpired)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptureSession {
    adapter: AdapterId,
    source: DiscoveredSource,
    session_id: u64,
    generation: u32,
}

impl CaptureSession {
    pub const fn new(
        adapter: AdapterId,
        source: DiscoveredSource,
        session_id: u64,
        generation: u32,
    ) -> Result<Self, AdapterError> {
        if session_id == 0 || generation == 0 {
            return Err(AdapterError::InvalidIdentifier);
        }
        Ok(Self {
            adapter,
            source,
            session_id,
            generation,
        })
    }

    pub const fn adapter(self) -> AdapterId {
        self.adapter
    }

    pub const fn source(self) -> DiscoveredSource {
        self.source
    }

    pub const fn session_id(self) -> u64 {
        self.session_id
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }

    pub fn matches(&self, adapter: AdapterDescriptor, source: DiscoveredSource) -> bool {
        self.adapter.value() == adapter.id.value()
            && self.source.source_id == source.source_id
            && self.source.capability == source.capability
            && self.source.capability.modality() == adapter.modality
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterError {
    InvalidIdentifier,
    Unsupported,
    Unavailable,
    DiscoveryBufferTooSmall,
    AuthorizationRequired,
    AuthorizationDenied,
    AuthorizationExpired,
    ClockDomainMismatch,
    InvalidSession,
    Buffer(ContractError),
    BackendFailure,
}

impl From<ContractError> for AdapterError {
    fn from(error: ContractError) -> Self {
        Self::Buffer(error)
    }
}

pub struct CaptureBuffer<'a> {
    bytes: &'a mut [u8],
    span_bytes: usize,
    layout: SampleLayout,
    capability: SensorCapability,
}

impl<'a> CaptureBuffer<'a> {
    pub fn new(
        bytes: &'a mut [u8],
        layout: SampleLayout,
        capability: SensorCapability,
    ) -> Result<Self, AdapterError> {
        if layout.encoding() != &capability.encoding() {
            return Err(AdapterError::Buffer(ContractError::IncompatibleEncoding));
        }
        if layout.rank() != capability.max_rank() {
            return Err(AdapterError::Buffer(ContractError::InvalidExtent));
        }
        let extents = layout.extents();
        let max_extents = capability.max_extents();
        let mut sample_count = 1u64;
        for axis in 0..usize::from(layout.rank()) {
            if extents[axis] > max_extents[axis] {
                return Err(AdapterError::Buffer(ContractError::InvalidExtent));
            }
            sample_count = sample_count
                .checked_mul(u64::from(extents[axis]))
                .ok_or(AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        }
        let span = layout.required_byte_span()?;
        let payload_bytes = sample_count
            .checked_mul(u64::from(capability.encoding().element_bytes()))
            .ok_or(AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        if payload_bytes > capability.max_payload_bytes() {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        let span_bytes = usize::try_from(span)
            .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        if span_bytes > bytes.len() {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        Ok(Self {
            bytes,
            span_bytes,
            layout,
            capability,
        })
    }

    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }

    pub const fn capability(&self) -> SensorCapability {
        self.capability
    }

    pub fn writable_bytes(&mut self) -> &mut [u8] {
        &mut self.bytes[..self.span_bytes]
    }

}

impl<'a> CaptureBuffer<'a> {
    fn complete_validated(self) -> CapturedSamples<'a> {
        CapturedSamples {
            bytes: &self.bytes[..self.span_bytes],
            layout: self.layout,
        }
    }
}

pub struct AuthorizedCaptureBuffer<'a> {
    session: CaptureSession,
    request: AuthorizedCapture,
    buffer: CaptureBuffer<'a>,
}

impl<'a> AuthorizedCaptureBuffer<'a> {
    pub fn new(
        adapter: AdapterDescriptor,
        session: CaptureSession,
        request: AuthorizedCapture,
        buffer: CaptureBuffer<'a>,
    ) -> Result<Self, AdapterError> {
        if !session.matches(adapter, request.source) {
            return Err(AdapterError::InvalidSession);
        }
        if buffer.capability != request.source.capability {
            return Err(AdapterError::Unsupported);
        }
        Ok(Self {
            session,
            request,
            buffer,
        })
    }

    pub const fn session(&self) -> CaptureSession {
        self.session
    }

    pub const fn source(&self) -> DiscoveredSource {
        self.request.source
    }

    pub const fn purpose_id(&self) -> u64 {
        self.request.purpose_id
    }

    pub const fn authorized_at(&self) -> ClockStamp {
        self.request.authorized_at
    }

    pub const fn expires_at(&self) -> ClockStamp {
        self.request.expires_at
    }

    pub const fn layout(&self) -> &SampleLayout {
        &self.buffer.layout
    }

    pub fn writable_bytes(&mut self) -> &mut [u8] {
        self.buffer.writable_bytes()
    }

    /// Complete this one-shot authorized handoff and expose only the validated
    /// sample span. This is not an execution attestation; a backend must check
    /// its trusted authorization clock again at the actual device read/write.
    pub fn complete(self) -> CapturedSamples<'a> {
        self.buffer.complete_validated()
    }
}

/// Validated read-only bytes; not proof of authorization, device access, or I/O.
#[derive(Debug, PartialEq)]
pub struct CapturedSamples<'a> {
    bytes: &'a [u8],
    layout: SampleLayout,
}

impl<'a> CapturedSamples<'a> {
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }
}

/// Lifecycle boundary only; no OS, network, or hardware backend is provided.
pub trait SensorAdapter {
    fn descriptor(&self) -> AdapterDescriptor;

    /// Trusted clock only; backend must recheck authorization immediately
    /// before device I/O.
    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError>;

    fn discover_backend(&mut self, out: &mut [DiscoveredSource]) -> Result<usize, AdapterError>;

    /// Enumerate into caller-owned output; never truncate an incomplete list.
    fn discover(&mut self, out: &mut [DiscoveredSource]) -> Result<usize, AdapterError> {
        let count = self.discover_backend(out)?;
        if count > out.len() {
            return Err(AdapterError::DiscoveryBufferTooSmall);
        }
        for (index, source) in out[..count].iter().enumerate() {
            if source.capability.modality() != self.descriptor().modality {
                return Err(AdapterError::Unsupported);
            }
            if out[..index]
                .iter()
                .any(|previous| previous.source_id == source.source_id)
            {
                return Err(AdapterError::InvalidIdentifier);
            }
        }
        Ok(count)
    }

    fn start_backend(&mut self, request: AuthorizedCapture) -> Result<CaptureSession, AdapterError>;

    fn capture_backend<'a>(
        &mut self,
        output: AuthorizedCaptureBuffer<'a>,
    ) -> Result<CapturedSamples<'a>, AdapterError>;

    fn stop_backend(&mut self, session: CaptureSession) -> Result<(), AdapterError>;

    fn start(&mut self, request: AuthorizedCapture) -> Result<CaptureSession, AdapterError> {
        let descriptor = self.descriptor();
        if request.source.capability.modality() != descriptor.modality {
            return Err(AdapterError::Unsupported);
        }
        request.validate_at(self.trusted_now()?)?;
        let source = request.source;
        let session = self.start_backend(request)?;
        if !session.matches(descriptor, source) {
            let _ = self.stop_backend(session);
            return Err(AdapterError::InvalidSession);
        }
        Ok(session)
    }

    fn capture<'a>(
        &mut self,
        output: AuthorizedCaptureBuffer<'a>,
    ) -> Result<CapturedSamples<'a>, AdapterError> {
        if !output
            .session
            .matches(self.descriptor(), output.request.source)
        {
            return Err(AdapterError::InvalidSession);
        }
        // Consume the request and check its expiry at dispatch time. The
        // backend must repeat its trusted check immediately before acquisition.
        output.request.validate_at(self.trusted_now()?)?;
        self.capture_backend(output)
    }

    fn stop(&mut self, session: CaptureSession) -> Result<(), AdapterError> {
        if session.adapter.value() != self.descriptor().id.value()
            || session.source.capability.modality() != self.descriptor().modality
        {
            return Err(AdapterError::InvalidSession);
        }
        self.stop_backend(session)
    }
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
