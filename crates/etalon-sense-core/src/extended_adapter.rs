//! Parallel adapter contract for host-defined modality/quantity identifiers.
//! Registry membership and unit semantics remain external; this module only
//! binds exact declarations to authorization and caller-owned buffers.

use super::{AdapterError, AdapterId};
use crate::{
    buffer::SampleLayout, extended_capability::ExtendedCapability, time::ClockStamp, ContractError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtendedAdapterDescriptor {
    id: AdapterId,
    modality_id: u64,
}

impl ExtendedAdapterDescriptor {
    pub const fn new(id: AdapterId, modality_id: u64) -> Result<Self, AdapterError> {
        if modality_id == 0 {
            return Err(AdapterError::InvalidIdentifier);
        }
        Ok(Self { id, modality_id })
    }

    pub const fn id(self) -> AdapterId {
        self.id
    }
    pub const fn modality_id(self) -> u64 {
        self.modality_id
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtendedDiscoveredSource {
    source_id: u64,
    capability: ExtendedCapability,
}

impl ExtendedDiscoveredSource {
    pub const fn new(source_id: u64, capability: ExtendedCapability) -> Result<Self, AdapterError> {
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
    pub const fn capability(self) -> ExtendedCapability {
        self.capability
    }
}

/// Crate-issued authorization assertion bound to an exact extended capability.
/// The constructor is crate-private so only a trusted host policy integration
/// can issue it. It does not itself prove policy evaluation or hardware trust.
#[derive(Debug, PartialEq)]
pub struct ExtendedCapturePermit {
    source_id: u64,
    purpose_id: u64,
    capability: ExtendedCapability,
    expires_at: ClockStamp,
}

impl ExtendedCapturePermit {
    pub(crate) const fn after_authorization(
        source_id: u64,
        purpose_id: u64,
        capability: ExtendedCapability,
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
        source: ExtendedDiscoveredSource,
        now: ClockStamp,
    ) -> Result<(), AdapterError> {
        if self.source_id != source.source_id || self.capability != source.capability {
            return Err(AdapterError::AuthorizationDenied);
        }
        check_expiry(self.expires_at, now)
    }
}

/// Validated per-operation request; callers cannot create it without an
/// in-crate authorization integration invoking the permit constructor.
#[derive(Debug, PartialEq)]
pub struct AuthorizedExtendedCapture {
    source: ExtendedDiscoveredSource,
    purpose_id: u64,
    authorized_at: ClockStamp,
    expires_at: ClockStamp,
}

impl AuthorizedExtendedCapture {
    pub fn new(
        source: ExtendedDiscoveredSource,
        permit: ExtendedCapturePermit,
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

    pub const fn source(&self) -> ExtendedDiscoveredSource {
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

    /// Validate the one-shot expiry interval against a trusted current time.
    /// This does not re-evaluate the host policy or external revocation state.
    pub fn validate_at(&self, now: ClockStamp) -> Result<(), AdapterError> {
        if now.clock_id() != self.expires_at.clock_id()
            || now.restart_epoch() != self.expires_at.restart_epoch()
        {
            return Err(AdapterError::ClockDomainMismatch);
        }
        if now.interval().lower_ns() < self.authorized_at.interval().upper_ns() {
            return Err(AdapterError::AuthorizationDenied);
        }
        check_expiry(self.expires_at, now)
    }
}

fn check_expiry(expires_at: ClockStamp, now: ClockStamp) -> Result<(), AdapterError> {
    if now.clock_id() != expires_at.clock_id() || now.restart_epoch() != expires_at.restart_epoch()
    {
        return Err(AdapterError::ClockDomainMismatch);
    }
    if now.interval().upper_ns() < expires_at.interval().lower_ns() {
        Ok(())
    } else {
        Err(AdapterError::AuthorizationExpired)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtendedCaptureSession {
    adapter: AdapterId,
    source: ExtendedDiscoveredSource,
    session_id: u64,
    generation: u32,
}

impl ExtendedCaptureSession {
    pub const fn new(
        adapter: AdapterId,
        source: ExtendedDiscoveredSource,
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
    pub const fn source(self) -> ExtendedDiscoveredSource {
        self.source
    }
    pub const fn session_id(self) -> u64 {
        self.session_id
    }
    pub const fn generation(self) -> u32 {
        self.generation
    }

    pub fn matches(
        &self,
        adapter: ExtendedAdapterDescriptor,
        source: ExtendedDiscoveredSource,
    ) -> bool {
        self.adapter.value() == adapter.id().value()
            && self.source.source_id == source.source_id
            && self.source.capability == source.capability
            && self.source.capability.modality_id() == adapter.modality_id()
    }
}

/// Caller-owned output buffer validated for one extended stream declaration.
pub struct ExtendedCaptureBuffer<'a> {
    bytes: &'a mut [u8],
    span_bytes: usize,
    layout: SampleLayout,
    capability: ExtendedCapability,
}

impl<'a> ExtendedCaptureBuffer<'a> {
    pub fn new(
        bytes: &'a mut [u8],
        layout: SampleLayout,
        capability: ExtendedCapability,
    ) -> Result<Self, AdapterError> {
        capability.validate_layout(&layout, None)?;
        let span = layout.required_byte_span()?;
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
    pub const fn capability(&self) -> ExtendedCapability {
        self.capability
    }
    pub fn writable_bytes(&mut self) -> &mut [u8] {
        &mut self.bytes[..self.span_bytes]
    }

    fn complete_validated(self) -> ExtendedCapturedSamples<'a> {
        ExtendedCapturedSamples {
            bytes: &self.bytes[..self.span_bytes],
            layout: self.layout,
        }
    }
}

pub struct AuthorizedExtendedCaptureBuffer<'a> {
    session: ExtendedCaptureSession,
    request: AuthorizedExtendedCapture,
    buffer: ExtendedCaptureBuffer<'a>,
}

impl<'a> AuthorizedExtendedCaptureBuffer<'a> {
    pub fn new(
        adapter: ExtendedAdapterDescriptor,
        session: ExtendedCaptureSession,
        request: AuthorizedExtendedCapture,
        buffer: ExtendedCaptureBuffer<'a>,
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

    pub const fn session(&self) -> ExtendedCaptureSession {
        self.session
    }
    pub const fn source(&self) -> ExtendedDiscoveredSource {
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
    /// Recheck this one-shot request against a trusted current clock reading.
    /// This checks the captured authorization time window only; it does not
    /// re-evaluate host policy, revocation, context, or authorization evidence.
    pub fn validate_at(&self, now: ClockStamp) -> Result<(), AdapterError> {
        self.request.validate_at(now)
    }
    pub fn writable_bytes(&mut self) -> &mut [u8] {
        self.buffer.writable_bytes()
    }

    /// Exposes only the validated sample span. A backend must check its trusted
    /// authorization clock immediately before actual device I/O.
    pub fn complete(self) -> ExtendedCapturedSamples<'a> {
        self.buffer.complete_validated()
    }
}

/// Validated read-only bytes; not proof of authorization, device access, or I/O.
#[derive(Debug, PartialEq)]
pub struct ExtendedCapturedSamples<'a> {
    bytes: &'a [u8],
    layout: SampleLayout,
}

impl<'a> ExtendedCapturedSamples<'a> {
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }
}

/// Parallel lifecycle interface for host-defined modality IDs. Authorization
/// comes from `AuthorizedExtendedCapture`; the backend must recheck trusted time
/// and its configured host policy immediately before device I/O.
pub trait ExtendedSensorAdapter {
    fn descriptor(&self) -> ExtendedAdapterDescriptor;
    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError>;
    fn discover_backend(
        &mut self,
        out: &mut [ExtendedDiscoveredSource],
    ) -> Result<usize, AdapterError>;
    fn start_backend(
        &mut self,
        request: AuthorizedExtendedCapture,
    ) -> Result<ExtendedCaptureSession, AdapterError>;
    fn capture_backend<'a>(
        &mut self,
        output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<ExtendedCapturedSamples<'a>, AdapterError>;
    fn stop_backend(&mut self, session: ExtendedCaptureSession) -> Result<(), AdapterError>;

    fn discover(&mut self, out: &mut [ExtendedDiscoveredSource]) -> Result<usize, AdapterError> {
        let count = self.discover_backend(out)?;
        if count > out.len() {
            return Err(AdapterError::DiscoveryBufferTooSmall);
        }
        for (index, source) in out[..count].iter().enumerate() {
            if source.capability.modality_id() != self.descriptor().modality_id() {
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

    fn start(
        &mut self,
        request: AuthorizedExtendedCapture,
    ) -> Result<ExtendedCaptureSession, AdapterError> {
        let descriptor = self.descriptor();
        if request.source.capability.modality_id() != descriptor.modality_id() {
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
        output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<ExtendedCapturedSamples<'a>, AdapterError> {
        if !output
            .session
            .matches(self.descriptor(), output.request.source)
        {
            return Err(AdapterError::InvalidSession);
        }
        output.request.validate_at(self.trusted_now()?)?;
        self.capture_backend(output)
    }

    fn stop(&mut self, session: ExtendedCaptureSession) -> Result<(), AdapterError> {
        if session.adapter.value() != self.descriptor().id().value()
            || session.source.capability.modality_id() != self.descriptor().modality_id()
        {
            return Err(AdapterError::InvalidSession);
        }
        self.stop_backend(session)
    }
}
