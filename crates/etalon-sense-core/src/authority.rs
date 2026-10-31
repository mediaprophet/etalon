//! Pluggable host-policy issuance seam for built-in and extended capture.
//!
//! The host decides whether a request is permitted. This module supplies no
//! default policy, identity model, guardian semantics, signatures, or proof of
//! authenticity. A nonzero context label may denote a public, environmental,
//! scientific, or any other host-defined context. Issuance only converts an
//! explicit host permit decision into the existing adapter request types.
//! A host requiring revocation or context revalidation must compose its policy
//! check with the actual capture operation: the adapter request retains source,
//! purpose and expiry but does not retain or recheck the complete policy context.

use crate::{
    adapter::{
        AdapterError, AuthorizedCapture, AuthorizedExtendedCapture, CapturePermit,
        DiscoveredSource, ExtendedCapturePermit, ExtendedDiscoveredSource,
    },
    time::ClockStamp,
};

/// Source declaration as seen by the host policy evaluator. Capability values
/// are preserved exactly, including extended registry IDs and unknown units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CaptureSource {
    BuiltIn(DiscoveredSource),
    Extended(ExtendedDiscoveredSource),
}

impl CaptureSource {
    pub const fn source_id(self) -> u64 {
        match self {
            Self::BuiltIn(source) => source.source_id(),
            Self::Extended(source) => source.source_id(),
        }
    }
}

/// Exact request passed to a host policy. Subject and actor may be absent for
/// public data, natural environments, non-human species, or unidentified cases.
/// IDs are opaque labels; this crate does not interpret or authenticate them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptureAuthorizationRequest {
    source: CaptureSource,
    subject_id: Option<u64>,
    actor_id: Option<u64>,
    context_id: u64,
    purpose_id: u64,
}

impl CaptureAuthorizationRequest {
    pub const fn new(
        source: CaptureSource,
        subject_id: Option<u64>,
        actor_id: Option<u64>,
        context_id: u64,
        purpose_id: u64,
    ) -> Result<Self, AuthorityError> {
        if context_id == 0 || purpose_id == 0
            || matches!(subject_id, Some(0))
            || matches!(actor_id, Some(0))
        {
            return Err(AuthorityError::InvalidRequest);
        }
        Ok(Self { source, subject_id, actor_id, context_id, purpose_id })
    }

    pub const fn source(self) -> CaptureSource { self.source }
    pub const fn subject_id(self) -> Option<u64> { self.subject_id }
    pub const fn actor_id(self) -> Option<u64> { self.actor_id }
    pub const fn context_id(self) -> u64 { self.context_id }
    pub const fn purpose_id(self) -> u64 { self.purpose_id }
}

/// Result from a host-provided policy evaluator. Permit carries only an
/// expiry; the host makes the authorization decision and retains its own
/// policy semantics. Unknown is distinct from denial and both fail closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturePolicyDecision {
    Permit { expires_at: ClockStamp },
    Deny,
    Unknown,
}

/// Host-owned policy authority. Implementations should bind decisions to the
/// complete request, including exact source capability, purpose, and context.
/// No implementation is provided by this crate.
pub trait HostCapturePolicy {
    fn decide(
        &self,
        request: &CaptureAuthorizationRequest,
        now: ClockStamp,
    ) -> CapturePolicyDecision;
}

/// Host-supplied current time source. Its trustworthiness is external to this
/// crate; its reading is still checked for domain and uncertainty at issuance.
pub trait TrustedCaptureClock {
    fn now(&mut self) -> Result<ClockStamp, AdapterError>;
}

/// Issuance failures. No failure is converted to an implicit permission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityError {
    InvalidRequest,
    ClockUnavailable,
    ClockDomainMismatch,
    ExpiredOrUncertain,
    Denied,
    Unknown,
    Adapter(AdapterError),
}

impl From<AdapterError> for AuthorityError {
    fn from(error: AdapterError) -> Self { Self::Adapter(error) }
}

/// Existing adapter request produced from an explicit host decision.
#[derive(Debug, PartialEq)]
pub enum AuthorizedCaptureRequest {
    BuiltIn(AuthorizedCapture),
    Extended(AuthorizedExtendedCapture),
}

/// Non-copyable issuance record retaining the context supplied to policy.
/// Converting it into an adapter request discards only this local wrapper; the
/// adapter still enforces its exact source/capability, purpose, and expiry.
#[derive(Debug, PartialEq)]
pub struct IssuedCaptureAuthorization {
    request: CaptureAuthorizationRequest,
    authorized_at: ClockStamp,
    expires_at: ClockStamp,
    capture: AuthorizedCaptureRequest,
}

impl IssuedCaptureAuthorization {
    pub const fn request(&self) -> &CaptureAuthorizationRequest { &self.request }
    pub const fn authorized_at(&self) -> ClockStamp { self.authorized_at }
    pub const fn expires_at(&self) -> ClockStamp { self.expires_at }
    pub fn into_capture_request(self) -> AuthorizedCaptureRequest { self.capture }
}

/// Stateless issuance entry point. It calls the supplied clock and policy on
/// every request; there is no cached/default grant and no way to bypass the
/// host decision through this API.
pub struct CaptureAuthority;

impl CaptureAuthority {
    pub fn issue<P, C>(
        request: CaptureAuthorizationRequest,
        policy: &P,
        clock: &mut C,
    ) -> Result<IssuedCaptureAuthorization, AuthorityError>
    where
        P: HostCapturePolicy,
        C: TrustedCaptureClock,
    {
        let now = clock.now().map_err(|_| AuthorityError::ClockUnavailable)?;
        let expires_at = match policy.decide(&request, now) {
            CapturePolicyDecision::Permit { expires_at } => expires_at,
            CapturePolicyDecision::Deny => return Err(AuthorityError::Denied),
            CapturePolicyDecision::Unknown => return Err(AuthorityError::Unknown),
        };
        validate_window(now, expires_at)?;

        let capture = match request.source {
            CaptureSource::BuiltIn(source) => {
                let permit = CapturePermit::after_authorization(
                    source.source_id(),
                    request.purpose_id,
                    source.capability(),
                    expires_at,
                )?;
                AuthorizedCaptureRequest::BuiltIn(AuthorizedCapture::new(source, permit, now)?)
            }
            CaptureSource::Extended(source) => {
                let permit = ExtendedCapturePermit::after_authorization(
                    source.source_id(),
                    request.purpose_id,
                    source.capability(),
                    expires_at,
                )?;
                AuthorizedCaptureRequest::Extended(AuthorizedExtendedCapture::new(source, permit, now)?)
            }
        };

        Ok(IssuedCaptureAuthorization { request, authorized_at: now, expires_at, capture })
    }
}

fn validate_window(now: ClockStamp, expires_at: ClockStamp) -> Result<(), AuthorityError> {
    if now.clock_id() != expires_at.clock_id()
        || now.restart_epoch() != expires_at.restart_epoch()
    {
        return Err(AuthorityError::ClockDomainMismatch);
    }
    // Require every plausible current time to precede every plausible expiry.
    // Overlap and equality fail closed.
    if now.interval().upper_ns() >= expires_at.interval().lower_ns() {
        return Err(AuthorityError::ExpiredOrUncertain);
    }
    Ok(())
}

#[cfg(test)]
#[path = "authority_tests.rs"]
mod tests;
