use etalon_sense_core::time::ClockStamp;

use crate::format::{validate_negotiated_format, CameraModality, FrameFormat, FrameFormatError, FrameDeclaration, RadiometricStatus};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameProviderError {
    Unavailable,
    CancelFailed,
    ReadFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameTimestamp {
    /// Provider supplied no timestamp.
    Unknown,
    /// Raw source ticks. Clock and unit IDs are host declarations, not UTC or
    /// calibrated time claims.
    Raw { ticks: i64, clock_domain_id: u64, unit_id: u64 },
    /// Host/provider mapping into an Etalon clock domain, with its uncertainty.
    Mapped(ClockStamp),
}

impl FrameTimestamp {
    fn validate(self) -> Result<(), FrameError> {
        if let Self::Raw { clock_domain_id, unit_id, .. } = self {
            if clock_domain_id == 0 || unit_id == 0 { return Err(FrameError::InvalidMetadata); }
        }
        Ok(())
    }
}

/// Timestamp/gap metadata supplied for one frame. A missing gap count means
/// continuity is unknown. Sequence values are source-local and reset on start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameMetadata {
    pub format_id: u64,
    pub sequence: u64,
    pub gap_before: Option<u64>,
    pub timestamp: FrameTimestamp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderFrame {
    pub bytes_written: u32,
    pub metadata: FrameMetadata,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegotiatedFrame {
    pub source_id: u64,
    pub modality: CameraModality,
    pub format: FrameFormat,
    pub radiometric: RadiometricStatus,
}

/// Host/device-specific input boundary. Implementations must describe exactly
/// the bytes and layout they write, write at most the supplied slice, return
/// frame metadata for that same image, and make `cancel` stop future device
/// capture before it returns success. These methods create no authenticity or
/// hardware-trust proof.
pub trait FrameProvider {
    fn start(&mut self) -> Result<NegotiatedFrame, FrameProviderError>;
    fn next_frame(&mut self, output: &mut [u8]) -> Result<ProviderFrame, FrameProviderError>;
    fn cancel(&mut self) -> Result<(), FrameProviderError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramePolicyOperation { Start, Capture }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramePolicyScope {
    source_id: u64,
    purpose_id: u64,
    modality: CameraModality,
}

impl FramePolicyScope {
    pub const fn source_id(self) -> u64 { self.source_id }
    pub const fn purpose_id(self) -> u64 { self.purpose_id }
    pub const fn modality(self) -> CameraModality { self.modality }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyDecision { Permit, Deny, Unknown }

/// Optional, pluggable host policy boundary. It must re-evaluate current host
/// policy and trusted time for each operation, fail closed on unknown, and run
/// the operation closure exactly once only within its permitted scope. No
/// universal human/guardian policy is defined here.
pub trait FramePolicyGate: Send + Sync {
    fn with_decision(
        &self,
        scope: FramePolicyScope,
        operation: FramePolicyOperation,
        perform: &mut dyn FnMut(),
    ) -> PolicyDecision;
}

/// Explicit composition choice. `HostManaged` does not claim or mint an
/// authorization; the embedding application owns any policy decision.
#[derive(Clone, Copy)]
pub enum FramePolicyMode<'a> {
    HostManaged,
    Gate(&'a dyn FramePolicyGate),
}

impl core::fmt::Debug for FramePolicyMode<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self { Self::HostManaged => f.write_str("HostManaged"), Self::Gate(_) => f.write_str("Gate(..)") }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    InvalidDeclaration,
    InvalidFormat,
    FormatMismatch,
    BufferTooSmall { required: usize },
    InvalidState,
    InvalidMetadata,
    SequenceRegression,
    GapMismatch,
    PolicyDenied,
    PolicyUnknown,
    Provider(FrameProviderError),
    CancelUnconfirmed,
}

impl From<FrameProviderError> for FrameError {
    fn from(value: FrameProviderError) -> Self { Self::Provider(value) }
}

impl From<FrameFormatError> for FrameError {
    fn from(value: FrameFormatError) -> Self { Self::InvalidFormat }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State { Ready, Active, StopFailed, Cancelled, Faulted }

/// Stateful validation/copy boundary for host-fed camera frames. The provider
/// controls how a camera is opened; this adapter validates negotiated source,
/// capability byte ceiling, format and every returned frame before exposing it.
pub struct FrameIngestAdapter<'g, P: FrameProvider> {
    provider: P,
    declaration: FrameDeclaration,
    policy: FramePolicyMode<'g>,
    state: State,
    active_purpose: u64,
    negotiated: Option<NegotiatedFrame>,
    last_sequence: Option<u64>,
}

impl<P: FrameProvider> FrameIngestAdapter<'static, P> {
    /// Create with a host-managed policy boundary. This does not imply that the
    /// capture is authorized; use `with_gate` to compose a gate where desired.
    pub fn new_host_managed(provider: P, declaration: FrameDeclaration) -> Self {
        Self {
            provider, declaration, policy: FramePolicyMode::HostManaged,
            state: State::Ready, active_purpose: 0, negotiated: None, last_sequence: None,
        }
    }

}

impl<'g, P: FrameProvider> FrameIngestAdapter<'g, P> {
    pub fn with_gate(
        provider: P,
        declaration: FrameDeclaration,
        gate: &'g dyn FramePolicyGate,
    ) -> Self {
        FrameIngestAdapter {
            provider, declaration, policy: FramePolicyMode::Gate(gate),
            state: State::Ready, active_purpose: 0, negotiated: None, last_sequence: None,
        }
    }

    pub const fn declaration(&self) -> FrameDeclaration { self.declaration }
    pub const fn negotiated(&self) -> Option<NegotiatedFrame> { self.negotiated }
    pub const fn is_active(&self) -> bool { matches!(self.state, State::Active) }

    pub fn start(&mut self, purpose_id: u64) -> Result<NegotiatedFrame, FrameError> {
        if self.state != State::Ready || purpose_id == 0 { return Err(FrameError::InvalidState); }
        let scope = self.scope(purpose_id);
        let mut result = None;
        let mut operation_attempted = false;
        let mut perform = || {
            operation_attempted = true;
            result = Some(self.provider.start());
        };
        let decision = run_policy(self.policy, scope, FramePolicyOperation::Start, &mut perform);
        drop(perform);
        if decision != PolicyDecision::Permit || result.is_none() {
            // A gate can panic or return deny after invoking the callback. If
            // start was attempted, conservatively stop a partially opened source
            // even if the provider panicked before returning its result.
            if operation_attempted && cancel_provider(&mut self.provider).is_err() {
                self.state = State::StopFailed;
                self.active_purpose = purpose_id;
                return Err(FrameError::CancelUnconfirmed);
            }
            return Err(policy_error(decision));
        }
        let negotiated = match result.expect("checked Some above") {
            Ok(negotiated) => negotiated,
            Err(error) => {
                if cancel_provider(&mut self.provider).is_err() {
                    self.state = State::StopFailed;
                    self.active_purpose = purpose_id;
                    return Err(FrameError::CancelUnconfirmed);
                }
                self.state = State::Faulted;
                return Err(FrameError::Provider(error));
            }
        };
        let valid = negotiated.source_id == self.declaration.source_id()
            && negotiated.modality == self.declaration.modality()
            && validate_negotiated_format(self.declaration, negotiated.format, negotiated.radiometric).is_ok();
        if !valid {
            if cancel_provider(&mut self.provider).is_err() {
                self.state = State::StopFailed;
                self.active_purpose = purpose_id;
                self.negotiated = Some(negotiated);
                return Err(FrameError::CancelUnconfirmed);
            }
            self.state = State::Faulted;
            return Err(FrameError::FormatMismatch);
        }
        self.active_purpose = purpose_id;
        self.negotiated = Some(negotiated);
        self.last_sequence = None;
        self.state = State::Active;
        Ok(negotiated)
    }

    /// Validate and copy one exact negotiated frame. The caller supplies all
    /// storage; any failure clears the entire supplied slice so partial or stale
    /// image bytes cannot be mistaken for a frame.
    pub fn capture(&mut self, output: &mut [u8]) -> Result<FrameMetadata, FrameError> {
        if self.state != State::Active { return Err(FrameError::InvalidState); }
        let negotiated = self.negotiated.ok_or(FrameError::InvalidState)?;
        let required = usize::try_from(negotiated.format.frame_bytes())
            .map_err(|_| FrameError::InvalidFormat)?;
        if output.len() < required { return Err(FrameError::BufferTooSmall { required }); }
        output.fill(0);
        let scope = self.scope(self.active_purpose);
        let mut result = None;
        let mut perform = || result = Some(self.provider.next_frame(&mut output[..required]));
        let decision = run_policy(self.policy, scope, FramePolicyOperation::Capture, &mut perform);
        drop(perform);
        if decision != PolicyDecision::Permit || result.is_none() {
            output.fill(0);
            let error = policy_error(decision);
            return match self.cancel() { Ok(()) => Err(error), Err(_) => Err(FrameError::CancelUnconfirmed) };
        }
        let packet = match result.expect("checked Some above") {
            Ok(packet) => packet,
            Err(error) => {
                output.fill(0);
                self.state = State::Faulted;
                let _ = cancel_provider(&mut self.provider);
                return Err(FrameError::Provider(error));
            }
        };
        if packet.bytes_written != negotiated.format.frame_bytes()
            || packet.metadata.format_id != negotiated.format.format_id()
        {
            output.fill(0);
            self.state = State::Faulted;
            let _ = cancel_provider(&mut self.provider);
            return Err(FrameError::FormatMismatch);
        }
        if let Err(error) = self.validate_metadata(packet.metadata) {
            output.fill(0);
            self.state = State::Faulted;
            let _ = cancel_provider(&mut self.provider);
            return Err(error);
        }
        self.last_sequence = Some(packet.metadata.sequence);
        Ok(packet.metadata)
    }

    /// Stop/cancel is always allowed so authorization loss cannot block local
    /// cleanup. A failed cancel leaves the same adapter in `StopFailed` for a
    /// retry; it does not claim the provider stopped.
    pub fn cancel(&mut self) -> Result<(), FrameError> {
        match self.state {
            State::Cancelled | State::Ready => { self.state = State::Cancelled; Ok(()) }
            State::Active | State::StopFailed | State::Faulted => {
                match cancel_provider(&mut self.provider) {
                    Ok(()) => {
                        self.state = State::Cancelled;
                        self.negotiated = None;
                        self.last_sequence = None;
                        Ok(())
                    }
                    Err(_) => { self.state = State::StopFailed; Err(FrameError::CancelUnconfirmed) }
                }
            }
        }
    }

    fn scope(&self, purpose_id: u64) -> FramePolicyScope {
        FramePolicyScope { source_id: self.declaration.source_id(), purpose_id, modality: self.declaration.modality() }
    }

    fn validate_metadata(&self, metadata: FrameMetadata) -> Result<(), FrameError> {
        metadata.timestamp.validate()?;
        if let Some(previous) = self.last_sequence {
            if metadata.sequence <= previous { return Err(FrameError::SequenceRegression); }
            let missing = metadata.sequence.checked_sub(previous).and_then(|delta| delta.checked_sub(1))
                .ok_or(FrameError::SequenceRegression)?;
            if matches!(metadata.gap_before, Some(reported) if reported != missing) {
                return Err(FrameError::GapMismatch);
            }
        }
        Ok(())
    }
}

impl<P: FrameProvider> Drop for FrameIngestAdapter<'_, P> {
    fn drop(&mut self) {
        if matches!(self.state, State::Active | State::StopFailed | State::Faulted) {
            let _ = cancel_provider(&mut self.provider);
        }
    }
}

fn run_policy(
    mode: FramePolicyMode<'_>,
    scope: FramePolicyScope,
    operation: FramePolicyOperation,
    perform: &mut dyn FnMut(),
) -> PolicyDecision {
    match mode {
        FramePolicyMode::HostManaged => {
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(perform)).is_ok() {
                PolicyDecision::Permit
            } else {
                PolicyDecision::Unknown
            }
        }
        FramePolicyMode::Gate(gate) => {
            let mut performed = false;
            let mut once = || { if !performed { performed = true; perform(); } };
            let decision = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                gate.with_decision(scope, operation, &mut once)
            })).unwrap_or(PolicyDecision::Unknown);
            drop(once);
            if !performed && decision == PolicyDecision::Permit { PolicyDecision::Unknown } else { decision }
        }
    }
}

/// Contain host-provider unwinding during best-effort cleanup. A panic does not
/// establish that capture stopped; callers retain retryable stop-failed state.
fn cancel_provider(provider: &mut impl FrameProvider) -> Result<(), FrameProviderError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| provider.cancel())) {
        Ok(result) => result,
        Err(_) => Err(FrameProviderError::CancelFailed),
    }
}

fn policy_error(decision: PolicyDecision) -> FrameError {
    match decision {
        PolicyDecision::Deny => FrameError::PolicyDenied,
        PolicyDecision::Unknown | PolicyDecision::Permit => FrameError::PolicyUnknown,
    }
}
