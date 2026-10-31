//! Bounded capture-session lifecycle and explicit authorization gate.
//!
//! A session validates caller-provided declarations and buffer views. It does
//! not open hardware, authenticate authorization, or verify a sensor backend.

use crate::buffer::{SampleLayout, SampleRegion};
use crate::capability::{Modality, SensorCapability};
use crate::quality::{SourceDescriptor, TrustState};
use crate::time::ClockStamp;
use crate::ContractError;

/// Lifecycle state of one capture session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionPhase {
    Ready,
    Running,
    Stopped,
    Revoked,
}

/// Result of a session operation that may be refused by its contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionError {
    Contract(ContractError),
    InvalidState,
    GrantScopeMismatch,
    GrantNotYetValid,
    GrantExpired,
    TimeUncertain,
    Revoked,
    CaptureNotActive,
    FrameBudgetExhausted,
}

impl From<ContractError> for SessionError {
    fn from(value: ContractError) -> Self {
        Self::Contract(value)
    }
}

/// Caller-supplied authorization assertion from an external policy authority.
///
/// Its fields are structurally checked, but are not signed, authenticated, or
/// otherwise verified here. A trusted external policy gate must authorize the
/// operation before the caller supplies this assertion. The session never
/// treats it as hardware or source trust.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalAuthorization {
    grant_id: u64,
    authority_id: u64,
    subject_id: u64,
    environment_id: u64,
    purpose_id: u64,
    modality: Modality,
    not_before: ClockStamp,
    expires_at: ClockStamp,
}

impl ExternalAuthorization {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        grant_id: u64,
        authority_id: u64,
        subject_id: u64,
        environment_id: u64,
        purpose_id: u64,
        modality: Modality,
        not_before: ClockStamp,
        expires_at: ClockStamp,
    ) -> Result<Self, SessionError> {
        if grant_id == 0
            || authority_id == 0
            || subject_id == 0
            || environment_id == 0
            || purpose_id == 0
        {
            return Err(SessionError::Contract(ContractError::InvalidIdentifier));
        }
        if not_before.clock_id() != expires_at.clock_id()
            || not_before.restart_epoch() != expires_at.restart_epoch()
        {
            return Err(SessionError::Contract(ContractError::ClockDomainMismatch));
        }
        if not_before.interval().upper_ns() >= expires_at.interval().lower_ns() {
            return Err(SessionError::Contract(ContractError::InvalidTime));
        }
        Ok(Self {
            grant_id,
            authority_id,
            subject_id,
            environment_id,
            purpose_id,
            modality,
            not_before,
            expires_at,
        })
    }

    pub const fn grant_id(self) -> u64 {
        self.grant_id
    }

    pub const fn authority_id(self) -> u64 {
        self.authority_id
    }

    pub const fn not_before(self) -> ClockStamp {
        self.not_before
    }

    pub const fn expires_at(self) -> ClockStamp {
        self.expires_at
    }
}

/// A bounded session for admitting samples from one declared source.
#[derive(Debug, PartialEq)]
pub struct CaptureSession {
    session_id: u64,
    source: SourceDescriptor,
    capability: SensorCapability,
    subject_id: u64,
    environment_id: u64,
    purpose_id: u64,
    frame_budget: u64,
    frames_admitted: u64,
    phase: SessionPhase,
    active_grant_id: Option<u64>,
    grant_not_before: Option<ClockStamp>,
    grant_expires_at: Option<ClockStamp>,
    last_check: Option<ClockStamp>,
}

impl CaptureSession {
    /// Create a ready session. `frame_budget` bounds the number of accepted
    /// frames in each start/stop interval and must be nonzero.
    pub const fn new(
        session_id: u64,
        source: SourceDescriptor,
        capability: SensorCapability,
        subject_id: u64,
        environment_id: u64,
        purpose_id: u64,
        frame_budget: u64,
    ) -> Result<Self, SessionError> {
        if session_id == 0 || subject_id == 0 || environment_id == 0 || purpose_id == 0 {
            return Err(SessionError::Contract(ContractError::InvalidIdentifier));
        }
        if frame_budget == 0 {
            return Err(SessionError::Contract(ContractError::InvalidExtent));
        }
        Ok(Self {
            session_id,
            source,
            capability,
            subject_id,
            environment_id,
            purpose_id,
            frame_budget,
            frames_admitted: 0,
            phase: SessionPhase::Ready,
            active_grant_id: None,
            grant_not_before: None,
            grant_expires_at: None,
            last_check: None,
        })
    }

    /// Start only with an unexpired external authorization assertion scoped to
    /// this session's subject, environment, purpose, and modality. This local
    /// check does not replace a trusted external policy gate.
    pub fn start(
        &mut self,
        now: ClockStamp,
        grant: ExternalAuthorization,
    ) -> Result<(), SessionError> {
        if matches!(self.phase, SessionPhase::Revoked) {
            return Err(SessionError::Revoked);
        }
        if !matches!(self.phase, SessionPhase::Ready | SessionPhase::Stopped) {
            return Err(SessionError::InvalidState);
        }
        if grant.subject_id != self.subject_id
            || grant.environment_id != self.environment_id
            || grant.purpose_id != self.purpose_id
            || grant.modality != self.capability.modality()
        {
            return Err(SessionError::GrantScopeMismatch);
        }
        check_grant_time(now, grant.not_before, grant.expires_at)?;

        self.phase = SessionPhase::Running;
        self.active_grant_id = Some(grant.grant_id);
        self.grant_not_before = Some(grant.not_before);
        self.grant_expires_at = Some(grant.expires_at);
        self.frames_admitted = 0;
        self.last_check = Some(now);
        Ok(())
    }

    /// Admit a bounded sample region while authorization is active. A frame is
    /// counted only after time, capability, layout, and region checks succeed.
    pub fn admit_frame(
        &mut self,
        now: ClockStamp,
        region: &SampleRegion,
        layout: &SampleLayout,
    ) -> Result<(), SessionError> {
        if !self.capture_permitted(now)? {
            return Err(SessionError::CaptureNotActive);
        }
        if self.frames_admitted >= self.frame_budget {
            self.stop();
            return Err(SessionError::FrameBudgetExhausted);
        }
        if layout.encoding() != &self.capability.encoding() {
            return Err(SessionError::Contract(ContractError::IncompatibleEncoding));
        }
        if layout.rank() != self.capability.max_rank() {
            return Err(SessionError::Contract(ContractError::InvalidExtent));
        }
        for axis in 0..usize::from(layout.rank()) {
            if layout.extents()[axis] > self.capability.max_extents()[axis] {
                return Err(SessionError::Contract(ContractError::OutOfBounds));
            }
        }
        region.validate_layout(layout)?;

        self.frames_admitted = self
            .frames_admitted
            .checked_add(1)
            .ok_or(SessionError::FrameBudgetExhausted)?;
        Ok(())
    }

    /// Check this session's local grant assertion and monotonic time before
    /// capture. The entire timestamp uncertainty interval must fit inside the
    /// grant window. This remains a local assertion check, not a trusted
    /// authorization decision; the caller must enforce policy externally.
    pub fn capture_permitted(&mut self, now: ClockStamp) -> Result<bool, SessionError> {
        if !matches!(self.phase, SessionPhase::Running) {
            return Ok(false);
        }
        if let Some(last_check) = self.last_check {
            if now.clock_id() != last_check.clock_id()
                || now.restart_epoch() != last_check.restart_epoch()
            {
                self.stop();
                return Err(SessionError::Contract(ContractError::ClockDomainMismatch));
            }
            if now.ticks_ns() < last_check.ticks_ns() {
                self.stop();
                return Err(SessionError::Contract(ContractError::InvalidTime));
            }
        }
        let (not_before, expires_at) = match (self.grant_not_before, self.grant_expires_at) {
            (Some(not_before), Some(expires_at)) => (not_before, expires_at),
            _ => {
                self.stop();
                return Err(SessionError::InvalidState);
            }
        };
        if let Err(error) = check_grant_time(now, not_before, expires_at) {
            self.stop();
            return match error {
                SessionError::GrantExpired => Ok(false),
                other => Err(other),
            };
        }
        self.last_check = Some(now);
        Ok(true)
    }

    /// Stop admission and discard the active grant reference. A stopped
    /// session may be restarted with a new authorization assertion.
    pub fn stop(&mut self) {
        if !matches!(self.phase, SessionPhase::Revoked) {
            self.phase = SessionPhase::Stopped;
            self.active_grant_id = None;
            self.grant_not_before = None;
            self.grant_expires_at = None;
            self.last_check = None;
        }
    }

    /// Revoke this session immediately. Revocation is terminal and idempotent.
    pub fn revoke(&mut self) {
        self.phase = SessionPhase::Revoked;
        self.active_grant_id = None;
        self.grant_not_before = None;
        self.grant_expires_at = None;
        self.last_check = None;
    }

    pub const fn phase(&self) -> SessionPhase {
        self.phase
    }

    pub const fn session_id(&self) -> u64 {
        self.session_id
    }

    pub const fn source(&self) -> SourceDescriptor {
        self.source
    }

    pub const fn source_trust_state(&self) -> TrustState {
        self.source.trust_state()
    }

    pub const fn capability(&self) -> SensorCapability {
        self.capability
    }

    pub const fn active_grant_id(&self) -> Option<u64> {
        self.active_grant_id
    }

    pub const fn frame_budget(&self) -> u64 {
        self.frame_budget
    }

    pub const fn frames_admitted(&self) -> u64 {
        self.frames_admitted
    }
}

/// Require all plausible `now` times to be inside an inclusive-start,
/// exclusive-expiry window. Ambiguous boundaries fail closed.
fn check_grant_time(
    now: ClockStamp,
    not_before: ClockStamp,
    expires_at: ClockStamp,
) -> Result<(), SessionError> {
    if now.clock_id() != not_before.clock_id()
        || now.restart_epoch() != not_before.restart_epoch()
        || now.clock_id() != expires_at.clock_id()
        || now.restart_epoch() != expires_at.restart_epoch()
    {
        return Err(SessionError::Contract(ContractError::ClockDomainMismatch));
    }

    let now_interval = now.interval();
    let start_interval = not_before.interval();
    let expiry_interval = expires_at.interval();
    if now_interval.upper_ns() < start_interval.lower_ns() {
        return Err(SessionError::GrantNotYetValid);
    }
    if now_interval.lower_ns() < start_interval.upper_ns() {
        return Err(SessionError::TimeUncertain);
    }
    if now_interval.lower_ns() >= expiry_interval.upper_ns() {
        return Err(SessionError::GrantExpired);
    }
    if now_interval.upper_ns() >= expiry_interval.lower_ns() {
        return Err(SessionError::TimeUncertain);
    }
    Ok(())
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
