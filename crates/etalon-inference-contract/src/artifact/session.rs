use super::*;
use crate::{
    ControllerBinding, Decision, ExecutionMode, MAX_CONTROL_EVIDENCE_BYTES, VerificationError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactPhase {
    Ready,
    AwaitingDecision,
    Approved,
    Stopped,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    InvalidConfig,
    InvalidEvidence,
    InvalidBinding,
    InvalidOutput,
    WrongPhase,
    WrongCandidate,
    ByteBudget,
    EventBudget,
    CounterExhausted,
    Expired,
    ClockRegression,
    OccupiedOutput,
    Denied,
    Steered,
    Verification(VerificationError),
}

/// Non-cloneable record consumed by the host's exact-byte delivery adapter.
/// It is metadata, not proof that delivery happened. Partial external delivery
/// cannot be rolled back or automatically retried by this protocol.
#[derive(Debug)]
pub struct AcceptedArtifact {
    key: ArtifactKey,
}
impl AcceptedArtifact {
    pub fn key(&self) -> &ArtifactKey {
        &self.key
    }
    pub fn into_key(self) -> ArtifactKey {
        self.key
    }
}
#[derive(Debug)]
struct Core {
    config: ArtifactConfig,
    mode: ExecutionMode,
    phase: ArtifactPhase,
    sequence: u64,
    last_tick: u64,
    candidate: Option<ArtifactKey>,
    approved_until: Option<u64>,
}
impl Core {
    fn new(config: ArtifactConfig, mode: ExecutionMode, now: u64) -> Result<Self, ArtifactError> {
        if !config.valid(mode != ExecutionMode::StandaloneDeveloper) {
            return Err(ArtifactError::InvalidConfig);
        }
        Ok(Self {
            config,
            mode,
            phase: ArtifactPhase::Ready,
            sequence: 0,
            last_tick: now,
            candidate: None,
            approved_until: None,
        })
    }
    fn fail<T>(&mut self, error: ArtifactError) -> Result<T, ArtifactError> {
        self.phase = ArtifactPhase::Stopped;
        self.approved_until = None;
        Err(error)
    }
    fn clock(&mut self, now: u64) -> Result<(), ArtifactError> {
        if now < self.last_tick {
            return self.fail(ArtifactError::ClockRegression);
        }
        self.last_tick = now;
        Ok(())
    }
    fn propose(&mut self, output: ArtifactOutput, now: u64) -> Result<ArtifactKey, ArtifactError> {
        self.clock(now)?;
        if self.phase != ArtifactPhase::Ready {
            return self.fail(ArtifactError::WrongPhase);
        }
        if output.byte_len > self.config.max_artifact_bytes {
            return self.fail(ArtifactError::ByteBudget);
        }
        if (output.byte_len == 0 && !self.config.allow_empty)
            || [
                output.content_digest,
                output.provenance_digest,
                output.observation_digest,
            ]
            .contains(&[0; 32])
            || output
                .parent
                .is_some_and(|p| p.sequence == 0 || p.record_digest == [0; 32])
        {
            return self.fail(ArtifactError::InvalidOutput);
        }
        let Some(sequence) = self.sequence.checked_add(1) else {
            return self.fail(ArtifactError::CounterExhausted);
        };
        if sequence > self.config.max_events {
            return self.fail(ArtifactError::EventBudget);
        }
        let key = ArtifactKey {
            identity: self.config.identity,
            mode: self.mode,
            event_sequence: sequence,
            output,
        };
        self.sequence = sequence;
        self.candidate = Some(key);
        self.approved_until = None;
        self.phase = if self.mode == ExecutionMode::StandaloneDeveloper {
            ArtifactPhase::Approved
        } else {
            ArtifactPhase::AwaitingDecision
        };
        Ok(key)
    }
    fn pending(&mut self, key: ArtifactKey, phase: ArtifactPhase) -> Result<(), ArtifactError> {
        if self.phase != phase {
            return self.fail(ArtifactError::WrongPhase);
        }
        if self.candidate != Some(key) {
            return self.fail(ArtifactError::WrongCandidate);
        }
        Ok(())
    }
    fn accept(
        &mut self,
        key: ArtifactKey,
        now: u64,
        out: &mut Option<AcceptedArtifact>,
    ) -> Result<(), ArtifactError> {
        if out.is_some() {
            return self.fail(ArtifactError::OccupiedOutput);
        }
        self.clock(now)?;
        self.pending(key, ArtifactPhase::Approved)?;
        if self.approved_until.is_some_and(|until| now >= until) {
            return self.fail(ArtifactError::Expired);
        }
        *out = Some(AcceptedArtifact { key });
        self.candidate = None;
        self.approved_until = None;
        self.phase = ArtifactPhase::Ready;
        Ok(())
    }
}
fn evidence_valid(evidence: &[u8]) -> bool {
    !evidence.is_empty() && evidence.len() <= MAX_CONTROL_EVIDENCE_BYTES
}

#[derive(Debug)]
pub struct DeveloperArtifactSession {
    core: Core,
}
impl DeveloperArtifactSession {
    pub fn new(config: ArtifactConfig, now: u64) -> Result<Self, ArtifactError> {
        Ok(Self {
            core: Core::new(config, ExecutionMode::StandaloneDeveloper, now)?,
        })
    }
    pub fn propose(
        &mut self,
        output: ArtifactOutput,
        now: u64,
    ) -> Result<ArtifactKey, ArtifactError> {
        self.core.propose(output, now)
    }
    pub fn accept_into(
        &mut self,
        key: ArtifactKey,
        now: u64,
        out: &mut Option<AcceptedArtifact>,
    ) -> Result<(), ArtifactError> {
        self.core.accept(key, now, out)
    }
    pub fn cancel(&mut self) {
        self.core.phase = ArtifactPhase::Stopped;
    }
    pub fn phase(&self) -> ArtifactPhase {
        self.core.phase
    }
}

#[derive(Debug)]
pub struct GovernedArtifactSession<V> {
    core: Core,
    verifier: V,
    binding: ControllerBinding,
}
impl<V: ArtifactVerifier> GovernedArtifactSession<V> {
    pub fn admit(
        mut verifier: V,
        request: ArtifactAdmission,
        evidence: &[u8],
    ) -> Result<Self, ArtifactError> {
        let core = Core::new(request.config, request.purpose.mode(), request.now_tick)?;
        if !evidence_valid(evidence) {
            return Err(ArtifactError::InvalidEvidence);
        }
        let binding = verifier
            .admit(&request, evidence)
            .map_err(ArtifactError::Verification)?;
        if binding.identity == [0; 32] || binding.purpose != request.purpose {
            return Err(ArtifactError::InvalidBinding);
        }
        if request.now_tick >= binding.expires_at_tick {
            return Err(ArtifactError::Expired);
        }
        Ok(Self {
            core,
            verifier,
            binding,
        })
    }
    fn live(&mut self, now: u64) -> Result<(), ArtifactError> {
        if self.core.phase == ArtifactPhase::Stopped {
            return Err(ArtifactError::WrongPhase);
        }
        self.core.clock(now)?;
        if now >= self.binding.expires_at_tick {
            return self.core.fail(ArtifactError::Expired);
        }
        let phase = self.core.phase;
        // A host error or unwind must never leave an approved artifact usable.
        self.core.phase = ArtifactPhase::Stopped;
        self.verifier
            .live(&self.binding, now)
            .map_err(ArtifactError::Verification)?;
        self.core.phase = phase;
        Ok(())
    }
    pub fn propose(
        &mut self,
        output: ArtifactOutput,
        now: u64,
    ) -> Result<ArtifactKey, ArtifactError> {
        self.live(now)?;
        self.core.propose(output, now)
    }
    pub fn decide(
        &mut self,
        decision: ArtifactDecision,
        evidence: &[u8],
        now: u64,
    ) -> Result<(), ArtifactError> {
        self.core
            .pending(decision.candidate, ArtifactPhase::AwaitingDecision)?;
        self.live(now)?;
        if !evidence_valid(evidence) {
            return self.core.fail(ArtifactError::InvalidEvidence);
        }
        if now >= decision.expires_at_tick {
            return self.core.fail(ArtifactError::Expired);
        }
        self.core.phase = ArtifactPhase::Stopped;
        self.verifier
            .decide(&self.binding, &decision, evidence)
            .map_err(ArtifactError::Verification)?;
        match decision.decision {
            Decision::Allow => {
                self.core.approved_until = Some(decision.expires_at_tick);
                self.core.phase = ArtifactPhase::Approved;
                Ok(())
            }
            Decision::Deny => self.core.fail(ArtifactError::Denied),
            Decision::Steer { .. } => self.core.fail(ArtifactError::Steered),
        }
    }
    pub fn accept_into(
        &mut self,
        key: ArtifactKey,
        now: u64,
        out: &mut Option<AcceptedArtifact>,
    ) -> Result<(), ArtifactError> {
        if out.is_some() {
            return self.core.fail(ArtifactError::OccupiedOutput);
        }
        self.core.pending(key, ArtifactPhase::Approved)?;
        self.live(now)?;
        self.core.accept(key, now, out)
    }
    pub fn disconnect(&mut self) {
        self.core.phase = ArtifactPhase::Stopped;
    }
    pub fn cancel(&mut self) {
        self.core.phase = ArtifactPhase::Stopped;
    }
    pub fn phase(&self) -> ArtifactPhase {
        self.core.phase
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
