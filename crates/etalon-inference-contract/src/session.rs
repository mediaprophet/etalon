use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Ready,
    AwaitingDecision,
    CommitReady,
    AbortRequired,
    Stopped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbortReason {
    Denied,
    Steered,
    ControllerFailure,
    Expired,
    Cancelled,
    ClockRegression,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionError {
    InvalidConfig,
    UnsupportedProfile,
    InvalidEvidence,
    InvalidBinding,
    Verification(VerificationError),
    WrongPhase,
    WrongCandidate,
    Expired,
    ClockRegression,
    CounterExhausted,
    InvalidCandidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionSnapshot {
    pub config: SessionConfig,
    pub mode: ExecutionMode,
    pub phase: Phase,
    pub candidate: Option<CandidateKey>,
    pub abort_reason: Option<AbortReason>,
}

#[derive(Debug)]
struct Core {
    snapshot: SessionSnapshot,
    sequence: u64,
    last_tick: u64,
    stop_after_abort: bool,
    approved_until: Option<u64>,
}

impl Core {
    fn new(config: SessionConfig, mode: ExecutionMode, now: u64) -> Result<Self, SessionError> {
        let governed = matches!(
            mode,
            ExecutionMode::GovernedApplication | ExecutionMode::IntegratedGovernanceTest
        );
        if !config.valid(governed) {
            return Err(SessionError::InvalidConfig);
        }
        Ok(Self {
            snapshot: SessionSnapshot {
                config,
                mode,
                phase: Phase::Ready,
                candidate: None,
                abort_reason: None,
            },
            sequence: 0,
            last_tick: now,
            stop_after_abort: false,
            approved_until: None,
        })
    }

    fn clock(&mut self, now: u64) -> Result<(), SessionError> {
        if now < self.last_tick {
            self.abort(AbortReason::ClockRegression, true);
            return Err(SessionError::ClockRegression);
        }
        self.last_tick = now;
        Ok(())
    }

    fn candidate(
        &mut self,
        output: CandidateOutput,
        now: u64,
    ) -> Result<CandidateKey, SessionError> {
        self.clock(now)?;
        if self.snapshot.phase != Phase::Ready {
            return Err(SessionError::WrongPhase);
        }
        if output.token_count == 0 || output.token_count > self.snapshot.config.max_candidate_tokens
        {
            return Err(SessionError::InvalidCandidate);
        }
        let Some(sequence) = self.sequence.checked_add(1) else {
            self.snapshot.phase = Phase::Stopped;
            return Err(SessionError::CounterExhausted);
        };
        self.sequence = sequence;
        let key = CandidateKey {
            identity: self.snapshot.config.identity,
            mode: self.snapshot.mode,
            sequence,
            output,
        };
        self.snapshot.candidate = Some(key);
        self.approved_until = None;
        self.snapshot.abort_reason = None;
        self.snapshot.phase = if self.snapshot.mode == ExecutionMode::StandaloneDeveloper {
            Phase::CommitReady
        } else {
            Phase::AwaitingDecision
        };
        Ok(key)
    }

    fn pending(&self, key: CandidateKey, phase: Phase) -> Result<(), SessionError> {
        if self.snapshot.phase != phase {
            return Err(SessionError::WrongPhase);
        }
        if self.snapshot.candidate != Some(key) {
            return Err(SessionError::WrongCandidate);
        }
        Ok(())
    }

    fn decision(&mut self, envelope: DecisionEnvelope, now: u64) -> Result<(), SessionError> {
        self.clock(now)?;
        self.pending(envelope.candidate, Phase::AwaitingDecision)?;
        if now >= envelope.expires_at_tick {
            self.abort(AbortReason::Expired, true);
            return Err(SessionError::Expired);
        }
        match envelope.decision {
            Decision::Allow => {
                self.approved_until = Some(envelope.expires_at_tick);
                self.snapshot.phase = Phase::CommitReady;
            }
            Decision::Deny => self.abort(AbortReason::Denied, true),
            Decision::Steer { .. } => self.abort(AbortReason::Steered, false),
        }
        Ok(())
    }

    fn commit(&mut self, key: CandidateKey, now: u64) -> Result<CandidateKey, SessionError> {
        self.clock(now)?;
        self.pending(key, Phase::CommitReady)?;
        if self.snapshot.mode != ExecutionMode::StandaloneDeveloper
            && self.approved_until.is_none_or(|deadline| now >= deadline)
        {
            self.abort(AbortReason::Expired, true);
            return Err(SessionError::Expired);
        }
        self.approved_until = None;
        self.snapshot.candidate = None;
        self.snapshot.phase = Phase::Ready;
        Ok(key)
    }

    fn abort(&mut self, reason: AbortReason, stop: bool) {
        self.approved_until = None;
        self.stop_after_abort |= stop;
        self.snapshot.abort_reason = Some(reason);
        self.snapshot.phase = if self.snapshot.candidate.is_some() {
            Phase::AbortRequired
        } else {
            Phase::Stopped
        };
    }

    fn acknowledge_restoration(&mut self, key: CandidateKey) -> Result<(), SessionError> {
        self.pending(key, Phase::AbortRequired)?;
        self.approved_until = None;
        self.snapshot.candidate = None;
        self.snapshot.phase = if self.stop_after_abort {
            Phase::Stopped
        } else {
            Phase::Ready
        };
        Ok(())
    }
}

#[derive(Debug)]
pub struct DeveloperSession {
    core: Core,
}

impl DeveloperSession {
    #[cfg(test)]
    pub(crate) fn exhaust_counter_for_test(&mut self) {
        self.core.sequence = u64::MAX;
    }
    pub fn standalone(config: SessionConfig, now: u64) -> Result<Self, SessionError> {
        Ok(Self {
            core: Core::new(config, ExecutionMode::StandaloneDeveloper, now)?,
        })
    }
    pub fn propose(
        &mut self,
        output: CandidateOutput,
        now: u64,
    ) -> Result<CandidateKey, SessionError> {
        self.core.candidate(output, now)
    }
    pub fn commit(&mut self, key: CandidateKey, now: u64) -> Result<CandidateKey, SessionError> {
        self.core.commit(key, now)
    }
    pub fn snapshot(&self) -> SessionSnapshot {
        self.core.snapshot
    }
    pub fn cancel(&mut self) {
        self.core.abort(AbortReason::Cancelled, true);
    }
    pub fn acknowledge_restoration(&mut self, key: CandidateKey) -> Result<(), SessionError> {
        self.core.acknowledge_restoration(key)
    }
}

#[derive(Debug)]
pub struct FixtureSession {
    core: Core,
}

impl FixtureSession {
    pub fn simulated(config: SessionConfig, now: u64) -> Result<Self, SessionError> {
        Ok(Self {
            core: Core::new(config, ExecutionMode::SimulatedGovernanceTest, now)?,
        })
    }
    pub fn propose(
        &mut self,
        output: CandidateOutput,
        now: u64,
    ) -> Result<CandidateKey, SessionError> {
        self.core.candidate(output, now)
    }
    pub fn decide(&mut self, envelope: DecisionEnvelope, now: u64) -> Result<(), SessionError> {
        self.core.decision(envelope, now)
    }
    pub fn commit(&mut self, key: CandidateKey, now: u64) -> Result<CandidateKey, SessionError> {
        self.core.commit(key, now)
    }
    pub fn snapshot(&self) -> SessionSnapshot {
        self.core.snapshot
    }
    pub fn disconnect(&mut self) {
        self.core.abort(AbortReason::ControllerFailure, true);
    }
    pub fn cancel(&mut self) {
        self.core.abort(AbortReason::Cancelled, true);
    }
    pub fn acknowledge_restoration(&mut self, key: CandidateKey) -> Result<(), SessionError> {
        self.core.acknowledge_restoration(key)
    }
}

#[derive(Debug)]
pub struct GovernedSession<V> {
    core: Core,
    verifier: V,
    binding: ControllerBinding,
}

fn evidence_valid(evidence: &[u8]) -> bool {
    !evidence.is_empty() && evidence.len() <= MAX_CONTROL_EVIDENCE_BYTES
}

impl<V: HostVerifier> GovernedSession<V> {
    /// The supplied verifier is trusted application code. Successful admission
    /// is not this library's proof of a real Sentinel integration.
    pub fn admit(
        mut verifier: V,
        request: AdmissionRequest,
        evidence: &[u8],
    ) -> Result<Self, SessionError> {
        if request.profile != GovernanceProfile::ApprovalBeforePublication {
            return Err(SessionError::UnsupportedProfile);
        }
        let core = Core::new(request.config, request.purpose.mode(), request.now_tick)?;
        if !evidence_valid(evidence) {
            return Err(SessionError::InvalidEvidence);
        }
        let binding = verifier
            .verify_admission(&request, evidence)
            .map_err(SessionError::Verification)?;
        if binding.identity == [0; 32] || binding.purpose != request.purpose {
            return Err(SessionError::InvalidBinding);
        }
        if request.now_tick >= binding.expires_at_tick {
            return Err(SessionError::Expired);
        }
        Ok(Self {
            core,
            verifier,
            binding,
        })
    }

    fn live(&mut self, now: u64) -> Result<(), SessionError> {
        self.core.clock(now)?;
        if matches!(
            self.core.snapshot.phase,
            Phase::AbortRequired | Phase::Stopped
        ) {
            return Err(SessionError::WrongPhase);
        }
        if now >= self.binding.expires_at_tick {
            self.core.abort(AbortReason::Expired, true);
            return Err(SessionError::Expired);
        }
        if let Err(error) = self.verifier.verify_liveness(&self.binding, now) {
            self.core.abort(AbortReason::ControllerFailure, true);
            return Err(SessionError::Verification(error));
        }
        Ok(())
    }

    pub fn propose(
        &mut self,
        output: CandidateOutput,
        now: u64,
    ) -> Result<CandidateKey, SessionError> {
        self.live(now)?;
        self.core.candidate(output, now)
    }

    pub fn decide(
        &mut self,
        envelope: DecisionEnvelope,
        evidence: &[u8],
        now: u64,
    ) -> Result<(), SessionError> {
        self.core
            .pending(envelope.candidate, Phase::AwaitingDecision)?;
        self.live(now)?;
        if !evidence_valid(evidence) {
            self.core.abort(AbortReason::ControllerFailure, true);
            return Err(SessionError::InvalidEvidence);
        }
        if let Err(error) = self
            .verifier
            .verify_decision(&self.binding, &envelope, evidence)
        {
            self.core.abort(AbortReason::ControllerFailure, true);
            return Err(SessionError::Verification(error));
        }
        self.core.decision(envelope, now)
    }

    /// Advances metadata only. The adapter owns the actual publication sink.
    pub fn commit(&mut self, key: CandidateKey, now: u64) -> Result<CandidateKey, SessionError> {
        self.core.pending(key, Phase::CommitReady)?;
        self.live(now)?;
        self.core.commit(key, now)
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        self.core.snapshot
    }
    pub fn controller(&self) -> ControllerBinding {
        self.binding
    }
    pub fn disconnect(&mut self) {
        self.core.abort(AbortReason::ControllerFailure, true);
    }
    pub fn cancel(&mut self) {
        self.core.abort(AbortReason::Cancelled, true);
    }
    pub fn acknowledge_restoration(&mut self, key: CandidateKey) -> Result<(), SessionError> {
        self.core.acknowledge_restoration(key)
    }
}
