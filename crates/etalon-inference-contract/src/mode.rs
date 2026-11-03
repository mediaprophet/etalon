#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    StandaloneDeveloper,
    SimulatedGovernanceTest,
    GovernedApplication,
    IntegratedGovernanceTest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GovernedPurpose {
    Application,
    IntegrationTest,
}

impl GovernedPurpose {
    pub(crate) fn mode(self) -> ExecutionMode {
        match self {
            Self::Application => ExecutionMode::GovernedApplication,
            Self::IntegrationTest => ExecutionMode::IntegratedGovernanceTest,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GovernanceProfile {
    ApprovalBeforePublication,
    PostTurn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Instrumentation {
    pub tracing: bool,
    pub profiling: bool,
}

/// The host must enforce uniqueness and monotonic generations across admissions.
/// This crate has no global registry: reusing an entire identity in a separately
/// constructed session can reuse candidate keys and is forbidden by that host contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionIdentity {
    pub session_id: u64,
    pub session_generation: u64,
    pub request_id: u64,
    pub request_generation: u64,
    pub model_id: [u8; 32],
    pub model_generation: u64,
    pub device_id: u64,
    pub device_generation: u64,
    pub policy_id: [u8; 32],
    pub policy_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionConfig {
    pub identity: SessionIdentity,
    pub instrumentation: Instrumentation,
    pub max_candidate_tokens: u32,
}

impl SessionConfig {
    pub(crate) fn valid(self, governed: bool) -> bool {
        self.identity.session_id != 0
            && self.identity.request_id != 0
            && self.identity.device_id != 0
            && self.identity.model_id != [0; 32]
            && self.max_candidate_tokens > 0
            && (!governed || self.identity.policy_id != [0; 32])
    }
}

/// Opaque caller digests; actual output, observations and digest computation are
/// owned by the embedding adapter. Zero digests are not an authenticity signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateOutput {
    pub output_digest: [u8; 32],
    pub observation_digest: [u8; 32],
    pub token_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateKey {
    pub identity: SessionIdentity,
    pub mode: ExecutionMode,
    pub sequence: u64,
    pub output: CandidateOutput,
}
