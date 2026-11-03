use crate::{CandidateKey, GovernanceProfile, GovernedPurpose, SessionConfig};

pub const MAX_CONTROL_EVIDENCE_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionRequest {
    pub config: SessionConfig,
    pub purpose: GovernedPurpose,
    pub profile: GovernanceProfile,
    pub now_tick: u64,
}

/// Claims returned by a trusted host verifier, never sufficient alone to create
/// a governed session. The verifier must authenticate and bind the admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerBinding {
    pub identity: [u8; 32],
    pub generation: u64,
    pub purpose: GovernedPurpose,
    pub expires_at_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationError {
    Unauthenticated,
    Disconnected,
    Expired,
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
    Steer { steering_digest: [u8; 32] },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionEnvelope {
    pub candidate: CandidateKey,
    pub decision: Decision,
    pub expires_at_tick: u64,
}

/// Trusted embedding boundary, not an authentication implementation. The host
/// must authenticate real controllers, reject fixture authority, bind all
/// request fields and check liveness/revocation. An implementation returning
/// fabricated success defeats that trust boundary. Verifier allocations and
/// I/O are outside this crate's allocation-free metadata guarantee.
pub trait HostVerifier {
    fn verify_admission(
        &mut self,
        request: &AdmissionRequest,
        evidence: &[u8],
    ) -> Result<ControllerBinding, VerificationError>;

    fn verify_decision(
        &mut self,
        binding: &ControllerBinding,
        envelope: &DecisionEnvelope,
        evidence: &[u8],
    ) -> Result<(), VerificationError>;

    fn verify_liveness(
        &mut self,
        binding: &ControllerBinding,
        now_tick: u64,
    ) -> Result<(), VerificationError>;
}
