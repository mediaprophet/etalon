use crate::{
    ControllerBinding, Decision, ExecutionMode, GovernedPurpose, Instrumentation, SessionIdentity,
    VerificationError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactMedia {
    Utf8Text,
    Utf8Html,
    Binary,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactSource {
    RenderedCompletion,
    GroundedReplacement,
    SemanticProjection,
    DeveloperArtifact,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentEventDomain {
    Token,
    Artifact,
}

/// A caller declaration for the host to verify, not proof of prior acceptance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParentEventBinding {
    pub domain: ParentEventDomain,
    pub identity: SessionIdentity,
    pub sequence: u64,
    pub record_digest: [u8; 32],
}

/// Digests must cover the exact immutable backing later delivered by the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactOutput {
    pub media: ArtifactMedia,
    pub source: ArtifactSource,
    pub byte_len: u64,
    pub content_digest: [u8; 32],
    pub provenance_digest: [u8; 32],
    pub observation_digest: [u8; 32],
    pub parent: Option<ParentEventBinding>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactConfig {
    pub identity: SessionIdentity,
    pub instrumentation: Instrumentation,
    pub max_artifact_bytes: u64,
    pub allow_empty: bool,
    pub max_events: u64,
}
impl ArtifactConfig {
    pub(super) fn valid(self, governed: bool) -> bool {
        self.identity.session_id != 0
            && self.identity.request_id != 0
            && self.identity.device_id != 0
            && self.identity.model_id != [0; 32]
            && (!governed || self.identity.policy_id != [0; 32])
            && self.max_events > 0
            && (self.max_artifact_bytes > 0 || self.allow_empty)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactKey {
    pub identity: SessionIdentity,
    pub mode: ExecutionMode,
    pub event_sequence: u64,
    pub output: ArtifactOutput,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactAdmission {
    pub config: ArtifactConfig,
    pub purpose: GovernedPurpose,
    pub now_tick: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactDecision {
    pub candidate: ArtifactKey,
    pub decision: Decision,
    pub expires_at_tick: u64,
}

/// Trusted host boundary; implementations must authenticate the real authority,
/// bind every admission/decision field and revalidate current actor/controller.
/// Token decisions cannot be submitted through this distinct artifact interface.
pub trait ArtifactVerifier {
    fn admit(
        &mut self,
        request: &ArtifactAdmission,
        evidence: &[u8],
    ) -> Result<ControllerBinding, VerificationError>;
    fn decide(
        &mut self,
        binding: &ControllerBinding,
        decision: &ArtifactDecision,
        evidence: &[u8],
    ) -> Result<(), VerificationError>;
    fn live(&mut self, binding: &ControllerBinding, now_tick: u64)
        -> Result<(), VerificationError>;
}
