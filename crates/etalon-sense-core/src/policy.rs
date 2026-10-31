//! Generic, host-owned policy boundary for sensing operations.
//!
//! IDs and commitments are opaque labels. This module defines no legal,
//! guardian, personhood, or identity model and verifies no cryptographic proof.
//! A host must supply its policy evaluator, epoch source, and revocation source.
//! Unknown answers fail closed. The final check and dispatch callback happen in
//! one synchronous call so callers cannot accidentally reuse a decision token.
//! This module is an optional host-composition boundary: capture and hot DSP
//! do not call it automatically. Scientific observation contexts can be
//! described without specifying a human subject, guardian, or grant.
//! A host that enables policy determines which rules apply to the actual data:
//! a scientific or environmental label alone does not prove people or sensitive
//! data are absent. This interface does not authenticate providers or make
//! external policy updates atomic with I/O.

use crate::ContractError;

/// Opaque, nonzero identifier interpreted only by the host policy framework.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyId(u64);

impl PolicyId {
    pub const fn new(value: u64) -> Result<Self, ContractError> {
        if value == 0 {
            Err(ContractError::InvalidIdentifier)
        } else {
            Ok(Self(value))
        }
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Opaque fixed-size commitment to caller-provided evidence.
///
/// Nonzero means only that a caller supplied a value; it does not authenticate
/// evidence or establish its truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvidenceCommitment([u8; 32]);

impl EvidenceCommitment {
    pub const fn new(value: [u8; 32]) -> Result<Self, ContractError> {
        if is_zero(&value) {
            Err(ContractError::InvalidIdentifier)
        } else {
            Ok(Self(value))
        }
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Opaque fixed-size commitment returned by a host policy evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionCommitment([u8; 32]);

impl DecisionCommitment {
    pub const fn new(value: [u8; 32]) -> Result<Self, ContractError> {
        if is_zero(&value) {
            Err(ContractError::InvalidIdentifier)
        } else {
            Ok(Self(value))
        }
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

const fn is_zero(bytes: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// Exact operation request passed to the host policy evaluator.
///
/// Subject and actor may be explicitly unspecified for natural environments,
/// non-human species, public data, or observations without an identified
/// principal. They are deliberately not required to be humans. `actor` has no
/// guardian or elevated-capacity flag; the host decides what, if any, authority
/// relation applies to these identifiers and supplied evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyRequest {
    subject: Option<PolicyId>,
    actor: Option<PolicyId>,
    source: PolicyId,
    operation: PolicyId,
    modality: PolicyId,
    context: PolicyId,
    purpose: PolicyId,
    revocation_authority: PolicyId,
    evidence: EvidenceCommitment,
}

impl PolicyRequest {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        subject: Option<PolicyId>,
        actor: Option<PolicyId>,
        source: PolicyId,
        operation: PolicyId,
        modality: PolicyId,
        context: PolicyId,
        purpose: PolicyId,
        revocation_authority: PolicyId,
        evidence: EvidenceCommitment,
    ) -> Self {
        Self {
            subject,
            actor,
            source,
            operation,
            modality,
            context,
            purpose,
            revocation_authority,
            evidence,
        }
    }

    pub const fn subject(&self) -> Option<PolicyId> {
        self.subject
    }
    pub const fn actor(&self) -> Option<PolicyId> {
        self.actor
    }
    pub const fn source(&self) -> PolicyId {
        self.source
    }
    pub const fn operation(&self) -> PolicyId {
        self.operation
    }
    pub const fn modality(&self) -> PolicyId {
        self.modality
    }
    pub const fn context(&self) -> PolicyId {
        self.context
    }
    pub const fn purpose(&self) -> PolicyId {
        self.purpose
    }
    pub const fn revocation_authority(&self) -> PolicyId {
        self.revocation_authority
    }
    pub const fn evidence(&self) -> &EvidenceCommitment {
        &self.evidence
    }
}

/// Explicit policy evaluation result. Missing evaluator knowledge must be
/// returned as `Unknown`, never converted to permission by this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyOutcome {
    Permit(DecisionCommitment),
    Deny,
    Unknown,
}

/// Host-owned interpretation of identity, consent, ontology and policy.
pub trait PolicyEvaluator {
    fn evaluate(&self, request: &PolicyRequest) -> PolicyOutcome;
}

/// Version of the applicable policy state. `None` means the host cannot
/// establish a current epoch and the request must fail closed.
pub trait PolicyEpochProvider {
    fn current_policy_epoch(&self, request: &PolicyRequest) -> Option<u64>;
}

/// Current revocation state for this exact request scope. The host is
/// responsible for mapping the request to its revocation domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RevocationSnapshot {
    authority: PolicyId,
    epoch: u64,
    revoked: bool,
}

impl RevocationSnapshot {
    pub const fn new(authority: PolicyId, epoch: u64, revoked: bool) -> Self {
        Self {
            authority,
            epoch,
            revoked,
        }
    }

    pub const fn authority(self) -> PolicyId {
        self.authority
    }
    pub const fn epoch(self) -> u64 {
        self.epoch
    }
    pub const fn is_revoked(self) -> bool {
        self.revoked
    }
}

/// Host-owned revocation lookup. Unknown state is represented by `None`.
pub trait RevocationProvider {
    fn current_revocation(&self, request: &PolicyRequest) -> Option<RevocationSnapshot>;
}

/// Failure from policy evaluation, freshness checks, or one-shot dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    Contract(ContractError),
    Denied,
    Unknown,
    Revoked,
    StateUnavailable,
    StateChanged,
    RequestMismatch,
}

/// Non-copyable authorization snapshot, bound to one exact request and the
/// policy/revocation epochs observed during initial evaluation.
///
/// This is advisory until consumed by [`dispatch_once`]. It is not a portable
/// permit and does not itself authorize hardware access.
#[derive(Debug, PartialEq, Eq)]
pub struct OperationDecision {
    request: PolicyRequest,
    decision: DecisionCommitment,
    policy_epoch: u64,
    revocation: RevocationSnapshot,
}

/// Evaluate once and capture the exact policy and revocation state. Missing
/// state, unknown policy, denial, or revocation fails closed.
pub fn authorize_once<E, P, R>(
    request: PolicyRequest,
    evaluator: &E,
    epochs: &P,
    revocations: &R,
) -> Result<OperationDecision, PolicyError>
where
    E: PolicyEvaluator,
    P: PolicyEpochProvider,
    R: RevocationProvider,
{
    let policy_epoch = epochs
        .current_policy_epoch(&request)
        .ok_or(PolicyError::StateUnavailable)?;
    let revocation = revocations
        .current_revocation(&request)
        .ok_or(PolicyError::StateUnavailable)?;
    if revocation.is_revoked() {
        return Err(PolicyError::Revoked);
    }
    if revocation.authority() != request.revocation_authority() {
        return Err(PolicyError::StateChanged);
    }
    let decision = match evaluator.evaluate(&request) {
        PolicyOutcome::Permit(commitment) => commitment,
        PolicyOutcome::Deny => return Err(PolicyError::Denied),
        PolicyOutcome::Unknown => return Err(PolicyError::Unknown),
    };
    Ok(OperationDecision {
        request,
        decision,
        policy_epoch,
        revocation,
    })
}

/// Recheck exact request, policy decision, policy epoch, and revocation state
/// immediately before invoking the supplied one-shot dispatch callback.
///
/// The callback is invoked only if all state is known, unchanged, not revoked,
/// and the evaluator returns the same decision commitment. It executes
/// synchronously after those checks; callers must not queue work from it and
/// dispatch later without another policy boundary. Providers must offer a
/// coherent snapshot or the host must serialize policy/revocation mutations
/// across this check and callback; this API alone cannot make external state
/// changes atomic.
pub fn dispatch_once<E, P, R, F, T>(
    decision: OperationDecision,
    request: &PolicyRequest,
    evaluator: &E,
    epochs: &P,
    revocations: &R,
    dispatch: F,
) -> Result<T, PolicyError>
where
    E: PolicyEvaluator,
    P: PolicyEpochProvider,
    R: RevocationProvider,
    F: FnOnce(&PolicyRequest, &DecisionCommitment) -> T,
{
    if &decision.request != request {
        return Err(PolicyError::RequestMismatch);
    }
    let policy_epoch = epochs
        .current_policy_epoch(request)
        .ok_or(PolicyError::StateUnavailable)?;
    let revocation = revocations
        .current_revocation(request)
        .ok_or(PolicyError::StateUnavailable)?;
    if revocation.is_revoked() {
        return Err(PolicyError::Revoked);
    }
    if revocation.authority() != request.revocation_authority() {
        return Err(PolicyError::StateChanged);
    }
    if policy_epoch != decision.policy_epoch || revocation != decision.revocation {
        return Err(PolicyError::StateChanged);
    }
    match evaluator.evaluate(request) {
        PolicyOutcome::Permit(commitment) if commitment == decision.decision => {}
        PolicyOutcome::Permit(_) => return Err(PolicyError::StateChanged),
        PolicyOutcome::Deny => return Err(PolicyError::Denied),
        PolicyOutcome::Unknown => return Err(PolicyError::Unknown),
    }
    Ok(dispatch(request, &decision.decision))
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
