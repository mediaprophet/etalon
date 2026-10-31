use super::*;
use core::cell::Cell;

struct TestHost {
    outcome: Cell<PolicyOutcome>,
    policy_epoch: Cell<Option<u64>>,
    revocation: Cell<Option<RevocationSnapshot>>,
}

impl TestHost {
    fn new(outcome: PolicyOutcome) -> Self {
        Self {
            outcome: Cell::new(outcome),
            policy_epoch: Cell::new(Some(4)),
            revocation: Cell::new(Some(RevocationSnapshot::new(id(90), 8, false))),
        }
    }
}

impl PolicyEvaluator for TestHost {
    fn evaluate(&self, _request: &PolicyRequest) -> PolicyOutcome {
        self.outcome.get()
    }
}

impl PolicyEpochProvider for TestHost {
    fn current_policy_epoch(&self, _request: &PolicyRequest) -> Option<u64> {
        self.policy_epoch.get()
    }
}

impl RevocationProvider for TestHost {
    fn current_revocation(&self, _request: &PolicyRequest) -> Option<RevocationSnapshot> {
        self.revocation.get()
    }
}

fn id(value: u64) -> PolicyId {
    PolicyId::new(value).unwrap()
}

fn commitment(value: u8) -> DecisionCommitment {
    let mut bytes = [0; 32];
    bytes[0] = value;
    DecisionCommitment::new(bytes).unwrap()
}

fn evidence() -> EvidenceCommitment {
    let mut bytes = [0; 32];
    bytes[0] = 1;
    EvidenceCommitment::new(bytes).unwrap()
}

fn request() -> PolicyRequest {
    PolicyRequest::new(Some(id(1)), Some(id(2)), id(3), id(4), id(5), id(6), id(7), id(90), evidence())
}

#[test]
fn opaque_ids_and_commitments_reject_zero() {
    assert_eq!(PolicyId::new(0), Err(ContractError::InvalidIdentifier));
    assert_eq!(EvidenceCommitment::new([0; 32]), Err(ContractError::InvalidIdentifier));
    assert_eq!(DecisionCommitment::new([0; 32]), Err(ContractError::InvalidIdentifier));
}

#[test]
fn unknown_policy_fails_closed() {
    let host = TestHost::new(PolicyOutcome::Unknown);
    assert_eq!(
        authorize_once(request(), &host, &host, &host).unwrap_err(),
        PolicyError::Unknown
    );
}

#[test]
fn denial_fails_closed() {
    let host = TestHost::new(PolicyOutcome::Deny);
    assert_eq!(
        authorize_once(request(), &host, &host, &host).unwrap_err(),
        PolicyError::Denied
    );
}

#[test]
fn missing_epoch_or_revocation_state_fails_closed() {
    let host = TestHost::new(PolicyOutcome::Permit(commitment(9)));
    host.policy_epoch.set(None);
    assert_eq!(
        authorize_once(request(), &host, &host, &host).unwrap_err(),
        PolicyError::StateUnavailable
    );

    host.policy_epoch.set(Some(4));
    host.revocation.set(None);
    assert_eq!(
        authorize_once(request(), &host, &host, &host).unwrap_err(),
        PolicyError::StateUnavailable
    );
}

#[test]
fn revoked_request_never_authorizes() {
    let host = TestHost::new(PolicyOutcome::Permit(commitment(9)));
    host.revocation.set(Some(RevocationSnapshot::new(id(90), 9, true)));
    assert_eq!(
        authorize_once(request(), &host, &host, &host).unwrap_err(),
        PolicyError::Revoked
    );
}

#[test]
fn revocation_snapshot_must_match_declared_authority() {
    let host = TestHost::new(PolicyOutcome::Permit(commitment(9)));
    host.revocation.set(Some(RevocationSnapshot::new(id(91), 8, false)));
    assert_eq!(
        authorize_once(request(), &host, &host, &host).unwrap_err(),
        PolicyError::StateChanged
    );
}

#[test]
fn dispatch_rechecks_and_runs_once_for_exact_scope() {
    let host = TestHost::new(PolicyOutcome::Permit(commitment(9)));
    let authorized = authorize_once(request(), &host, &host, &host).unwrap();
    let calls = Cell::new(0);
    let result = dispatch_once(
        authorized,
        &request(),
        &host,
        &host,
        &host,
        |req, decision| {
            calls.set(calls.get() + 1);
            (req.operation().get(), decision.as_bytes()[0])
        },
    )
    .unwrap();
    assert_eq!(result, (4, 9));
    assert_eq!(calls.get(), 1);
}

#[test]
fn decision_is_bound_to_entire_request() {
    let host = TestHost::new(PolicyOutcome::Permit(commitment(9)));
    let authorized = authorize_once(request(), &host, &host, &host).unwrap();
    let mismatched = PolicyRequest::new(Some(id(1)), Some(id(2)), id(3), id(44), id(5), id(6), id(7), id(90), evidence());
    let calls = Cell::new(0);
    assert_eq!(
        dispatch_once(authorized, &mismatched, &host, &host, &host, |_, _| calls.set(1))
            .unwrap_err(),
        PolicyError::RequestMismatch
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn dispatch_rejects_changed_policy_or_revocation_epoch() {
    let host = TestHost::new(PolicyOutcome::Permit(commitment(9)));
    let authorized = authorize_once(request(), &host, &host, &host).unwrap();
    host.policy_epoch.set(Some(5));
    assert_eq!(
        dispatch_once(authorized, &request(), &host, &host, &host, |_, _| ())
            .unwrap_err(),
        PolicyError::StateChanged
    );

    let authorized = authorize_once(request(), &host, &host, &host).unwrap();
    host.revocation.set(Some(RevocationSnapshot::new(id(90), 10, false)));
    assert_eq!(
        dispatch_once(authorized, &request(), &host, &host, &host, |_, _| ())
            .unwrap_err(),
        PolicyError::StateChanged
    );
}

#[test]
fn actor_identifier_does_not_imply_guardian_authority() {
    let host = TestHost::new(PolicyOutcome::Unknown);
    let req = request();
    assert_eq!(req.actor().unwrap().get(), 2);
    assert_eq!(
        authorize_once(req, &host, &host, &host).unwrap_err(),
        PolicyError::Unknown
    );
}

#[test]
fn unspecified_subject_and_actor_are_supported_for_observation_contexts() {
    let host = TestHost::new(PolicyOutcome::Permit(commitment(9)));
    let public_science = PolicyRequest::new(None, None, id(3), id(4), id(5), id(6), id(7), id(90), evidence());
    let decision = authorize_once(public_science, &host, &host, &host).unwrap();
    let result = dispatch_once(decision, &public_science, &host, &host, &host, |req, _| {
        (req.subject(), req.actor())
    })
    .unwrap();
    assert_eq!(result, (None, None));
}
