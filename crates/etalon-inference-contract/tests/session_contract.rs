use etalon_inference_contract::*;
use std::{cell::Cell, rc::Rc};

fn config() -> SessionConfig {
    SessionConfig {
        identity: SessionIdentity {
            session_id: 7,
            session_generation: 2,
            request_id: 11,
            request_generation: 3,
            model_id: [1; 32],
            model_generation: 5,
            device_id: 13,
            device_generation: 7,
            policy_id: [2; 32],
            policy_generation: 11,
        },
        instrumentation: Instrumentation {
            tracing: true,
            profiling: true,
        },
        max_candidate_tokens: 4,
    }
}

fn output() -> CandidateOutput {
    CandidateOutput {
        output_digest: [3; 32],
        observation_digest: [4; 32],
        token_count: 2,
    }
}

fn envelope(candidate: CandidateKey, decision: Decision, expires: u64) -> DecisionEnvelope {
    DecisionEnvelope {
        candidate,
        decision,
        expires_at_tick: expires,
    }
}

// This fixture models a trusted host's decisions; it does not authenticate Qualia.
struct TestHost {
    live: Rc<Cell<bool>>,
}
impl HostVerifier for TestHost {
    fn verify_admission(
        &mut self,
        request: &AdmissionRequest,
        evidence: &[u8],
    ) -> Result<ControllerBinding, VerificationError> {
        if evidence != b"test-authorized" {
            return Err(VerificationError::Unauthenticated);
        }
        Ok(ControllerBinding {
            identity: [9; 32],
            generation: 4,
            purpose: request.purpose,
            expires_at_tick: 100,
        })
    }
    fn verify_decision(
        &mut self,
        _: &ControllerBinding,
        _: &DecisionEnvelope,
        evidence: &[u8],
    ) -> Result<(), VerificationError> {
        if evidence == b"test-authorized" {
            Ok(())
        } else {
            Err(VerificationError::Unauthenticated)
        }
    }
    fn verify_liveness(&mut self, _: &ControllerBinding, _: u64) -> Result<(), VerificationError> {
        if self.live.get() {
            Ok(())
        } else {
            Err(VerificationError::Disconnected)
        }
    }
}

fn request() -> AdmissionRequest {
    AdmissionRequest {
        config: config(),
        purpose: GovernedPurpose::Application,
        profile: GovernanceProfile::ApprovalBeforePublication,
        now_tick: 0,
    }
}

fn governed() -> (GovernedSession<TestHost>, Rc<Cell<bool>>) {
    let live = Rc::new(Cell::new(true));
    let session = GovernedSession::admit(
        TestHost { live: live.clone() },
        request(),
        b"test-authorized",
    )
    .unwrap();
    (session, live)
}

#[test]
fn developer_needs_no_policy_controller_or_permissions_and_debug_does_not_change_mode() {
    let mut c = config();
    c.identity.policy_id = [0; 32];
    let mut s = DeveloperSession::standalone(c, 0).unwrap();
    let candidate = s.propose(output(), 1).unwrap();
    assert_eq!(candidate.mode, ExecutionMode::StandaloneDeveloper);
    assert_eq!(s.snapshot().config.instrumentation, c.instrumentation);
    assert_eq!(s.commit(candidate, 2), Ok(candidate));
    assert_eq!(s.commit(candidate, 2), Err(SessionError::WrongPhase));
    assert_eq!(s.snapshot().phase, Phase::Ready);
}

#[test]
fn absent_fixture_authority_and_post_turn_are_explicitly_refused() {
    for evidence in [b"".as_slice(), b"fixture-controller".as_slice()] {
        let result = GovernedSession::admit(
            TestHost {
                live: Rc::new(Cell::new(true)),
            },
            request(),
            evidence,
        );
        assert!(result.is_err());
    }
    let mut r = request();
    r.profile = GovernanceProfile::PostTurn;
    assert!(matches!(
        GovernedSession::admit(
            TestHost {
                live: Rc::new(Cell::new(true))
            },
            r,
            b"test-authorized"
        ),
        Err(SessionError::UnsupportedProfile)
    ));
    let s = FixtureSession::simulated(config(), 0).unwrap();
    assert_eq!(s.snapshot().mode, ExecutionMode::SimulatedGovernanceTest);
    let mut r = request();
    r.purpose = GovernedPurpose::IntegrationTest;
    let s = GovernedSession::admit(
        TestHost {
            live: Rc::new(Cell::new(true)),
        },
        r,
        b"test-authorized",
    )
    .unwrap();
    assert_eq!(s.snapshot().mode, ExecutionMode::IntegratedGovernanceTest);
}

#[test]
fn every_candidate_identity_field_and_digest_is_bound_before_decision() {
    let (mut s, _) = governed();
    let key = s.propose(output(), 1).unwrap();
    let mut wrong = [key; 15];
    wrong[0].identity.session_id += 1;
    wrong[1].identity.session_generation += 1;
    wrong[2].identity.request_id += 1;
    wrong[3].identity.request_generation += 1;
    wrong[4].identity.model_id[0] ^= 1;
    wrong[5].identity.model_generation += 1;
    wrong[6].identity.device_id += 1;
    wrong[7].identity.device_generation += 1;
    wrong[8].identity.policy_id[0] ^= 1;
    wrong[9].identity.policy_generation += 1;
    wrong[10].sequence += 1;
    wrong[11].mode = ExecutionMode::StandaloneDeveloper;
    wrong[12].output.output_digest[0] ^= 1;
    wrong[13].output.observation_digest[0] ^= 1;
    wrong[14].output.token_count += 1;
    for invalid in wrong {
        assert_eq!(
            s.decide(
                envelope(invalid, Decision::Allow, 50),
                b"test-authorized",
                2
            ),
            Err(SessionError::WrongCandidate)
        );
        assert_eq!(s.snapshot().candidate, Some(key));
        assert_eq!(s.snapshot().phase, Phase::AwaitingDecision);
    }
    assert_eq!(
        s.decide(envelope(key, Decision::Allow, 50), b"test-authorized", 2),
        Ok(())
    );
    assert_eq!(s.commit(key, 3), Ok(key));
}

#[test]
fn duplicate_and_stale_decisions_never_commit_twice() {
    let (mut s, _) = governed();
    let first = s.propose(output(), 1).unwrap();
    let allow = envelope(first, Decision::Allow, 50);
    s.decide(allow, b"test-authorized", 2).unwrap();
    assert_eq!(
        s.decide(allow, b"test-authorized", 2),
        Err(SessionError::WrongPhase)
    );
    s.commit(first, 3).unwrap();
    assert_eq!(s.commit(first, 3), Err(SessionError::WrongPhase));
    let second = s.propose(output(), 4).unwrap();
    assert_eq!(second.sequence, first.sequence + 1);
    assert_eq!(
        s.decide(allow, b"test-authorized", 4),
        Err(SessionError::WrongCandidate)
    );
    assert_eq!(s.snapshot().candidate, Some(second));
}

#[test]
fn deny_and_loss_require_matching_restoration_and_do_not_commit() {
    for disconnect in [false, true] {
        let (mut s, _) = governed();
        let key = s.propose(output(), 1).unwrap();
        if disconnect {
            s.disconnect();
        } else {
            s.decide(envelope(key, Decision::Deny, 50), b"test-authorized", 2)
                .unwrap();
        }
        assert_eq!(s.snapshot().phase, Phase::AbortRequired);
        assert!(s.commit(key, 3).is_err());
        let mut wrong = key;
        wrong.sequence += 1;
        assert_eq!(
            s.acknowledge_restoration(wrong),
            Err(SessionError::WrongCandidate)
        );
        s.acknowledge_restoration(key).unwrap();
        assert_eq!(s.snapshot().phase, Phase::Stopped);
        assert_eq!(s.snapshot().candidate, None);
        assert!(s.propose(output(), 4).is_err());
    }
}

#[test]
fn loss_after_allow_and_invalid_decision_evidence_fail_closed() {
    let (mut s, live) = governed();
    let key = s.propose(output(), 1).unwrap();
    s.decide(envelope(key, Decision::Allow, 50), b"test-authorized", 2)
        .unwrap();
    live.set(false);
    assert_eq!(
        s.commit(key, 3),
        Err(SessionError::Verification(VerificationError::Disconnected))
    );
    assert_eq!(s.snapshot().phase, Phase::AbortRequired);
    s.acknowledge_restoration(key).unwrap();
    assert_eq!(s.snapshot().phase, Phase::Stopped);
    let (mut s, _) = governed();
    let key = s.propose(output(), 1).unwrap();
    assert!(s
        .decide(envelope(key, Decision::Allow, 50), b"invalid", 2)
        .is_err());
    assert_eq!(s.snapshot().phase, Phase::AbortRequired);
}

#[test]
fn accepted_allow_expiry_is_enforced_at_commit_for_fixture_and_governed() {
    let mut fixture = FixtureSession::simulated(config(), 0).unwrap();
    let key = fixture.propose(output(), 0).unwrap();
    fixture
        .decide(envelope(key, Decision::Allow, 2), 1)
        .unwrap();
    assert_eq!(fixture.commit(key, 2), Err(SessionError::Expired));
    assert_eq!(fixture.snapshot().phase, Phase::AbortRequired);
    let (mut s, _) = governed();
    let key = s.propose(output(), 0).unwrap();
    s.decide(envelope(key, Decision::Allow, 2), b"test-authorized", 1)
        .unwrap();
    assert_eq!(s.commit(key, 2), Err(SessionError::Expired));
    assert_eq!(s.snapshot().phase, Phase::AbortRequired);
}

#[test]
fn expired_decision_and_clock_regression_never_allow() {
    let (mut s, _) = governed();
    let key = s.propose(output(), 1).unwrap();
    assert_eq!(
        s.decide(envelope(key, Decision::Allow, 2), b"test-authorized", 2),
        Err(SessionError::Expired)
    );
    assert_eq!(s.snapshot().phase, Phase::AbortRequired);
    let mut dev = DeveloperSession::standalone(config(), 10).unwrap();
    let key = dev.propose(output(), 11).unwrap();
    assert_eq!(dev.commit(key, 10), Err(SessionError::ClockRegression));
    assert_eq!(dev.snapshot().phase, Phase::AbortRequired);
}

#[test]
fn invalid_config_and_token_bound_are_rejected_without_pending_candidate() {
    let mut invalid = [config(); 5];
    invalid[0].identity.session_id = 0;
    invalid[1].identity.request_id = 0;
    invalid[2].identity.device_id = 0;
    invalid[3].identity.model_id = [0; 32];
    invalid[4].max_candidate_tokens = 0;
    for c in invalid {
        assert!(DeveloperSession::standalone(c, 0).is_err());
    }
    let mut dev = DeveloperSession::standalone(config(), 0).unwrap();
    for count in [0, 5, u32::MAX] {
        assert_eq!(
            dev.propose(
                CandidateOutput {
                    token_count: count,
                    ..output()
                },
                1
            ),
            Err(SessionError::InvalidCandidate)
        );
        assert_eq!(dev.snapshot().phase, Phase::Ready);
        assert_eq!(dev.snapshot().candidate, None);
    }
}
