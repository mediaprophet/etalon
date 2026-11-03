use crate::*;
use core::cell::Cell;

fn config() -> SessionConfig {
    SessionConfig {
        identity: SessionIdentity {
            session_id: 1,
            session_generation: 1,
            request_id: 2,
            request_generation: 1,
            model_id: [3; 32],
            model_generation: 1,
            device_id: 4,
            device_generation: 1,
            policy_id: [5; 32],
            policy_generation: 1,
        },
        instrumentation: Instrumentation::default(),
        max_candidate_tokens: 8,
    }
}

fn output() -> CandidateOutput {
    CandidateOutput {
        output_digest: [6; 32],
        observation_digest: [7; 32],
        token_count: 1,
    }
}

fn request(purpose: GovernedPurpose) -> AdmissionRequest {
    AdmissionRequest {
        config: config(),
        purpose,
        profile: GovernanceProfile::ApprovalBeforePublication,
        now_tick: 0,
    }
}

fn envelope(candidate: CandidateKey, decision: Decision, expiry: u64) -> DecisionEnvelope {
    DecisionEnvelope {
        candidate,
        decision,
        expires_at_tick: expiry,
    }
}

// Synthetic verifier for contract tests only; provides no real authentication.
struct TestVerifier<'a> {
    live: &'a Cell<bool>,
    calls: &'a Cell<u32>,
    purpose: GovernedPurpose,
}

impl HostVerifier for TestVerifier<'_> {
    fn verify_admission(
        &mut self,
        _: &AdmissionRequest,
        evidence: &[u8],
    ) -> Result<ControllerBinding, VerificationError> {
        self.calls.set(self.calls.get() + 1);
        if evidence != b"admission" {
            return Err(VerificationError::Unauthenticated);
        }
        Ok(ControllerBinding {
            identity: [8; 32],
            generation: 9,
            purpose: self.purpose,
            expires_at_tick: 100,
        })
    }
    fn verify_decision(
        &mut self,
        binding: &ControllerBinding,
        _: &DecisionEnvelope,
        evidence: &[u8],
    ) -> Result<(), VerificationError> {
        assert_eq!(binding.identity, [8; 32]);
        assert_eq!(binding.generation, 9);
        self.calls.set(self.calls.get() + 1);
        if evidence == b"decision" {
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

fn governed<'a>(live: &'a Cell<bool>, calls: &'a Cell<u32>) -> GovernedSession<TestVerifier<'a>> {
    GovernedSession::admit(
        TestVerifier {
            live,
            calls,
            purpose: GovernedPurpose::Application,
        },
        request(GovernedPurpose::Application),
        b"admission",
    )
    .unwrap()
}

#[test]
fn developer_needs_no_policy_permission_or_controller() {
    let mut cfg = config();
    cfg.identity.policy_id = [0; 32];
    let mut session = DeveloperSession::standalone(cfg, 0).unwrap();
    let key = session.propose(output(), 1).unwrap();
    assert_eq!(session.snapshot().mode, ExecutionMode::StandaloneDeveloper);
    assert_eq!(session.commit(key, u64::MAX), Ok(key));
    assert_eq!(session.snapshot().phase, Phase::Ready);
}

#[test]
fn instrumentation_does_not_change_any_mode_or_commit_rule() {
    for tracing in [false, true] {
        for profiling in [false, true] {
            let mut cfg = config();
            cfg.instrumentation = Instrumentation { tracing, profiling };
            let mut developer = DeveloperSession::standalone(cfg, 0).unwrap();
            let key = developer.propose(output(), 1).unwrap();
            assert_eq!(developer.commit(key, 2), Ok(key));
            let mut fixture = FixtureSession::simulated(cfg, 0).unwrap();
            let key = fixture.propose(output(), 1).unwrap();
            assert_eq!(fixture.commit(key, 1), Err(SessionError::WrongPhase));
            fixture
                .decide(envelope(key, Decision::Allow, 9), 2)
                .unwrap();
            assert_eq!(fixture.commit(key, 3), Ok(key));
            assert_eq!(
                fixture.snapshot().mode,
                ExecutionMode::SimulatedGovernanceTest
            );
            let live = Cell::new(true);
            let calls = Cell::new(0);
            let mut req = request(GovernedPurpose::Application);
            req.config = cfg;
            let mut session = GovernedSession::admit(
                TestVerifier {
                    live: &live,
                    calls: &calls,
                    purpose: req.purpose,
                },
                req,
                b"admission",
            )
            .unwrap();
            let key = session.propose(output(), 1).unwrap();
            assert_eq!(session.commit(key, 1), Err(SessionError::WrongPhase));
            session
                .decide(envelope(key, Decision::Allow, 9), b"decision", 2)
                .unwrap();
            assert_eq!(session.commit(key, 3), Ok(key));
            assert_eq!(session.snapshot().mode, ExecutionMode::GovernedApplication);
        }
    }
}

#[test]
fn post_turn_is_explicitly_unsupported_before_verifier() {
    let live = Cell::new(true);
    let calls = Cell::new(0);
    let mut req = request(GovernedPurpose::Application);
    req.profile = GovernanceProfile::PostTurn;
    let result = GovernedSession::admit(
        TestVerifier {
            live: &live,
            calls: &calls,
            purpose: req.purpose,
        },
        req,
        b"admission",
    );
    assert!(matches!(result, Err(SessionError::UnsupportedProfile)));
    assert_eq!(calls.get(), 0);
}

#[test]
fn integrated_tests_require_matching_controller_purpose() {
    let live = Cell::new(true);
    let calls = Cell::new(0);
    let req = request(GovernedPurpose::IntegrationTest);
    let result = GovernedSession::admit(
        TestVerifier {
            live: &live,
            calls: &calls,
            purpose: GovernedPurpose::Application,
        },
        req,
        b"admission",
    );
    assert!(matches!(result, Err(SessionError::InvalidBinding)));
    let session = GovernedSession::admit(
        TestVerifier {
            live: &live,
            calls: &calls,
            purpose: req.purpose,
        },
        req,
        b"admission",
    )
    .unwrap();
    assert_eq!(
        session.snapshot().mode,
        ExecutionMode::IntegratedGovernanceTest
    );
}

#[test]
fn admission_rejects_missing_oversize_and_unauthenticated_evidence() {
    let live = Cell::new(true);
    let calls = Cell::new(0);
    for evidence in [&b""[..], &[0; MAX_CONTROL_EVIDENCE_BYTES + 1][..]] {
        let result = GovernedSession::admit(
            TestVerifier {
                live: &live,
                calls: &calls,
                purpose: GovernedPurpose::Application,
            },
            request(GovernedPurpose::Application),
            evidence,
        );
        assert!(matches!(result, Err(SessionError::InvalidEvidence)));
    }
    assert_eq!(calls.get(), 0);
    let result = GovernedSession::admit(
        TestVerifier {
            live: &live,
            calls: &calls,
            purpose: GovernedPurpose::Application,
        },
        request(GovernedPurpose::Application),
        b"fixture",
    );
    assert!(matches!(
        result,
        Err(SessionError::Verification(
            VerificationError::Unauthenticated
        ))
    ));
}

#[test]
fn forged_decision_stops_governed_session_without_fallback() {
    let live = Cell::new(true);
    let calls = Cell::new(0);
    let mut session = governed(&live, &calls);
    let key = session.propose(output(), 1).unwrap();
    assert_eq!(
        session.decide(envelope(key, Decision::Allow, 9), b"fixture", 2),
        Err(SessionError::Verification(
            VerificationError::Unauthenticated
        ))
    );
    assert_eq!(session.snapshot().phase, Phase::AbortRequired);
    assert_eq!(session.commit(key, 3), Err(SessionError::WrongPhase));
    session.acknowledge_restoration(key).unwrap();
    assert_eq!(session.snapshot().phase, Phase::Stopped);
    assert_eq!(session.propose(output(), 4), Err(SessionError::WrongPhase));
}

#[test]
fn loss_after_allow_prevents_publication() {
    let live = Cell::new(true);
    let calls = Cell::new(0);
    let mut session = governed(&live, &calls);
    let key = session.propose(output(), 1).unwrap();
    session
        .decide(envelope(key, Decision::Allow, 9), b"decision", 2)
        .unwrap();
    live.set(false);
    assert_eq!(
        session.commit(key, 3),
        Err(SessionError::Verification(VerificationError::Disconnected))
    );
    assert_eq!(session.snapshot().phase, Phase::AbortRequired);
    live.set(true);
    session.acknowledge_restoration(key).unwrap();
    assert_eq!(session.propose(output(), 4), Err(SessionError::WrongPhase));
}

#[test]
fn deny_requires_acknowledgement_and_cannot_restart() {
    let mut session = FixtureSession::simulated(config(), 0).unwrap();
    let key = session.propose(output(), 1).unwrap();
    session.decide(envelope(key, Decision::Deny, 9), 2).unwrap();
    assert_eq!(session.snapshot().phase, Phase::AbortRequired);
    let mut wrong = key;
    wrong.sequence += 1;
    assert_eq!(
        session.acknowledge_restoration(wrong),
        Err(SessionError::WrongCandidate)
    );
    session.acknowledge_restoration(key).unwrap();
    assert_eq!(session.snapshot().phase, Phase::Stopped);
}

#[test]
fn steer_requires_restoration_then_uses_new_monotonic_candidate() {
    let mut session = FixtureSession::simulated(config(), 0).unwrap();
    let key = session.propose(output(), 1).unwrap();
    session
        .decide(
            envelope(
                key,
                Decision::Steer {
                    steering_digest: [9; 32],
                },
                9,
            ),
            2,
        )
        .unwrap();
    assert_eq!(session.propose(output(), 3), Err(SessionError::WrongPhase));
    session.acknowledge_restoration(key).unwrap();
    let next = session.propose(output(), 4).unwrap();
    assert_eq!(next.sequence, key.sequence + 1);
    assert_eq!(
        session.decide(envelope(key, Decision::Allow, 9), 5),
        Err(SessionError::WrongCandidate)
    );
}

#[test]
fn duplicate_decision_and_commit_cannot_advance_state() {
    let live = Cell::new(true);
    let calls = Cell::new(0);
    let mut session = governed(&live, &calls);
    let key = session.propose(output(), 1).unwrap();
    let decision = envelope(key, Decision::Allow, 9);
    session.decide(decision, b"decision", 2).unwrap();
    assert_eq!(
        session.decide(decision, b"decision", 2),
        Err(SessionError::WrongPhase)
    );
    session.commit(key, 3).unwrap();
    assert_eq!(session.commit(key, 3), Err(SessionError::WrongPhase));
    assert_eq!(calls.get(), 2);
}

#[test]
fn identity_generations_and_fixture_mode_are_bound() {
    let live = Cell::new(true);
    let calls = Cell::new(0);
    let mut session = governed(&live, &calls);
    let key = session.propose(output(), 1).unwrap();
    for index in 0..7 {
        let mut wrong = key;
        match index {
            0 => wrong.identity.session_generation += 1,
            1 => wrong.identity.request_generation += 1,
            2 => wrong.identity.model_generation += 1,
            3 => wrong.identity.device_generation += 1,
            4 => wrong.identity.policy_generation += 1,
            5 => wrong.identity.session_id += 1,
            _ => wrong.mode = ExecutionMode::SimulatedGovernanceTest,
        }
        assert_eq!(
            session.decide(envelope(wrong, Decision::Allow, 9), b"decision", 2),
            Err(SessionError::WrongCandidate)
        );
    }
    assert_eq!(calls.get(), 1);
    assert_eq!(session.snapshot().phase, Phase::AwaitingDecision);
}

#[test]
fn accepted_allow_expires_at_commit_boundary_in_fixture_and_governed() {
    let mut fixture = FixtureSession::simulated(config(), 0).unwrap();
    let key = fixture.propose(output(), 1).unwrap();
    fixture
        .decide(envelope(key, Decision::Allow, 4), 2)
        .unwrap();
    assert_eq!(fixture.commit(key, 4), Err(SessionError::Expired));
    assert_eq!(fixture.snapshot().phase, Phase::AbortRequired);
    let live = Cell::new(true);
    let calls = Cell::new(0);
    let mut session = governed(&live, &calls);
    let key = session.propose(output(), 1).unwrap();
    session
        .decide(envelope(key, Decision::Allow, 4), b"decision", 2)
        .unwrap();
    assert_eq!(session.commit(key, 4), Err(SessionError::Expired));
    assert_eq!(session.snapshot().phase, Phase::AbortRequired);
}

#[test]
fn decision_and_controller_expiry_fail_closed() {
    let live = Cell::new(true);
    let calls = Cell::new(0);
    let mut session = governed(&live, &calls);
    let key = session.propose(output(), 1).unwrap();
    assert_eq!(
        session.decide(envelope(key, Decision::Allow, 2), b"decision", 2),
        Err(SessionError::Expired)
    );
    let mut session = governed(&live, &calls);
    let key = session.propose(output(), 1).unwrap();
    session
        .decide(envelope(key, Decision::Allow, 200), b"decision", 2)
        .unwrap();
    assert_eq!(session.commit(key, 100), Err(SessionError::Expired));
}

#[test]
fn clock_regression_and_counter_exhaustion_stop() {
    let mut session = DeveloperSession::standalone(config(), 4).unwrap();
    let key = session.propose(output(), 5).unwrap();
    assert_eq!(session.commit(key, 4), Err(SessionError::ClockRegression));
    session.acknowledge_restoration(key).unwrap();
    assert_eq!(session.snapshot().phase, Phase::Stopped);
    let mut session = DeveloperSession::standalone(config(), 0).unwrap();
    session.exhaust_counter_for_test();
    assert_eq!(
        session.propose(output(), 1),
        Err(SessionError::CounterExhausted)
    );
    assert_eq!(session.snapshot().phase, Phase::Stopped);
}

#[test]
fn invalid_config_and_token_bounds_are_rejected() {
    let mut cfg = config();
    cfg.max_candidate_tokens = 0;
    assert!(matches!(
        DeveloperSession::standalone(cfg, 0),
        Err(SessionError::InvalidConfig)
    ));
    let mut req = request(GovernedPurpose::Application);
    req.config.identity.policy_id = [0; 32];
    let live = Cell::new(true);
    let calls = Cell::new(0);
    assert!(matches!(
        GovernedSession::admit(
            TestVerifier {
                live: &live,
                calls: &calls,
                purpose: req.purpose
            },
            req,
            b"admission"
        ),
        Err(SessionError::InvalidConfig)
    ));
    let mut session = DeveloperSession::standalone(config(), 0).unwrap();
    for count in [0, 9, u32::MAX] {
        let mut candidate = output();
        candidate.token_count = count;
        assert_eq!(
            session.propose(candidate, 1),
            Err(SessionError::InvalidCandidate)
        );
        assert_eq!(session.snapshot().phase, Phase::Ready);
    }
}
