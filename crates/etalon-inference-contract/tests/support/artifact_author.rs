use super::*;
use crate::{GovernedPurpose, Instrumentation, SessionIdentity};
use std::cell::Cell;

fn identity() -> SessionIdentity {
    SessionIdentity {
        session_id: 1,
        session_generation: 2,
        request_id: 3,
        request_generation: 4,
        model_id: [5; 32],
        model_generation: 6,
        device_id: 7,
        device_generation: 8,
        policy_id: [9; 32],
        policy_generation: 10,
    }
}
fn config() -> ArtifactConfig {
    ArtifactConfig {
        identity: identity(),
        instrumentation: Instrumentation {
            tracing: true,
            profiling: true,
        },
        max_artifact_bytes: 16,
        allow_empty: false,
        max_events: 8,
    }
}
fn output() -> ArtifactOutput {
    ArtifactOutput {
        media: ArtifactMedia::Utf8Html,
        source: ArtifactSource::GroundedReplacement,
        byte_len: 7,
        content_digest: [11; 32],
        provenance_digest: [12; 32],
        observation_digest: [13; 32],
        parent: Some(ParentEventBinding {
            domain: ParentEventDomain::Token,
            identity: identity(),
            sequence: 1,
            record_digest: [14; 32],
        }),
    }
}
// Deliberately a state-machine fixture. It is not a real controller or Sentinel.
#[derive(Debug)]
struct Fixture<'a> {
    fault: &'a Cell<u8>,
}
impl ArtifactVerifier for Fixture<'_> {
    fn admit(
        &mut self,
        r: &ArtifactAdmission,
        e: &[u8],
    ) -> Result<ControllerBinding, VerificationError> {
        if e != b"admit" {
            return Err(VerificationError::Unauthenticated);
        }
        Ok(ControllerBinding {
            identity: [15; 32],
            generation: 1,
            purpose: r.purpose,
            expires_at_tick: 100,
        })
    }
    fn decide(
        &mut self,
        _: &ControllerBinding,
        _: &ArtifactDecision,
        e: &[u8],
    ) -> Result<(), VerificationError> {
        if self.fault.get() == 3 {
            panic!("decision fixture unwind");
        }
        if e == b"decide" {
            Ok(())
        } else {
            Err(VerificationError::Unauthenticated)
        }
    }
    fn live(&mut self, _: &ControllerBinding, _: u64) -> Result<(), VerificationError> {
        match self.fault.get() {
            1 => Err(VerificationError::Disconnected),
            2 => panic!("liveness fixture unwind"),
            _ => Ok(()),
        }
    }
}
fn session(f: &Cell<u8>) -> GovernedArtifactSession<Fixture<'_>> {
    GovernedArtifactSession::admit(
        Fixture { fault: f },
        ArtifactAdmission {
            config: config(),
            purpose: GovernedPurpose::IntegrationTest,
            now_tick: 1,
        },
        b"admit",
    )
    .unwrap()
}
fn allow(s: &mut GovernedArtifactSession<Fixture<'_>>, k: ArtifactKey) {
    s.decide(
        ArtifactDecision {
            candidate: k,
            decision: Decision::Allow,
            expires_at_tick: 50,
        },
        b"decide",
        3,
    )
    .unwrap();
}

#[test]
fn explicit_developer_and_governed_instrumentation_keep_distinct_authority_paths() {
    let mut d = DeveloperArtifactSession::new(config(), 1).unwrap();
    let k = d.propose(output(), 2).unwrap();
    assert_eq!(k.mode, ExecutionMode::StandaloneDeveloper);
    let mut out = None;
    d.accept_into(k, 3, &mut out).unwrap();
    assert_eq!(out.take().unwrap().into_key(), k);
    let f = Cell::new(0);
    let mut s = session(&f);
    let k = s.propose(output(), 2).unwrap();
    assert_eq!(s.phase(), ArtifactPhase::AwaitingDecision);
    assert_eq!(k.mode, ExecutionMode::IntegratedGovernanceTest);
    assert_eq!(
        s.accept_into(k, 3, &mut None),
        Err(ArtifactError::WrongPhase)
    );
    assert_eq!(s.phase(), ArtifactPhase::Stopped);
}
#[test]
fn allow_enqueues_once_and_event_sequence_prevents_replay() {
    let f = Cell::new(0);
    let mut s = session(&f);
    let k = s.propose(output(), 2).unwrap();
    allow(&mut s, k);
    let mut out = None;
    s.accept_into(k, 4, &mut out).unwrap();
    assert_eq!(out.take().unwrap().into_key(), k);
    let next = s.propose(output(), 5).unwrap();
    assert_eq!(next.event_sequence, k.event_sequence + 1);
    assert_eq!(
        s.decide(
            ArtifactDecision {
                candidate: k,
                decision: Decision::Allow,
                expires_at_tick: 50
            },
            b"decide",
            6
        ),
        Err(ArtifactError::WrongCandidate)
    );
    assert_eq!(s.phase(), ArtifactPhase::Stopped);
}
#[test]
fn every_payload_identity_parent_and_mode_change_refuses_before_verification() {
    let changes: &[fn(&mut ArtifactKey)] = &[
        |k| k.output.media = ArtifactMedia::Binary,
        |k| k.output.source = ArtifactSource::RenderedCompletion,
        |k| k.output.byte_len += 1,
        |k| k.output.content_digest[0] ^= 1,
        |k| k.output.provenance_digest[0] ^= 1,
        |k| k.output.observation_digest[0] ^= 1,
        |k| k.output.parent = None,
        |k| k.output.parent.as_mut().unwrap().domain = ParentEventDomain::Artifact,
        |k| k.output.parent.as_mut().unwrap().sequence += 1,
        |k| k.output.parent.as_mut().unwrap().record_digest[0] ^= 1,
        |k| {
            k.output
                .parent
                .as_mut()
                .unwrap()
                .identity
                .request_generation += 1
        },
        |k| k.identity.session_id += 1,
        |k| k.identity.session_generation += 1,
        |k| k.identity.request_id += 1,
        |k| k.identity.request_generation += 1,
        |k| k.identity.model_id[0] ^= 1,
        |k| k.identity.model_generation += 1,
        |k| k.identity.device_id += 1,
        |k| k.identity.device_generation += 1,
        |k| k.identity.policy_id[0] ^= 1,
        |k| k.identity.policy_generation += 1,
        |k| k.event_sequence += 1,
        |k| k.mode = ExecutionMode::StandaloneDeveloper,
    ];
    for change in changes {
        let f = Cell::new(0);
        let mut s = session(&f);
        let mut k = s.propose(output(), 2).unwrap();
        change(&mut k);
        assert_eq!(
            s.decide(
                ArtifactDecision {
                    candidate: k,
                    decision: Decision::Allow,
                    expires_at_tick: 50
                },
                b"decide",
                3
            ),
            Err(ArtifactError::WrongCandidate)
        );
        assert_eq!(s.phase(), ArtifactPhase::Stopped);
    }
}
#[test]
fn occupied_output_never_overwrites_or_consumes_another_record() {
    let mut d = DeveloperArtifactSession::new(config(), 1).unwrap();
    let first = d.propose(output(), 2).unwrap();
    let mut out = None;
    d.accept_into(first, 3, &mut out).unwrap();
    let f = Cell::new(0);
    let mut s = session(&f);
    let k = s.propose(output(), 2).unwrap();
    allow(&mut s, k);
    assert_eq!(
        s.accept_into(k, 4, &mut out),
        Err(ArtifactError::OccupiedOutput)
    );
    assert_eq!(out.unwrap().into_key(), first);
    assert_eq!(s.phase(), ArtifactPhase::Stopped);
}
#[test]
fn deny_steer_loss_expiry_clock_and_unwind_are_terminal() {
    for decision in [
        Decision::Deny,
        Decision::Steer {
            steering_digest: [1; 32],
        },
    ] {
        let f = Cell::new(0);
        let mut s = session(&f);
        let k = s.propose(output(), 2).unwrap();
        assert!(
            s.decide(
                ArtifactDecision {
                    candidate: k,
                    decision,
                    expires_at_tick: 50
                },
                b"decide",
                3
            )
            .is_err()
        );
        assert!(s.accept_into(k, 4, &mut None).is_err());
        assert_eq!(s.phase(), ArtifactPhase::Stopped);
    }
    for fault in [1, 2, 3] {
        let f = Cell::new(0);
        let mut s = session(&f);
        let k = s.propose(output(), 2).unwrap();
        if fault != 3 {
            allow(&mut s, k);
        }
        f.set(fault);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if fault == 3 {
                s.decide(
                    ArtifactDecision {
                        candidate: k,
                        decision: Decision::Allow,
                        expires_at_tick: 50,
                    },
                    b"decide",
                    4,
                )
            } else {
                s.accept_into(k, 4, &mut None)
            }
        }));
        assert!(result.is_err() || result.unwrap().is_err());
        assert_eq!(s.phase(), ArtifactPhase::Stopped);
    }
    for now in [0, 50, 100] {
        let f = Cell::new(0);
        let mut s = session(&f);
        let k = s.propose(output(), 2).unwrap();
        allow(&mut s, k);
        assert!(s.accept_into(k, now, &mut None).is_err());
        assert_eq!(s.phase(), ArtifactPhase::Stopped);
    }
}
#[test]
fn byte_and_event_bounds_include_explicit_empty_policy() {
    let mut c = config();
    c.max_artifact_bytes = 0;
    c.allow_empty = true;
    let mut d = DeveloperArtifactSession::new(c, 1).unwrap();
    let mut empty = output();
    empty.byte_len = 0;
    let k = d.propose(empty, 2).unwrap();
    d.accept_into(k, 3, &mut None).unwrap();
    for (value, expected) in [
        (0, ArtifactError::InvalidOutput),
        (17, ArtifactError::ByteBudget),
    ] {
        let mut d = DeveloperArtifactSession::new(config(), 1).unwrap();
        let mut o = output();
        o.byte_len = value;
        assert_eq!(d.propose(o, 2), Err(expected));
        assert_eq!(d.phase(), ArtifactPhase::Stopped);
    }
    let mut c = config();
    c.max_events = 1;
    let mut d = DeveloperArtifactSession::new(c, 1).unwrap();
    let k = d.propose(output(), 2).unwrap();
    d.accept_into(k, 3, &mut None).unwrap();
    assert_eq!(d.propose(output(), 4), Err(ArtifactError::EventBudget));
}
#[test]
fn unset_digests_and_missing_or_oversized_evidence_refuse() {
    for change in [
        |o: &mut ArtifactOutput| o.content_digest = [0; 32],
        |o: &mut ArtifactOutput| o.provenance_digest = [0; 32],
        |o: &mut ArtifactOutput| o.observation_digest = [0; 32],
        |o: &mut ArtifactOutput| o.parent.as_mut().unwrap().record_digest = [0; 32],
    ] {
        let mut d = DeveloperArtifactSession::new(config(), 1).unwrap();
        let mut o = output();
        change(&mut o);
        assert_eq!(d.propose(o, 2), Err(ArtifactError::InvalidOutput));
    }
    let f = Cell::new(0);
    let request = ArtifactAdmission {
        config: config(),
        purpose: GovernedPurpose::IntegrationTest,
        now_tick: 1,
    };
    assert!(matches!(
        GovernedArtifactSession::admit(Fixture { fault: &f }, request, b""),
        Err(ArtifactError::InvalidEvidence)
    ));
    assert!(matches!(
        GovernedArtifactSession::admit(Fixture { fault: &f }, request, &[1; 4097]),
        Err(ArtifactError::InvalidEvidence)
    ));
    let mut s = session(&f);
    let k = s.propose(output(), 2).unwrap();
    assert_eq!(
        s.decide(
            ArtifactDecision {
                candidate: k,
                decision: Decision::Allow,
                expires_at_tick: 50
            },
            b"",
            3
        ),
        Err(ArtifactError::InvalidEvidence)
    );
    assert_eq!(s.phase(), ArtifactPhase::Stopped);
}

#[test]
fn metadata_acceptance_and_refusal_are_zero_alloc_with_positive_control() {
    assert!(crate::allocation::positive_control() > 0);
    let f = Cell::new(0);
    let mut s = session(&f);
    let (record, allocations) = crate::allocation::measure(|| {
        let key = s.propose(output(), 2).unwrap();
        allow(&mut s, key);
        let mut record = None;
        s.accept_into(key, 4, &mut record).unwrap();
        assert!(s.accept_into(key, 5, &mut None).is_err());
        record
    });
    assert_eq!(allocations, 0);
    assert!(record.is_some());
    let mut d = DeveloperArtifactSession::new(config(), 1).unwrap();
    let (_, allocations) = crate::allocation::measure(|| {
        let mut value = output();
        value.byte_len = 17;
        assert_eq!(d.propose(value, 2), Err(ArtifactError::ByteBudget));
    });
    assert_eq!(allocations, 0);
}
