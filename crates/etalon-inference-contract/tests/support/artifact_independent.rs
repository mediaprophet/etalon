use crate::artifact::*;
use crate::{
    ControllerBinding, Decision, ExecutionMode, GovernedPurpose, Instrumentation, SessionIdentity,
    VerificationError,
};
use std::cell::Cell;
fn config() -> ArtifactConfig {
    ArtifactConfig {
        identity: SessionIdentity {
            session_id: 31,
            session_generation: 0,
            request_id: 32,
            request_generation: 0,
            model_id: [33; 32],
            model_generation: 0,
            device_id: 34,
            device_generation: 0,
            policy_id: [35; 32],
            policy_generation: 0,
        },
        instrumentation: Instrumentation {
            tracing: true,
            profiling: true,
        },
        max_artifact_bytes: 9,
        allow_empty: false,
        max_events: 2,
    }
}
fn output() -> ArtifactOutput {
    ArtifactOutput {
        media: ArtifactMedia::Utf8Html,
        source: ArtifactSource::GroundedReplacement,
        byte_len: 9,
        content_digest: [36; 32],
        provenance_digest: [37; 32],
        observation_digest: [38; 32],
        parent: Some(ParentEventBinding {
            domain: ParentEventDomain::Token,
            identity: config().identity,
            sequence: 1,
            record_digest: [39; 32],
        }),
    }
}
// Controlled metadata fixture only. This never authenticates a controller or bytes.
struct MetadataVerifier<'a>(&'a Cell<u8>);
impl ArtifactVerifier for MetadataVerifier<'_> {
    fn admit(
        &mut self,
        r: &ArtifactAdmission,
        e: &[u8],
    ) -> Result<ControllerBinding, VerificationError> {
        if e != b"admission" {
            return Err(VerificationError::Unauthenticated);
        }
        Ok(ControllerBinding {
            identity: [40; 32],
            generation: 0,
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
        match self.0.get() {
            3 => panic!("decision unwind"),
            4 => Err(VerificationError::Unauthenticated),
            _ if e == b"decision" => Ok(()),
            _ => Err(VerificationError::Unauthenticated),
        }
    }
    fn live(&mut self, _: &ControllerBinding, _: u64) -> Result<(), VerificationError> {
        match self.0.get() {
            1 => Err(VerificationError::Disconnected),
            2 => panic!("liveness unwind"),
            _ => Ok(()),
        }
    }
}
fn session(f: &Cell<u8>) -> GovernedArtifactSession<MetadataVerifier<'_>> {
    GovernedArtifactSession::admit(
        MetadataVerifier(f),
        ArtifactAdmission {
            config: config(),
            purpose: GovernedPurpose::Application,
            now_tick: 1,
        },
        b"admission",
    )
    .unwrap()
}
fn allow(s: &mut GovernedArtifactSession<MetadataVerifier<'_>>, key: ArtifactKey, until: u64) {
    s.decide(
        ArtifactDecision {
            candidate: key,
            decision: Decision::Allow,
            expires_at_tick: until,
        },
        b"decision",
        2,
    )
    .unwrap();
}
#[test]
fn independent_artifact_allow_is_once_only_and_does_not_count_tokens() {
    let f = Cell::new(0);
    let mut s = session(&f);
    let mut slot = None;
    for n in 1..=2 {
        let key = s.propose(output(), 2).unwrap();
        assert_eq!(key.event_sequence, n);
        assert_ne!(key.mode, ExecutionMode::StandaloneDeveloper);
        allow(&mut s, key, 10);
        s.accept_into(key, 2, &mut slot).unwrap();
        assert_eq!(slot.take().unwrap().into_key(), key);
    }
    assert_eq!(s.propose(output(), 2), Err(ArtifactError::EventBudget));
    assert_eq!(s.phase(), ArtifactPhase::Stopped);
    let mut s = session(&f);
    let key = s.propose(output(), 2).unwrap();
    allow(&mut s, key, 10);
    s.accept_into(key, 2, &mut slot).unwrap();
    let metadata = slot.take().unwrap().into_key();
    assert_eq!(
        s.accept_into(metadata, 2, &mut slot),
        Err(ArtifactError::WrongPhase)
    );
    assert!(slot.is_none());
}
#[test]
fn independent_all_candidate_metadata_and_parent_fields_are_compared() {
    let f = Cell::new(0);
    for field in 0..23 {
        let mut s = session(&f);
        let key = s.propose(output(), 2).unwrap();
        let mut wrong = key;
        match field {
            0 => wrong.identity.session_id += 1,
            1 => wrong.identity.session_generation += 1,
            2 => wrong.identity.request_id += 1,
            3 => wrong.identity.request_generation += 1,
            4 => wrong.identity.model_id[0] ^= 1,
            5 => wrong.identity.model_generation += 1,
            6 => wrong.identity.device_id += 1,
            7 => wrong.identity.device_generation += 1,
            8 => wrong.identity.policy_id[0] ^= 1,
            9 => wrong.identity.policy_generation += 1,
            10 => wrong.event_sequence += 1,
            11 => wrong.mode = ExecutionMode::StandaloneDeveloper,
            12 => wrong.output.media = ArtifactMedia::Utf8Text,
            13 => wrong.output.source = ArtifactSource::RenderedCompletion,
            14 => wrong.output.byte_len -= 1,
            15 => wrong.output.content_digest[0] ^= 1,
            16 => wrong.output.provenance_digest[0] ^= 1,
            17 => wrong.output.observation_digest[0] ^= 1,
            18 => wrong.output.parent.as_mut().unwrap().domain = ParentEventDomain::Artifact,
            19 => wrong.output.parent.as_mut().unwrap().sequence += 1,
            20 => {
                wrong
                    .output
                    .parent
                    .as_mut()
                    .unwrap()
                    .identity
                    .request_generation += 1
            }
            21 => wrong.output.parent.as_mut().unwrap().record_digest[0] ^= 1,
            22 => wrong.output.parent = None,
            _ => unreachable!(),
        };
        assert_eq!(
            s.decide(
                ArtifactDecision {
                    candidate: wrong,
                    decision: Decision::Allow,
                    expires_at_tick: 10
                },
                b"decision",
                2
            ),
            Err(ArtifactError::WrongCandidate),
            "field {field}"
        );
        assert_eq!(s.phase(), ArtifactPhase::Stopped);
        let mut out = None;
        assert!(s.accept_into(key, 2, &mut out).is_err());
        assert!(out.is_none());
    }
}
#[test]
fn independent_deadline_clock_loss_and_callback_unwind_fail_closed() {
    let f = Cell::new(0);
    let mut s = session(&f);
    let key = s.propose(output(), 2).unwrap();
    allow(&mut s, key, 5);
    let mut out = None;
    assert_eq!(s.accept_into(key, 5, &mut out), Err(ArtifactError::Expired));
    assert!(out.is_none());
    let mut s = session(&f);
    let key = s.propose(output(), 2).unwrap();
    allow(&mut s, key, 200);
    assert_eq!(
        s.accept_into(key, 100, &mut out),
        Err(ArtifactError::Expired)
    );
    let mut s = session(&f);
    let key = s.propose(output(), 2).unwrap();
    allow(&mut s, key, 10);
    assert_eq!(
        s.accept_into(key, 1, &mut out),
        Err(ArtifactError::ClockRegression)
    );
    for fault in 1..=4 {
        f.set(0);
        let mut s = session(&f);
        let key = s.propose(output(), 2).unwrap();
        if fault <= 2 {
            allow(&mut s, key, 10);
        }
        f.set(fault);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if fault <= 2 {
                s.accept_into(key, 2, &mut out)
            } else {
                s.decide(
                    ArtifactDecision {
                        candidate: key,
                        decision: Decision::Allow,
                        expires_at_tick: 10,
                    },
                    b"decision",
                    2,
                )
            }
        }));
        if fault == 2 || fault == 3 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert_eq!(s.phase(), ArtifactPhase::Stopped);
        f.set(0);
        assert!(s.accept_into(key, 2, &mut out).is_err());
        assert!(out.is_none());
    }
}
#[test]
fn independent_deny_steer_and_occupied_record_cannot_publish() {
    let f = Cell::new(0);
    for decision in [
        Decision::Deny,
        Decision::Steer {
            steering_digest: [1; 32],
        },
    ] {
        let mut s = session(&f);
        let key = s.propose(output(), 2).unwrap();
        assert!(s
            .decide(
                ArtifactDecision {
                    candidate: key,
                    decision,
                    expires_at_tick: 10
                },
                b"decision",
                2
            )
            .is_err());
        let mut out = None;
        assert!(s.accept_into(key, 2, &mut out).is_err());
        assert!(out.is_none());
    }
    let mut dev = DeveloperArtifactSession::new(config(), 1).unwrap();
    let old = dev.propose(output(), 1).unwrap();
    let mut slot = None;
    dev.accept_into(old, 1, &mut slot).unwrap();
    let mut s = session(&f);
    let key = s.propose(output(), 2).unwrap();
    allow(&mut s, key, 10);
    assert_eq!(
        s.accept_into(key, 2, &mut slot),
        Err(ArtifactError::OccupiedOutput)
    );
    assert_eq!(*slot.as_ref().unwrap().key(), old);
    assert_eq!(s.phase(), ArtifactPhase::Stopped);
}
#[test]
fn independent_zero_generation_compatibility_and_empty_byte_rules() {
    let mut c = config();
    c.allow_empty = true;
    c.max_artifact_bytes = 0;
    let mut d = DeveloperArtifactSession::new(c, 0).unwrap();
    let mut empty = output();
    empty.byte_len = 0;
    let key = d.propose(empty, 0).unwrap();
    assert_eq!(key.identity.session_generation, 0);
    let mut out = None;
    d.accept_into(key, 0, &mut out).unwrap();
    for field in 0..6 {
        let mut d = DeveloperArtifactSession::new(config(), 0).unwrap();
        let mut x = output();
        match field {
            0 => x.byte_len = 0,
            1 => x.byte_len = 10,
            2 => x.content_digest = [0; 32],
            3 => x.provenance_digest = [0; 32],
            4 => x.observation_digest = [0; 32],
            5 => x.parent.as_mut().unwrap().sequence = 0,
            _ => unreachable!(),
        };
        assert!(d.propose(x, 0).is_err());
        assert_eq!(d.phase(), ArtifactPhase::Stopped);
    }
}
#[test]
fn independent_metadata_hot_success_and_failure_have_zero_allocation_calls() {
    assert!(crate::allocation::positive_control() > 0);
    let f = Cell::new(0);
    let (_, count) = crate::allocation::measure(|| {
        let mut s = session(&f);
        let key = s.propose(output(), 2).unwrap();
        allow(&mut s, key, 10);
        let mut out = None;
        s.accept_into(key, 2, &mut out).unwrap();
        assert_eq!(*out.take().unwrap().key(), key);
        let key = s.propose(output(), 2).unwrap();
        allow(&mut s, key, 10);
        f.set(1);
        assert!(s.accept_into(key, 2, &mut out).is_err());
        assert!(out.is_none());
    });
    assert_eq!(count, 0);
}
