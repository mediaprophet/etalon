use super::*;
use crate::{Instrumentation, SessionIdentity};

#[test]
fn sequence_overflow_refuses_without_exposing_internal_state() {
    let config = ArtifactConfig {
        identity: SessionIdentity {
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
        },
        instrumentation: Instrumentation::default(),
        max_artifact_bytes: 16,
        allow_empty: false,
        max_events: u64::MAX,
    };
    let output = ArtifactOutput {
        media: ArtifactMedia::Utf8Text,
        source: ArtifactSource::RenderedCompletion,
        byte_len: 7,
        content_digest: [11; 32],
        provenance_digest: [12; 32],
        observation_digest: [13; 32],
        parent: None,
    };
    let mut session = DeveloperArtifactSession::new(config, 1).unwrap();
    session.core.sequence = u64::MAX;
    assert_eq!(
        session.propose(output, 2),
        Err(ArtifactError::CounterExhausted)
    );
    assert_eq!(session.phase(), ArtifactPhase::Stopped);
}
