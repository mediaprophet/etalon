use super::*;
use crate::buffer::BufferToken;
use crate::capability::Modality;
use crate::quality::{SourceKind, TrustState};
use crate::quantity::{PhysicalQuantity, SampleEncoding, Unit};
use crate::time::ClockStamp;

fn capability() -> SensorCapability {
    SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexF32,
        [8, 2, 0, 0],
        None,
    )
    .unwrap()
}

fn source() -> SourceDescriptor {
    SourceDescriptor::new(10, SourceKind::Live).unwrap()
}

fn session(frame_budget: u64) -> CaptureSession {
    CaptureSession::new(1, source(), capability(), 2, 3, 4, frame_budget).unwrap()
}

fn stamp(ticks_ns: u64) -> ClockStamp {
    ClockStamp::new(7, 2, ticks_ns, 0).unwrap()
}

fn uncertain_stamp(ticks_ns: u64, uncertainty_ns: u64) -> ClockStamp {
    ClockStamp::new(7, 2, ticks_ns, uncertainty_ns).unwrap()
}

fn stamp_in(clock_id: u64, epoch: u32, ticks_ns: u64) -> ClockStamp {
    ClockStamp::new(clock_id, epoch, ticks_ns, 0).unwrap()
}

fn grant(modality: Modality, start: u64, expiry: u64) -> ExternalAuthorization {
    ExternalAuthorization::new(11, 12, 2, 3, 4, modality, stamp(start), stamp(expiry)).unwrap()
}

fn frame() -> (SampleRegion, SampleLayout) {
    let layout =
        SampleLayout::new(2, [4, 2, 0, 0], [16, 64, 0, 0], SampleEncoding::ComplexF32).unwrap();
    let region = SampleRegion::new(BufferToken::new(99, 1).unwrap(), 0, 128, 128, 4).unwrap();
    (region, layout)
}

#[test]
fn capture_requires_explicit_matching_external_grant() {
    let mut s = session(4);
    assert!(!s.capture_permitted(stamp(10)).unwrap());
    assert_eq!(
        s.start(stamp(10), grant(Modality::Bluetooth, 0, 100)),
        Err(SessionError::GrantScopeMismatch)
    );
    assert_eq!(s.phase(), SessionPhase::Ready);
    s.start(stamp(10), grant(Modality::Wifi, 0, 100)).unwrap();
    assert!(s.capture_permitted(stamp(11)).unwrap());
    assert_eq!(s.active_grant_id(), Some(11));
}

#[test]
fn authorization_grant_must_match_subject_environment_and_purpose() {
    let mut s = session(4);
    let wrong_scope =
        ExternalAuthorization::new(11, 12, 2, 3, 99, Modality::Wifi, stamp(0), stamp(100)).unwrap();
    assert_eq!(
        s.start(stamp(10), wrong_scope),
        Err(SessionError::GrantScopeMismatch)
    );
    assert_eq!(s.phase(), SessionPhase::Ready);
}

#[test]
fn invalid_authorization_assertions_are_rejected() {
    assert_eq!(
        ExternalAuthorization::new(0, 12, 2, 3, 4, Modality::Wifi, stamp(0), stamp(100)),
        Err(SessionError::Contract(ContractError::InvalidIdentifier))
    );
    assert_eq!(
        ExternalAuthorization::new(11, 12, 2, 3, 4, Modality::Wifi, stamp(100), stamp(100)),
        Err(SessionError::Contract(ContractError::InvalidTime))
    );
}

#[test]
fn authorization_window_requires_one_explicit_clock_epoch() {
    assert_eq!(
        ExternalAuthorization::new(
            11,
            12,
            2,
            3,
            4,
            Modality::Wifi,
            stamp_in(7, 2, 10),
            stamp_in(7, 3, 100),
        ),
        Err(SessionError::Contract(ContractError::ClockDomainMismatch))
    );

    let mut s = session(1);
    assert_eq!(
        s.start(stamp_in(8, 2, 20), grant(Modality::Wifi, 10, 100)),
        Err(SessionError::Contract(ContractError::ClockDomainMismatch))
    );
    assert_eq!(s.phase(), SessionPhase::Ready);
}

#[test]
fn uncertain_grant_boundaries_fail_closed() {
    let uncertain_grant = ExternalAuthorization::new(
        11,
        12,
        2,
        3,
        4,
        Modality::Wifi,
        uncertain_stamp(10, 1),
        uncertain_stamp(30, 1),
    )
    .unwrap();

    let mut before = session(1);
    assert_eq!(
        before.start(stamp(10), uncertain_grant),
        Err(SessionError::TimeUncertain)
    );
    assert_eq!(before.phase(), SessionPhase::Ready);

    let mut at_uncertain_expiry = session(1);
    at_uncertain_expiry
        .start(stamp(12), uncertain_grant)
        .unwrap();
    assert_eq!(
        at_uncertain_expiry.capture_permitted(uncertain_stamp(28, 1)),
        Err(SessionError::TimeUncertain)
    );
    assert_eq!(at_uncertain_expiry.phase(), SessionPhase::Stopped);
}

#[test]
fn clock_domain_or_epoch_change_stops_an_active_session() {
    let mut wrong_clock = session(1);
    wrong_clock
        .start(stamp(10), grant(Modality::Wifi, 0, 100))
        .unwrap();
    assert_eq!(
        wrong_clock.capture_permitted(stamp_in(8, 2, 11)),
        Err(SessionError::Contract(ContractError::ClockDomainMismatch))
    );
    assert_eq!(wrong_clock.phase(), SessionPhase::Stopped);

    let mut wrong_epoch = session(1);
    wrong_epoch
        .start(stamp(10), grant(Modality::Wifi, 0, 100))
        .unwrap();
    assert_eq!(
        wrong_epoch.capture_permitted(stamp_in(7, 3, 11)),
        Err(SessionError::Contract(ContractError::ClockDomainMismatch))
    );
    assert_eq!(wrong_epoch.phase(), SessionPhase::Stopped);
}

#[test]
fn grant_window_is_checked_and_expiry_stops_capture() {
    let mut s = session(4);
    assert_eq!(
        s.start(stamp(9), grant(Modality::Wifi, 10, 100)),
        Err(SessionError::GrantNotYetValid)
    );
    assert_eq!(
        s.start(stamp(100), grant(Modality::Wifi, 10, 100)),
        Err(SessionError::GrantExpired)
    );
    s.start(stamp(10), grant(Modality::Wifi, 10, 20)).unwrap();
    assert!(s.capture_permitted(stamp(19)).unwrap());
    assert!(!s.capture_permitted(stamp(20)).unwrap());
    assert_eq!(s.phase(), SessionPhase::Stopped);
    assert_eq!(s.active_grant_id(), None);
}

#[test]
fn wrong_time_order_stops_session_fail_closed() {
    let mut s = session(4);
    s.start(stamp(10), grant(Modality::Wifi, 0, 100)).unwrap();
    assert_eq!(
        s.capture_permitted(stamp(9)),
        Err(SessionError::Contract(ContractError::InvalidTime))
    );
    assert_eq!(s.phase(), SessionPhase::Stopped);
    assert!(!s.capture_permitted(stamp(11)).unwrap());
}

#[test]
fn frame_is_counted_only_after_capability_and_buffer_checks() {
    let mut s = session(2);
    s.start(stamp(10), grant(Modality::Wifi, 0, 100)).unwrap();
    let (region, layout) = frame();
    s.admit_frame(stamp(11), &region, &layout).unwrap();
    assert_eq!(s.frames_admitted(), 1);

    let wrong_encoding =
        SampleLayout::new(1, [4, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::F32).unwrap();
    assert_eq!(
        s.admit_frame(stamp(12), &region, &wrong_encoding),
        Err(SessionError::Contract(ContractError::IncompatibleEncoding))
    );
    assert_eq!(s.frames_admitted(), 1);

    let over_extent =
        SampleLayout::new(2, [9, 2, 0, 0], [16, 144, 0, 0], SampleEncoding::ComplexF32).unwrap();
    assert_eq!(
        s.admit_frame(stamp(13), &region, &over_extent),
        Err(SessionError::Contract(ContractError::OutOfBounds))
    );
    assert_eq!(s.frames_admitted(), 1);
}

#[test]
fn frame_budget_stops_the_stream() {
    let mut s = session(1);
    s.start(stamp(10), grant(Modality::Wifi, 0, 100)).unwrap();
    let (region, layout) = frame();
    s.admit_frame(stamp(11), &region, &layout).unwrap();
    assert_eq!(
        s.admit_frame(stamp(12), &region, &layout),
        Err(SessionError::FrameBudgetExhausted)
    );
    assert_eq!(s.phase(), SessionPhase::Stopped);
    assert_eq!(s.frames_admitted(), 1);
}

#[test]
fn stop_allows_restart_but_revocation_is_terminal() {
    let mut s = session(2);
    s.start(stamp(10), grant(Modality::Wifi, 0, 100)).unwrap();
    s.stop();
    assert_eq!(s.phase(), SessionPhase::Stopped);
    assert!(!s.capture_permitted(stamp(11)).unwrap());
    s.start(stamp(12), grant(Modality::Wifi, 0, 100)).unwrap();
    s.revoke();
    assert_eq!(s.phase(), SessionPhase::Revoked);
    assert!(!s.capture_permitted(stamp(13)).unwrap());
    assert_eq!(
        s.start(stamp(14), grant(Modality::Wifi, 0, 100)),
        Err(SessionError::Revoked)
    );
    s.stop();
    assert_eq!(s.phase(), SessionPhase::Revoked);
}

#[test]
fn live_source_does_not_imply_verified_hardware() {
    let mut s = session(1);
    assert_eq!(s.source().kind(), SourceKind::Live);
    assert_eq!(s.source_trust_state(), TrustState::Unverified);
    s.start(stamp(10), grant(Modality::Wifi, 0, 100)).unwrap();
    assert_eq!(s.source_trust_state(), TrustState::Unverified);
}
