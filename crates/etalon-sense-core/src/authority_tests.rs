use super::*;
use crate::{
    capability::{Modality, SensorCapability},
    extended_capability::ExtendedCapability,
    quantity::{PhysicalQuantity, SampleEncoding, Unit},
};
use core::cell::Cell;

fn stamp(clock: u64, epoch: u32, ticks: u64, uncertainty: u64) -> ClockStamp {
    ClockStamp::new(clock, epoch, ticks, uncertainty).unwrap()
}

fn builtin_source() -> DiscoveredSource {
    let capability = SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexI16,
        [8, 0, 0, 0],
        None,
    )
    .unwrap();
    DiscoveredSource::new(10, capability).unwrap()
}

fn extended_source() -> ExtendedDiscoveredSource {
    let capability = ExtendedCapability::new(
        501,
        502,
        None,
        SampleEncoding::ComplexF32,
        [4, 0, 0, 0],
        None,
    )
    .unwrap();
    ExtendedDiscoveredSource::new(11, capability).unwrap()
}

struct FixedClock(Result<ClockStamp, AdapterError>);

impl TrustedCaptureClock for FixedClock {
    fn now(&mut self) -> Result<ClockStamp, AdapterError> {
        self.0
    }
}

struct FixedPolicy {
    decision: CapturePolicyDecision,
    seen: Cell<Option<CaptureAuthorizationRequest>>,
}

impl HostCapturePolicy for FixedPolicy {
    fn decide(
        &self,
        request: &CaptureAuthorizationRequest,
        _now: ClockStamp,
    ) -> CapturePolicyDecision {
        self.seen.set(Some(*request));
        self.decision
    }
}

fn policy(decision: CapturePolicyDecision) -> FixedPolicy {
    FixedPolicy {
        decision,
        seen: Cell::new(None),
    }
}

#[test]
fn public_environment_request_can_issue_existing_builtin_request_type() {
    let source = builtin_source();
    let request =
        CaptureAuthorizationRequest::new(CaptureSource::BuiltIn(source), None, None, 800, 900)
            .unwrap();
    let now = stamp(1, 4, 100, 2);
    let expires = stamp(1, 4, 500, 2);
    let policy = policy(CapturePolicyDecision::Permit {
        expires_at: expires,
    });
    let mut clock = FixedClock(Ok(now));
    let issued = CaptureAuthority::issue(request, &policy, &mut clock).unwrap();

    assert_eq!(policy.seen.get(), Some(request));
    assert_eq!(issued.request().context_id(), 800);
    assert_eq!(issued.request().subject_id(), None);
    assert_eq!(issued.request().actor_id(), None);
    assert_eq!(issued.authorized_at(), now);
    assert_eq!(issued.expires_at(), expires);
    match issued.into_capture_request() {
        AuthorizedCaptureRequest::BuiltIn(capture) => {
            assert_eq!(capture.source(), source);
            assert_eq!(capture.purpose_id(), 900);
        }
        AuthorizedCaptureRequest::Extended(_) => panic!("wrong source variant"),
    }
}

#[test]
fn exact_extended_capability_and_unknown_unit_reach_host_policy() {
    let source = extended_source();
    let request = CaptureAuthorizationRequest::new(
        CaptureSource::Extended(source),
        Some(101),
        Some(102),
        103,
        104,
    )
    .unwrap();
    let expires = stamp(7, 2, 1000, 0);
    let policy = policy(CapturePolicyDecision::Permit {
        expires_at: expires,
    });
    let mut clock = FixedClock(Ok(stamp(7, 2, 100, 0)));
    let issued = CaptureAuthority::issue(request, &policy, &mut clock).unwrap();

    let seen = policy.seen.get().unwrap();
    match seen.source() {
        CaptureSource::Extended(declared) => {
            assert_eq!(declared, source);
            assert_eq!(declared.capability().modality_id(), 501);
            assert_eq!(declared.capability().quantity_id(), 502);
            assert_eq!(declared.capability().unit_id(), None);
        }
        CaptureSource::BuiltIn(_) => panic!("wrong source variant"),
    }
    match issued.into_capture_request() {
        AuthorizedCaptureRequest::Extended(capture) => {
            assert_eq!(capture.source(), source);
            assert_eq!(capture.purpose_id(), 104);
        }
        AuthorizedCaptureRequest::BuiltIn(_) => panic!("wrong source variant"),
    }
}

#[test]
fn deny_unknown_and_clock_failure_never_issue_authorization() {
    let request = CaptureAuthorizationRequest::new(
        CaptureSource::BuiltIn(builtin_source()),
        None,
        None,
        1,
        2,
    )
    .unwrap();
    let expires = stamp(1, 0, 1000, 0);
    let mut clock = FixedClock(Ok(stamp(1, 0, 10, 0)));
    assert_eq!(
        CaptureAuthority::issue(request, &policy(CapturePolicyDecision::Deny), &mut clock)
            .unwrap_err(),
        AuthorityError::Denied
    );
    assert_eq!(
        CaptureAuthority::issue(request, &policy(CapturePolicyDecision::Unknown), &mut clock)
            .unwrap_err(),
        AuthorityError::Unknown
    );
    assert_eq!(
        CaptureAuthority::issue(
            request,
            &policy(CapturePolicyDecision::Permit {
                expires_at: expires
            }),
            &mut FixedClock(Err(AdapterError::Unavailable))
        )
        .unwrap_err(),
        AuthorityError::ClockUnavailable
    );
}

#[test]
fn expiry_domain_and_uncertainty_fail_closed() {
    let request = CaptureAuthorizationRequest::new(
        CaptureSource::BuiltIn(builtin_source()),
        None,
        None,
        1,
        2,
    )
    .unwrap();
    let policy_for = |expires_at| policy(CapturePolicyDecision::Permit { expires_at });
    let now = stamp(2, 1, 100, 5);

    assert_eq!(
        CaptureAuthority::issue(
            request,
            &policy_for(stamp(3, 1, 500, 0)),
            &mut FixedClock(Ok(now))
        )
        .unwrap_err(),
        AuthorityError::ClockDomainMismatch
    );
    assert_eq!(
        CaptureAuthority::issue(
            request,
            &policy_for(stamp(2, 2, 500, 0)),
            &mut FixedClock(Ok(now))
        )
        .unwrap_err(),
        AuthorityError::ClockDomainMismatch
    );
    assert_eq!(
        CaptureAuthority::issue(
            request,
            &policy_for(stamp(2, 1, 105, 0)),
            &mut FixedClock(Ok(now))
        )
        .unwrap_err(),
        AuthorityError::ExpiredOrUncertain
    );
    assert_eq!(
        CaptureAuthority::issue(
            request,
            &policy_for(stamp(2, 1, 90, 0)),
            &mut FixedClock(Ok(stamp(2, 1, 100, 0)))
        )
        .unwrap_err(),
        AuthorityError::ExpiredOrUncertain
    );
}

#[test]
fn malformed_context_or_principal_labels_are_rejected() {
    assert_eq!(
        CaptureAuthorizationRequest::new(
            CaptureSource::BuiltIn(builtin_source()),
            None,
            None,
            0,
            2
        ),
        Err(AuthorityError::InvalidRequest)
    );
    assert_eq!(
        CaptureAuthorizationRequest::new(
            CaptureSource::BuiltIn(builtin_source()),
            Some(0),
            None,
            1,
            2
        ),
        Err(AuthorityError::InvalidRequest)
    );
    assert_eq!(
        CaptureAuthorizationRequest::new(
            CaptureSource::BuiltIn(builtin_source()),
            None,
            Some(0),
            1,
            2
        ),
        Err(AuthorityError::InvalidRequest)
    );
}
