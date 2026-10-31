use super::{
    plan_retention, ClockDomain, ResourceId, RetentionAction, RetentionCandidate,
    RetentionPlanEntry, RetentionPolicy, RevocationState,
};
use crate::{time::ClockStamp, ContractError};

fn id(byte: u8) -> ResourceId {
    ResourceId::new([byte; 16]).unwrap()
}

fn stamp(ticks_ns: u64, uncertainty_ns: u64) -> ClockStamp {
    ClockStamp::new(5, 3, ticks_ns, uncertainty_ns).unwrap()
}

fn policy(max_resources: usize, max_bytes: u64) -> RetentionPolicy {
    RetentionPolicy::new(ClockDomain::new(5, 3).unwrap(), max_resources, max_bytes)
}

fn candidate(
    resource_id: u8,
    size_bytes: u64,
    expiry: Option<ClockStamp>,
    revocation: RevocationState,
) -> RetentionCandidate {
    RetentionCandidate::new(id(resource_id), size_bytes, expiry, revocation)
}

fn entry(resource_id: u8, action: RetentionAction, expiry: u64, size: u64) -> RetentionPlanEntry {
    RetentionPlanEntry {
        id: id(resource_id),
        action,
        expiry_ticks_ns: expiry,
        size_bytes: size,
    }
}

#[test]
fn planning_returns_deterministic_id_order_and_expiry_decisions() {
    let candidates = [
        candidate(3, 2, Some(stamp(200, 0)), RevocationState::Clear),
        candidate(1, 2, Some(stamp(90, 0)), RevocationState::Clear),
        candidate(2, 2, Some(stamp(100, 0)), RevocationState::Revoked),
    ];
    let mut out = [entry(8, RetentionAction::Retain, 0, 0); 3];
    assert_eq!(
        plan_retention(policy(4, 8), stamp(100, 0), &candidates, &mut out),
        Ok(3)
    );
    assert_eq!(out[0].id(), id(1));
    assert_eq!(out[0].action(), RetentionAction::RecommendDeleteExpired);
    assert_eq!(out[1].id(), id(2));
    assert_eq!(out[1].action(), RetentionAction::RecommendDeleteRevoked);
    assert_eq!(out[2].id(), id(3));
    assert_eq!(out[2].action(), RetentionAction::Retain);
}

#[test]
fn capacity_keeps_latest_expiry_then_opaque_id_tie_break_and_sorts_output() {
    let candidates = [
        candidate(3, 5, Some(stamp(300, 0)), RevocationState::Clear),
        candidate(2, 4, Some(stamp(200, 0)), RevocationState::Clear),
        candidate(1, 4, Some(stamp(200, 0)), RevocationState::Clear),
    ];
    let mut out = [entry(9, RetentionAction::Retain, 0, 0); 3];
    assert_eq!(
        plan_retention(policy(2, 9), stamp(100, 0), &candidates, &mut out),
        Ok(3)
    );
    assert_eq!(out[0].id(), id(1));
    assert_eq!(out[0].action(), RetentionAction::Retain);
    assert_eq!(out[1].id(), id(2));
    assert_eq!(out[1].action(), RetentionAction::RecommendDeleteCapacity);
    assert_eq!(out[2].id(), id(3));
    assert_eq!(out[2].action(), RetentionAction::Retain);
}

#[test]
fn unknown_expiry_and_unknown_revocation_fail_closed_without_writing() {
    let sentinel = entry(8, RetentionAction::Retain, 123, 4);
    let mut out = [sentinel; 1];
    let unknown_expiry = [candidate(1, 4, None, RevocationState::Revoked)];
    assert_eq!(
        plan_retention(policy(1, 4), stamp(100, 0), &unknown_expiry, &mut out),
        Err(ContractError::InvalidTime)
    );
    assert_eq!(out, [sentinel]);

    let unknown_revocation = [candidate(
        1,
        4,
        Some(stamp(200, 0)),
        RevocationState::Unknown,
    )];
    assert_eq!(
        plan_retention(policy(1, 4), stamp(100, 0), &unknown_revocation, &mut out),
        Err(ContractError::InvalidSource)
    );
    assert_eq!(out, [sentinel]);
}

#[test]
fn mismatched_clock_domains_and_ambiguous_expiry_fail_closed() {
    let candidates = [candidate(
        1,
        1,
        Some(ClockStamp::new(6, 3, 200, 0).unwrap()),
        RevocationState::Clear,
    )];
    let mut out = [entry(8, RetentionAction::Retain, 0, 0); 1];
    assert_eq!(
        plan_retention(policy(1, 1), stamp(100, 0), &candidates, &mut out),
        Err(ContractError::ClockDomainMismatch)
    );
    assert_eq!(
        plan_retention(
            policy(1, 1),
            ClockStamp::new(6, 3, 100, 0).unwrap(),
            &[candidate(1, 1, Some(stamp(200, 0)), RevocationState::Clear)],
            &mut out
        ),
        Err(ContractError::ClockDomainMismatch)
    );
    assert_eq!(
        plan_retention(
            policy(1, 1),
            stamp(100, 10),
            &[candidate(
                1,
                1,
                Some(stamp(105, 10)),
                RevocationState::Clear
            )],
            &mut out
        ),
        Err(ContractError::InvalidTime)
    );
}

#[test]
fn exact_expiry_is_expired_and_pre_expiry_uncertainty_stays_live() {
    let candidates = [
        candidate(1, 1, Some(stamp(100, 0)), RevocationState::Clear),
        candidate(2, 1, Some(stamp(130, 10)), RevocationState::Clear),
    ];
    let mut out = [entry(8, RetentionAction::Retain, 0, 0); 2];
    assert_eq!(
        plan_retention(policy(2, 2), stamp(100, 0), &candidates, &mut out),
        Ok(2)
    );
    assert_eq!(out[0].action(), RetentionAction::RecommendDeleteExpired);
    assert_eq!(out[1].action(), RetentionAction::Retain);
}

#[test]
fn invalid_ids_duplicates_capacity_and_arithmetic_overflow_fail_closed() {
    assert_eq!(
        ResourceId::new([0; 16]),
        Err(ContractError::InvalidIdentifier)
    );
    assert_eq!(
        ClockDomain::new(0, 1),
        Err(ContractError::InvalidIdentifier)
    );

    let duplicate = [
        candidate(1, 1, Some(stamp(200, 0)), RevocationState::Clear),
        candidate(1, 1, Some(stamp(300, 0)), RevocationState::Clear),
    ];
    let sentinel = entry(8, RetentionAction::Retain, 123, 4);
    let mut out = [sentinel; 2];
    assert_eq!(
        plan_retention(policy(2, 2), stamp(100, 0), &duplicate, &mut out),
        Err(ContractError::InvalidIdentifier)
    );
    assert_eq!(out, [sentinel; 2]);

    let overflow = [
        candidate(1, u64::MAX, Some(stamp(200, 0)), RevocationState::Clear),
        candidate(2, 1, Some(stamp(300, 0)), RevocationState::Clear),
    ];
    assert_eq!(
        plan_retention(policy(2, u64::MAX), stamp(100, 0), &overflow, &mut out),
        Err(ContractError::ArithmeticOverflow)
    );
    assert_eq!(out, [sentinel; 2]);

    let one = [candidate(1, 1, Some(stamp(200, 0)), RevocationState::Clear)];
    let mut short = [sentinel; 0];
    assert_eq!(
        plan_retention(policy(0, 0), stamp(100, 0), &one, &mut short),
        Err(ContractError::OutOfBounds)
    );
}
