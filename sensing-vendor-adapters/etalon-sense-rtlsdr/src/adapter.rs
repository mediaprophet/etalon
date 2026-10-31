//! Optional bridge from native-operation gates to Etalon capture authority.

use std::cell::RefCell;
use etalon_sense_core::{
    authority::{CaptureAuthorizationRequest, CaptureAuthority, CaptureSource,
        HostCapturePolicy, IssuedCaptureAuthorization, TrustedCaptureClock},
    capability::Modality,
    quantity::{PhysicalQuantity, SampleEncoding, Unit},
    time::ClockStamp,
};
use crate::{Config, GateContext, RtlSdrGate, Stage};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureGateError { InvalidSource, InvalidGeneration }

/// Bind the exact native operation, including settings, stage and device
/// identity, to the core policy request. Implementations supply host semantics;
/// this library does not infer an ontology mapping or accept a default scope.
pub trait CaptureOperationScope {
    fn matches(&self, request: &CaptureAuthorizationRequest, native: &GateContext<'_>) -> bool;
}

/// Fresh clock/policy issuance is performed at every native gate callback.
/// Its scope callback must independently authorize tuning/configuration; a
/// generic capture capability does not describe those controls. Use a separate
/// instance/request for operations requiring a different purpose or authority.
/// The deployment host still owns the immutable operation lease: pre/post
/// checks cannot cancel an in-flight synchronous driver call or undo capture.
/// The original pre-operation expiry remains binding through its post callback;
/// repeated prechecks cannot renew it. A failed operation with no post callback
/// leaves this gate fail-closed; create a new gate for a new operation/session.
pub struct RtlSdrCaptureGate<'a, P, C, S> {
    request: CaptureAuthorizationRequest,
    generation: u32,
    device_index: u32,
    pair_capacity: usize,
    policy: &'a P,
    clock: RefCell<&'a mut C>,
    scope: &'a S,
    window: RefCell<Option<OperationWindow>>,
}

struct OperationWindow {
    before: Stage,
    requested: Option<Config>,
    pair_count: Option<usize>,
    authorized_at: ClockStamp,
    expires_at: ClockStamp,
}

impl OperationWindow {
    fn matches(&self, before: Stage, native: &GateContext<'_>) -> bool {
        self.before == before && self.requested == native.requested && self.pair_count == native.pair_count
    }

    fn contains(&self, now: ClockStamp) -> bool {
        valid_time_window(self.authorized_at, self.expires_at, now)
    }

    fn narrow(&mut self, issued: &IssuedCaptureAuthorization, checked_now: ClockStamp) -> bool {
        if !self.contains(checked_now) { return false; }
        let expiry = issued.expires_at();
        if expiry.interval().lower_ns() < self.expires_at.interval().lower_ns() {
            self.expires_at = expiry;
        }
        self.authorized_at = checked_now;
        true
    }
}

impl<'a, P, C, S> RtlSdrCaptureGate<'a, P, C, S>
where P: HostCapturePolicy, C: TrustedCaptureClock, S: CaptureOperationScope {
    pub fn new(request: CaptureAuthorizationRequest, generation: u32, device_index: u32,
        policy: &'a P, clock: &'a mut C, scope: &'a S) -> Result<Self, CaptureGateError> {
        if generation == 0 { return Err(CaptureGateError::InvalidGeneration); }
        let pair_capacity = source_capacity(request.source())?;
        Ok(Self { request, generation, device_index, pair_capacity, policy,
            clock: RefCell::new(clock), scope, window: RefCell::new(None) })
    }

    pub const fn request(&self) -> &CaptureAuthorizationRequest { &self.request }
}

impl<P, C, S> RtlSdrGate for RtlSdrCaptureGate<'_, P, C, S>
where P: HostCapturePolicy, C: TrustedCaptureClock, S: CaptureOperationScope {
    fn allow(&self, native: &GateContext<'_>) -> bool {
        if native.target_id != self.request.source().source_id()
            || native.generation != self.generation || native.device_index != self.device_index
            || (matches!(native.stage, Stage::Read | Stage::ReadReady | Stage::PostRead) && native.pair_count.is_none())
            || native.pair_count.is_some_and(|count| count == 0 || count > self.pair_capacity) {
            return false;
        }
        let Ok(mut current) = self.window.try_borrow_mut() else { return false; };
        let (before, is_post, is_recheck) = match native.stage {
            Stage::Open => (Stage::Open, false, false), Stage::PostOpen => (Stage::Open, true, false),
            Stage::Configure => (Stage::Configure, false, false), Stage::PostConfigure => (Stage::Configure, true, false),
            Stage::Read => (Stage::Read, false, false), Stage::ReadReady => (Stage::Read, false, true),
            Stage::PostRead => (Stage::Read, true, false),
        };
        if current.as_ref().is_some_and(|window| !window.matches(before, native))
            || ((is_post || is_recheck) && current.is_none())
            || (!is_post && !is_recheck && current.is_some()) { return false; }
        if !self.scope.matches(&self.request, native) {
            if is_post { *current = None; }
            return false;
        }
        let Ok(mut clock) = self.clock.try_borrow_mut() else { return false; };
        let issued = CaptureAuthority::issue(self.request, self.policy, &mut **clock);
        let accepted = match issued {
            Ok(issued) => {
                // Policy callbacks may block; issuance's earlier clock reading
                // cannot establish that the grant is still current on return.
                match (**clock).now() {
                    Ok(now) if valid_time_window(issued.authorized_at(), issued.expires_at(), now) => {
                        if let Some(window) = current.as_mut() { window.narrow(&issued, now) }
                        else {
                            *current = Some(OperationWindow { before, requested: native.requested,
                                pair_count: native.pair_count, authorized_at: now,
                                expires_at: issued.expires_at() });
                            true
                        }
                    }
                    _ => false,
                }
            }
            Err(_) => false,
        };
        if is_post { *current = None; }
        accepted
    }
}

fn valid_time_window(authorized_at: ClockStamp, expires_at: ClockStamp, now: ClockStamp) -> bool {
    now.clock_id() == expires_at.clock_id()
        && now.restart_epoch() == expires_at.restart_epoch()
        && now.interval().lower_ns() >= authorized_at.interval().upper_ns()
        && now.interval().upper_ns() < expires_at.interval().lower_ns()
}

fn source_capacity(source: CaptureSource) -> Result<usize, CaptureGateError> {
    let (encoding, rank, extents) = match source {
        CaptureSource::BuiltIn(source) => {
            let capability = source.capability();
            if capability.modality() != Modality::Sdr
                || capability.quantity() != PhysicalQuantity::Dimensionless
                || capability.unit() != Unit::Unitless {
                return Err(CaptureGateError::InvalidSource);
            }
            (capability.encoding(), capability.max_rank(), capability.max_extents())
        }
        CaptureSource::Extended(source) => {
            let capability = source.capability();
            (capability.encoding(), capability.rank(), capability.max_extents())
        }
    };
    if encoding != SampleEncoding::U8 || rank != 2 || extents[1] != 2
        || extents[0] == 0 || extents[0] as usize > super::conversion::MAX_CONVERSION_PAIRS {
        return Err(CaptureGateError::InvalidSource);
    }
    Ok(extents[0] as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use etalon_sense_core::{adapter::{AdapterError, DiscoveredSource},
        authority::CapturePolicyDecision, capability::SensorCapability, time::ClockStamp};

    struct Clock;
    impl TrustedCaptureClock for Clock {
        fn now(&mut self) -> Result<ClockStamp, AdapterError> { Ok(ClockStamp::new(1, 1, 10, 0).unwrap()) }
    }
    struct Policy { calls: Cell<usize>, permit: Cell<bool> }
    impl HostCapturePolicy for Policy {
        fn decide(&self, _: &CaptureAuthorizationRequest, _: ClockStamp) -> CapturePolicyDecision {
            self.calls.set(self.calls.get() + 1);
            if self.permit.get() { CapturePolicyDecision::Permit { expires_at: ClockStamp::new(1, 1, 100, 0).unwrap() } }
            else { CapturePolicyDecision::Deny }
        }
    }
    struct Scope;
    impl CaptureOperationScope for Scope {
        fn matches(&self, _: &CaptureAuthorizationRequest, native: &GateContext<'_>) -> bool {
            matches!(native.stage, Stage::Read | Stage::ReadReady | Stage::PostRead)
        }
    }
    fn request() -> CaptureAuthorizationRequest {
        let capability = SensorCapability::new(Modality::Sdr, PhysicalQuantity::Dimensionless,
            Unit::Unitless, SampleEncoding::U8, [4, 2, 0, 0], None).unwrap();
        CaptureAuthorizationRequest::new(CaptureSource::BuiltIn(DiscoveredSource::new(7, capability).unwrap()),
            None, None, 11, 12).unwrap()
    }
    fn native() -> GateContext<'static> {
        GateContext { stage: Stage::Read, target_id: 7, generation: 2, device_index: 3,
            requested: None, applied: None, identity: None, pair_count: Some(4) }
    }

    #[test]
    fn exact_native_scope_precedes_core_policy_issuance() {
        let policy = Policy { calls: Cell::new(0), permit: Cell::new(true) };
        let mut clock = Clock;
        let gate = RtlSdrCaptureGate::new(request(), 2, 3, &policy, &mut clock, &Scope).unwrap();
        let mut context = native();
        assert!(gate.allow(&context));
        assert_eq!(policy.calls.get(), 1);
        context.generation = 3;
        assert!(!gate.allow(&context));
        context.generation = 2;
        context.pair_count = Some(5);
        assert!(!gate.allow(&context));
        context.pair_count = None;
        assert!(!gate.allow(&context));
        assert_eq!(policy.calls.get(), 1);
    }

    #[test]
    fn policy_is_rechecked_after_native_operation() {
        let policy = Policy { calls: Cell::new(0), permit: Cell::new(true) };
        let mut clock = Clock;
        let gate = RtlSdrCaptureGate::new(request(), 2, 3, &policy, &mut clock, &Scope).unwrap();
        let mut context = native();
        assert!(gate.allow(&context));
        policy.permit.set(false);
        context.stage = Stage::PostRead;
        assert!(!gate.allow(&context));
        assert_eq!(policy.calls.get(), 2);
    }

    #[test]
    fn rolling_policy_expiry_cannot_renew_the_original_read_window() {
        struct ChangingClock<'a>(&'a Cell<u64>);
        impl TrustedCaptureClock for ChangingClock<'_> {
            fn now(&mut self) -> Result<ClockStamp, AdapterError> {
                Ok(ClockStamp::new(1, 1, self.0.get(), 0).unwrap())
            }
        }
        struct RollingPolicy;
        impl HostCapturePolicy for RollingPolicy {
            fn decide(&self, _: &CaptureAuthorizationRequest, now: ClockStamp) -> CapturePolicyDecision {
                CapturePolicyDecision::Permit { expires_at: ClockStamp::new(1, 1, now.ticks_ns() + 5, 0).unwrap() }
            }
        }
        let ticks = Cell::new(10);
        let mut clock = ChangingClock(&ticks);
        let gate = RtlSdrCaptureGate::new(request(), 2, 3, &RollingPolicy, &mut clock, &Scope).unwrap();
        let mut context = native();
        context.stage = Stage::ReadReady;
        assert!(!gate.allow(&context));
        context.stage = Stage::Read;
        assert!(gate.allow(&context)); // Original expiry = 15.
        ticks.set(12);
        context.stage = Stage::ReadReady;
        assert!(gate.allow(&context)); // A new expiry = 17 cannot replace 15.
        ticks.set(16);
        context.stage = Stage::PostRead;
        assert!(!gate.allow(&context)); // Fresh policy permits until 21; original expired.
        assert!(!gate.allow(&context)); // Consumed post window cannot be replayed.
    }

    #[test]
    fn immediate_read_permission_can_shorten_but_never_extend_the_window() {
        struct ChangingClock<'a>(&'a Cell<u64>);
        impl TrustedCaptureClock for ChangingClock<'_> {
            fn now(&mut self) -> Result<ClockStamp, AdapterError> {
                Ok(ClockStamp::new(1, 1, self.0.get(), 0).unwrap())
            }
        }
        struct ShorteningPolicy;
        impl HostCapturePolicy for ShorteningPolicy {
            fn decide(&self, _: &CaptureAuthorizationRequest, now: ClockStamp) -> CapturePolicyDecision {
                let expiry = if now.ticks_ns() == 20 { 30 } else { 100 };
                CapturePolicyDecision::Permit { expires_at: ClockStamp::new(1, 1, expiry, 0).unwrap() }
            }
        }
        let ticks = Cell::new(10);
        let mut clock = ChangingClock(&ticks);
        let gate = RtlSdrCaptureGate::new(request(), 2, 3, &ShorteningPolicy, &mut clock, &Scope).unwrap();
        let mut context = native();
        assert!(gate.allow(&context));
        ticks.set(20);
        context.stage = Stage::ReadReady;
        assert!(gate.allow(&context));
        ticks.set(50);
        context.stage = Stage::PostRead;
        assert!(!gate.allow(&context));
    }

    #[test]
    fn slow_policy_must_not_return_an_already_expired_native_grant() {
        struct ChangingClock<'a>(&'a Cell<u64>);
        impl TrustedCaptureClock for ChangingClock<'_> {
            fn now(&mut self) -> Result<ClockStamp, AdapterError> {
                Ok(ClockStamp::new(1, 1, self.0.get(), 0).unwrap())
            }
        }
        struct SlowPolicy<'a>(&'a Cell<u64>);
        impl HostCapturePolicy for SlowPolicy<'_> {
            fn decide(&self, _: &CaptureAuthorizationRequest, _: ClockStamp) -> CapturePolicyDecision {
                self.0.set(20);
                CapturePolicyDecision::Permit { expires_at: ClockStamp::new(1, 1, 15, 0).unwrap() }
            }
        }
        let ticks = Cell::new(10);
        let mut clock = ChangingClock(&ticks);
        let policy = SlowPolicy(&ticks);
        let gate = RtlSdrCaptureGate::new(request(), 2, 3, &policy, &mut clock, &Scope).unwrap();
        assert!(!gate.allow(&native()));
    }
}
