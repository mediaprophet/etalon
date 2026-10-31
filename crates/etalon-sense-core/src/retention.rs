use crate::{time::ClockStamp, ContractError};

/// Monotonic clock epoch used to interpret policy expiries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockDomain {
    clock_id: u64,
    restart_epoch: u32,
}

impl ClockDomain {
    pub const fn new(clock_id: u64, restart_epoch: u32) -> Result<Self, ContractError> {
        if clock_id == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self {
            clock_id,
            restart_epoch,
        })
    }

    pub const fn clock_id(self) -> u64 {
        self.clock_id
    }

    pub const fn restart_epoch(self) -> u32 {
        self.restart_epoch
    }

    pub const fn matches(self, stamp: ClockStamp) -> bool {
        self.clock_id == stamp.clock_id() && self.restart_epoch == stamp.restart_epoch()
    }
}

/// Bounded policy for making retention recommendations. It does not enforce
/// storage quotas or delete resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionPolicy {
    clock: ClockDomain,
    max_resources: usize,
    max_bytes: u64,
}

impl RetentionPolicy {
    pub const fn new(clock: ClockDomain, max_resources: usize, max_bytes: u64) -> Self {
        Self {
            clock,
            max_resources,
            max_bytes,
        }
    }

    pub const fn clock(self) -> ClockDomain {
        self.clock
    }

    pub const fn max_resources(self) -> usize {
        self.max_resources
    }

    pub const fn max_bytes(self) -> u64 {
        self.max_bytes
    }
}

/// Opaque stable resource identifier. Its bytes are used only for equality
/// and deterministic tie-breaking; no meaning is inferred from them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId([u8; 16]);

impl ResourceId {
    pub const fn new(bytes: [u8; 16]) -> Result<Self, ContractError> {
        let mut i = 0;
        let mut nonzero = false;
        while i < bytes.len() {
            if bytes[i] != 0 {
                nonzero = true;
            }
            i += 1;
        }
        if !nonzero {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self(bytes))
    }

    pub const fn bytes(self) -> [u8; 16] {
        self.0
    }
}

/// Revocation state supplied by the authority layer. Unknown is not treated as
/// permission to retain a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RevocationState {
    Unknown,
    Clear,
    Revoked,
}

/// Caller-supplied metadata for one resource. `None` expiry is preserved here
/// but rejected by planning; resource metadata is not itself authoritative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionCandidate {
    id: ResourceId,
    size_bytes: u64,
    expires_at: Option<ClockStamp>,
    revocation: RevocationState,
}

impl RetentionCandidate {
    pub const fn new(
        id: ResourceId,
        size_bytes: u64,
        expires_at: Option<ClockStamp>,
        revocation: RevocationState,
    ) -> Self {
        Self {
            id,
            size_bytes,
            expires_at,
            revocation,
        }
    }

    pub const fn id(self) -> ResourceId {
        self.id
    }

    pub const fn size_bytes(self) -> u64 {
        self.size_bytes
    }

    pub const fn expires_at(self) -> Option<ClockStamp> {
        self.expires_at
    }

    pub const fn revocation(self) -> RevocationState {
        self.revocation
    }
}

/// Recommendation emitted by the planner. These values request no operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetentionAction {
    Retain,
    RecommendDeleteExpired,
    RecommendDeleteRevoked,
    RecommendDeleteCapacity,
}

/// Caller-buffered plan entry. A caller must separately authorize and perform
/// any deletion; this record does not attest that an action occurred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionPlanEntry {
    id: ResourceId,
    action: RetentionAction,
    expiry_ticks_ns: u64,
    size_bytes: u64,
}

impl RetentionPlanEntry {
    /// Construct an inert caller-owned output slot. Its contents are not a
    /// retention decision until returned by a successful plan operation.
    pub const fn output_slot(id: ResourceId) -> Self {
        Self { id, action: RetentionAction::Retain, expiry_ticks_ns: 0, size_bytes: 0 }
    }

    pub const fn id(self) -> ResourceId {
        self.id
    }

    pub const fn action(self) -> RetentionAction {
        self.action
    }

    pub const fn size_bytes(self) -> u64 {
        self.size_bytes
    }
}

/// Plan resource retention in caller-owned output. Entries are returned in
/// ascending opaque-ID order. Among live resources, later expiry center ticks
/// win quota selection; equal expiry is broken by ascending ID. Expiry uncertainty that
/// overlaps `now` is indeterminate and fails the entire plan without writing
/// output. No resource is deleted or otherwise changed.
pub fn plan_retention(
    policy: RetentionPolicy,
    now: ClockStamp,
    candidates: &[RetentionCandidate],
    out: &mut [RetentionPlanEntry],
) -> Result<usize, ContractError> {
    if out.len() < candidates.len() {
        return Err(ContractError::OutOfBounds);
    }
    if !policy.clock.matches(now) {
        return Err(ContractError::ClockDomainMismatch);
    }

    let mut total_bytes = 0u64;
    for (index, candidate) in candidates.iter().enumerate() {
        total_bytes = total_bytes
            .checked_add(candidate.size_bytes)
            .ok_or(ContractError::ArithmeticOverflow)?;
        if candidate.revocation == RevocationState::Unknown {
            return Err(ContractError::InvalidSource);
        }
        let expiry = candidate.expires_at.ok_or(ContractError::InvalidTime)?;
        if !policy.clock.matches(expiry) {
            return Err(ContractError::ClockDomainMismatch);
        }
        expiry_state(now, expiry)?;
        for prior in &candidates[..index] {
            if prior.id == candidate.id {
                return Err(ContractError::InvalidIdentifier);
            }
        }
    }
    let _validated_total_bytes = total_bytes;

    for (slot, candidate) in out.iter_mut().zip(candidates.iter()) {
        let expiry = candidate.expires_at.expect("validated above");
        let action = if candidate.revocation == RevocationState::Revoked {
            RetentionAction::RecommendDeleteRevoked
        } else if expiry_state(now, expiry)? {
            RetentionAction::RecommendDeleteExpired
        } else {
            RetentionAction::Retain
        };
        *slot = RetentionPlanEntry {
            id: candidate.id,
            action,
            expiry_ticks_ns: expiry.ticks_ns(),
            size_bytes: candidate.size_bytes,
        };
    }

    out[..candidates.len()].sort_unstable_by(|left, right| {
        right
            .expiry_ticks_ns
            .cmp(&left.expiry_ticks_ns)
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut retained_count = 0usize;
    let mut retained_bytes = 0u64;
    for entry in &mut out[..candidates.len()] {
        if entry.action != RetentionAction::Retain {
            continue;
        }
        let remaining_bytes = policy.max_bytes.saturating_sub(retained_bytes);
        if retained_count < policy.max_resources && entry.size_bytes <= remaining_bytes {
            retained_count += 1;
            retained_bytes += entry.size_bytes;
        } else {
            entry.action = RetentionAction::RecommendDeleteCapacity;
        }
    }

    out[..candidates.len()].sort_unstable_by_key(|entry| entry.id);
    Ok(candidates.len())
}

fn expiry_state(now: ClockStamp, expiry: ClockStamp) -> Result<bool, ContractError> {
    if now.clock_id() != expiry.clock_id() || now.restart_epoch() != expiry.restart_epoch() {
        return Err(ContractError::ClockDomainMismatch);
    }
    let now_interval = now.interval();
    let expiry_interval = expiry.interval();
    if now_interval.upper_ns() < expiry_interval.lower_ns() {
        Ok(false)
    } else if now_interval.lower_ns() >= expiry_interval.upper_ns() {
        Ok(true)
    } else {
        Err(ContractError::InvalidTime)
    }
}

#[cfg(test)]
#[path = "retention_tests.rs"]
mod tests;
