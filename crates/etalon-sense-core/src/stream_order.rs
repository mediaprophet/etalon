//! Explicit, bounded stream sequence tracking without implicit wraparound.
use crate::{time::ClockStamp, ContractError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceEvent {
    First,
    Contiguous,
    /// Number of sequence numbers absent between accepted observations.
    Gap { missing: u64 },
}

/// Tracks one stream generation. A new generation requires a new tracker;
/// this type cannot infer device restarts or authenticate sequence values.
#[derive(Debug)]
pub struct StreamOrder {
    stream_id: u64,
    generation: u32,
    last: Option<(u64, Option<ClockStamp>)>,
    last_known_timestamp: Option<ClockStamp>,
}

impl StreamOrder {
    pub const fn new(stream_id: u64, generation: u32) -> Result<Self, ContractError> {
        if stream_id == 0 || generation == 0 { return Err(ContractError::InvalidIdentifier); }
        Ok(Self { stream_id, generation, last: None, last_known_timestamp: None })
    }

    /// Accept a strictly increasing sequence. Known stamps are checked against
    /// the last known clock reading, including across intervening unknowns.
    /// Unknown stamps skip time comparison but never reset sequence ordering.
    pub fn observe(&mut self, stream_id: u64, generation: u32, sequence: u64, stamp: ClockStamp) -> Result<SequenceEvent, ContractError> {
        self.observe_optional(stream_id, generation, sequence, Some(stamp))
    }

    /// Admit a sequence with an optional timestamp. Sequence ordering is always
    /// enforced; known stamps compare to the last known stamp across unknowns.
    pub fn observe_optional(&mut self, stream_id: u64, generation: u32, sequence: u64, stamp: Option<ClockStamp>) -> Result<SequenceEvent, ContractError> {
        if stream_id != self.stream_id || generation != self.generation {
            return Err(ContractError::InvalidSource);
        }
        let event = if let Some((previous_sequence, _)) = self.last {
            let delta = sequence.checked_sub(previous_sequence).ok_or(ContractError::InvalidSource)?;
            if delta == 0 { return Err(ContractError::InvalidSource); }
            if delta == 1 { SequenceEvent::Contiguous } else { SequenceEvent::Gap { missing: delta - 1 } }
        } else { SequenceEvent::First };
        if let (Some(stamp), Some(previous_stamp)) = (stamp, self.last_known_timestamp) {
            if stamp.clock_id() != previous_stamp.clock_id()
                || stamp.restart_epoch() != previous_stamp.restart_epoch()
            {
                return Err(ContractError::ClockDomainMismatch);
            }
            if stamp.ticks_ns() < previous_stamp.ticks_ns() {
                return Err(ContractError::InvalidTime);
            }
        }
        if let Some(stamp) = stamp {
            self.last_known_timestamp = Some(stamp);
        }
        self.last = Some((sequence, stamp));
        Ok(event)
    }

    pub const fn stream_id(&self) -> u64 { self.stream_id }
    pub const fn generation(&self) -> u32 { self.generation }
    /// Latest sequence only when that observation has a timestamp. Use
    /// `last_observation` when sequence state must include unknown timestamps.
    pub const fn last(&self) -> Option<(u64, ClockStamp)> {
        match self.last {
            Some((sequence, Some(stamp))) => Some((sequence, stamp)),
            _ => None,
        }
    }

    /// Latest sequence with its explicit optional timestamp.
    pub const fn last_observation(&self) -> Option<(u64, Option<ClockStamp>)> { self.last }

    pub const fn last_known_timestamp(&self) -> Option<ClockStamp> { self.last_known_timestamp }
}

#[cfg(test)]
#[path = "stream_order_tests.rs"]
mod tests;
