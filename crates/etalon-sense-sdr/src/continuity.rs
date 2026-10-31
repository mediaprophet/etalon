//! Optional caller-owned monotonic tracker for host-reported SDR stream data.
//!
//! Unknown stream, generation, or sequence fields are skipped and never clear
//! the core tracker. A tracker is bound to one nonzero stream/generation; a new
//! generation requires a new tracker, matching the shared `StreamOrder` API.

use etalon_sense_core::{
    stream_order::{SequenceEvent, StreamOrder},
    time::ClockStamp,
    ContractError,
};

use crate::{SdrFrame, SdrPowerFrame, SdrReadReport};

#[derive(Debug)]
pub struct SdrContinuityTracker {
    order: StreamOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdrContinuityError {
    Contract(ContractError),
}

impl SdrContinuityTracker {
    pub const fn new(stream_id: u64, generation: u32) -> Result<Self, SdrContinuityError> {
        match StreamOrder::new(stream_id, generation) {
            Ok(order) => Ok(Self { order }),
            Err(error) => Err(SdrContinuityError::Contract(error)),
        }
    }

    /// Apply metadata from a reader report before or after acquiring its frame.
    /// Missing stream/generation/sequence returns `Ok(None)` without changing
    /// previously accepted sequence or timestamp state.
    pub fn observe_report(
        &mut self,
        report: SdrReadReport,
    ) -> Result<Option<SequenceEvent>, SdrContinuityError> {
        self.observe_parts(
            report.stream_id(),
            report.generation(),
            report.sequence(),
            report.timestamp(),
        )
    }

    pub fn observe_frame(
        &mut self,
        frame: SdrFrame<'_>,
    ) -> Result<Option<SequenceEvent>, SdrContinuityError> {
        self.observe_parts(frame.stream_id(), frame.generation(), frame.sequence(), frame.timestamp())
    }

    pub fn observe_power_frame(
        &mut self,
        frame: SdrPowerFrame<'_>,
    ) -> Result<Option<SequenceEvent>, SdrContinuityError> {
        self.observe_parts(frame.stream_id(), frame.generation(), frame.sequence(), frame.timestamp())
    }

    fn observe_parts(
        &mut self,
        stream_id: Option<u64>,
        generation: Option<u32>,
        sequence: Option<u64>,
        timestamp: Option<ClockStamp>,
    ) -> Result<Option<SequenceEvent>, SdrContinuityError> {
        let (Some(stream_id), Some(generation), Some(sequence)) = (stream_id, generation, sequence) else {
            return Ok(None);
        };
        self.order
            .observe_optional(stream_id, generation, sequence, timestamp)
            .map(Some)
            .map_err(SdrContinuityError::Contract)
    }

    pub const fn stream_id(&self) -> u64 { self.order.stream_id() }
    pub const fn generation(&self) -> u32 { self.order.generation() }
    pub const fn last_known_timestamp(&self) -> Option<ClockStamp> { self.order.last_known_timestamp() }
    pub const fn last_observation(&self) -> Option<(u64, Option<ClockStamp>)> {
        self.order.last_observation()
    }
}

#[cfg(test)]
#[path = "continuity_tests.rs"]
mod tests;
