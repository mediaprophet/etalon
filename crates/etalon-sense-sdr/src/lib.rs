//! Bounded bridge from a host-owned SDR reader into Etalon sample buffers.
//!
//! This crate contains no concrete SDR driver, device discovery, OS access,
//! radio control, authorization authority, or hardware attestation. A host
//! implements [`SdrReader`] around its selected driver and owns that device.
//! Reader IDs, requested/applied RF settings, clock stamps, and overrun reports
//! are declarations from that host/reader and remain unverified. They do not
//! establish RF tuning, clock accuracy, capture provenance, or physical truth.
//! `SdrReader::read_i16_iq` is synchronous: implementations must return within
//! their host-selected read budget, must not unwind, and must ensure acquisition
//! for the requested block is stopped or otherwise bounded when they return.
//! Cancellation of a blocked driver read is the host's responsibility; this
//! bridge cannot interrupt it or undo radio-side acquisition.
//!
//! Each request uses a caller-provided buffer sized for exactly one bounded
//! block of interleaved signed-i16 IQ samples. Successful output is a borrowed
//! view over that buffer; the crate allocates no sample storage.
#![no_std]
#![forbid(unsafe_code)]

use core::sync::atomic::{AtomicBool, Ordering};
use etalon_sense_core::{buffer::SampleLayout, quantity::SampleEncoding, time::ClockStamp};
pub use etalon_signal::execution::{AllocationAdmission, ExecutionAllocationClass};
use etalon_signal::{
    decode::ByteOrder,
    execution::{
        execute_i16_iq_power_into_with_admission, ExecutionError, ExecutionTarget,
        RegisteredSignalProvider,
    },
};

pub const BYTES_PER_IQ_SAMPLE: usize = 4;
pub const MAX_IQ_SAMPLES_PER_READ: usize = 1_048_576;

pub mod continuity;
pub mod observation;
pub use continuity::{SdrContinuityError, SdrContinuityTracker};
pub use observation::{
    adapt_sdr_frame, adapt_sdr_power_frame, AdaptError as ObservationAdaptError,
    SdrCaptureMetadata, SdrObservation, SdrObservationMapping, SdrPayload,
};

/// A positive exact sample rate represented as samples per second.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleRateRatio {
    samples: u64,
    seconds: u64,
}

impl SampleRateRatio {
    pub const fn new(samples: u64, seconds: u64) -> Result<Self, SdrError> {
        if samples == 0 || seconds == 0 {
            return Err(SdrError::InvalidTuning);
        }
        Ok(Self { samples, seconds })
    }

    pub const fn samples(self) -> u64 {
        self.samples
    }
    pub const fn seconds(self) -> u64 {
        self.seconds
    }
}

/// Caller-requested center frequency and complex sample rate.
///
/// Center frequency is expressed in integer hertz. Bandwidth is an optional
/// host declaration in hertz; this contract does not assume that bandwidth is
/// equal to sample rate or that a particular driver can realize it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RfTuning {
    center_frequency_hz: u64,
    sample_rate: SampleRateRatio,
    bandwidth_hz: Option<u64>,
}

impl RfTuning {
    pub const fn new(
        center_frequency_hz: u64,
        sample_rate: SampleRateRatio,
        bandwidth_hz: Option<u64>,
    ) -> Result<Self, SdrError> {
        if center_frequency_hz == 0 || matches!(bandwidth_hz, Some(0)) {
            return Err(SdrError::InvalidTuning);
        }
        Ok(Self {
            center_frequency_hz,
            sample_rate,
            bandwidth_hz,
        })
    }

    pub const fn center_frequency_hz(self) -> u64 {
        self.center_frequency_hz
    }
    pub const fn sample_rate(self) -> SampleRateRatio {
        self.sample_rate
    }
    pub const fn bandwidth_hz(self) -> Option<u64> {
        self.bandwidth_hz
    }
}

/// Host request for one bounded block. `reader_id` is an opaque host assertion,
/// not proof of driver or physical device identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdrReadRequest {
    reader_id: u64,
    tuning: RfTuning,
    sample_count: usize,
    byte_order: IqByteOrder,
    max_read_duration_ns: u64,
}

impl SdrReadRequest {
    pub const fn new(
        reader_id: u64,
        tuning: RfTuning,
        sample_count: usize,
        byte_order: IqByteOrder,
        max_read_duration_ns: u64,
    ) -> Result<Self, SdrError> {
        if reader_id == 0 {
            return Err(SdrError::InvalidReader);
        }
        if sample_count == 0 || sample_count > MAX_IQ_SAMPLES_PER_READ {
            return Err(SdrError::InvalidSampleCount);
        }
        if max_read_duration_ns == 0 {
            return Err(SdrError::InvalidReadBudget);
        }
        if sample_count.checked_mul(BYTES_PER_IQ_SAMPLE).is_none() {
            return Err(SdrError::CapacityExceeded);
        }
        Ok(Self {
            reader_id,
            tuning,
            sample_count,
            byte_order,
            max_read_duration_ns,
        })
    }

    pub const fn reader_id(self) -> u64 {
        self.reader_id
    }
    pub const fn tuning(self) -> RfTuning {
        self.tuning
    }
    pub const fn sample_count(self) -> usize {
        self.sample_count
    }
    pub const fn byte_order(self) -> IqByteOrder {
        self.byte_order
    }
    /// Maximum elapsed read duration supplied by the host, not an absolute timestamp.
    pub const fn max_read_duration_ns(self) -> u64 {
        self.max_read_duration_ns
    }

    pub const fn required_bytes(self) -> usize {
        // `new` checked this product and sample_count is private.
        self.sample_count * BYTES_PER_IQ_SAMPLE
    }
}

/// Byte order for each signed-i16 I and Q component. Samples are interleaved
/// I then Q; this declaration does not describe any higher-level CSI format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IqByteOrder {
    LittleEndianIq,
    BigEndianIq,
}

/// Reader-reported tuning state. `Reported` is not proof of a hardware lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppliedTuning {
    Unknown,
    Reported(RfTuning),
}

/// Overrun information from a host reader. `NoneReported` only records that the
/// reader reported no overrun; it cannot prove that no samples were lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverrunReport {
    Unknown,
    NoneReported,
    LostSamples(u64),
}

/// Reader-declared instant represented by the batch timestamp. `Unknown` is
/// valid and prevents consumers from assuming a block start or midpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampAnchor {
    Unknown,
    FirstSample,
    LastSample,
    ReadCompletion,
}

/// Fixed unverified status carried by every successful frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureTrust {
    HostReportedUnverified,
}

/// Metadata supplied by the host reader for one returned block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdrReadReport {
    bytes_written: usize,
    sample_count: usize,
    timestamp: Option<ClockStamp>,
    timestamp_anchor: TimestampAnchor,
    applied_tuning: AppliedTuning,
    overrun: OverrunReport,
    stream_id: Option<u64>,
    generation: Option<u32>,
    sequence: Option<u64>,
}

impl SdrReadReport {
    pub const fn new(
        bytes_written: usize,
        sample_count: usize,
        timestamp: Option<ClockStamp>,
        timestamp_anchor: TimestampAnchor,
        applied_tuning: AppliedTuning,
        overrun: OverrunReport,
    ) -> Result<Self, SdrError> {
        if matches!(overrun, OverrunReport::LostSamples(0)) {
            return Err(SdrError::InvalidReport);
        }
        if timestamp.is_none() && timestamp_anchor != TimestampAnchor::Unknown {
            return Err(SdrError::InvalidReport);
        }
        Ok(Self {
            bytes_written,
            sample_count,
            timestamp,
            timestamp_anchor,
            applied_tuning,
            overrun,
            stream_id: None,
            generation: None,
            sequence: None,
        })
    }

    /// Attach host-reported stream continuity metadata. Existing `new` callers
    /// remain source-compatible and leave all continuity fields unknown.
    /// Sequence zero is valid; stream ID and generation zero are reserved.
    pub const fn with_stream_position(
        mut self,
        stream_id: Option<u64>,
        generation: Option<u32>,
        sequence: Option<u64>,
    ) -> Result<Self, SdrError> {
        if matches!(stream_id, Some(0)) || matches!(generation, Some(0)) {
            return Err(SdrError::InvalidReport);
        }
        if stream_id.is_none() && (generation.is_some() || sequence.is_some()) {
            return Err(SdrError::InvalidReport);
        }
        self.stream_id = stream_id;
        self.generation = generation;
        self.sequence = sequence;
        Ok(self)
    }

    pub const fn bytes_written(self) -> usize {
        self.bytes_written
    }
    pub const fn sample_count(self) -> usize {
        self.sample_count
    }
    pub const fn timestamp(self) -> Option<ClockStamp> {
        self.timestamp
    }
    pub const fn timestamp_anchor(self) -> TimestampAnchor {
        self.timestamp_anchor
    }
    pub const fn applied_tuning(self) -> AppliedTuning {
        self.applied_tuning
    }
    pub const fn overrun(self) -> OverrunReport {
        self.overrun
    }
    pub const fn stream_id(self) -> Option<u64> {
        self.stream_id
    }
    pub const fn generation(self) -> Option<u32> {
        self.generation
    }
    pub const fn sequence(self) -> Option<u64> {
        self.sequence
    }
}

/// Host-owned synchronous acquisition operation. Implementations may call an
/// actual SDR driver; this interface itself performs no device or OS operations.
/// Implementations must honor `request.max_read_duration_ns()`, observe the
/// supplied cancellation flag while waiting, must not unwind,
/// and must not leave unbounded/background acquisition running after return.
/// The host owns cancellation for drivers that can block; this trait cannot
/// interrupt a blocked read.
pub trait SdrReader {
    /// Stable opaque declaration identifying this reader instance. Nonzero is
    /// required, but identity is not cryptographically verified here.
    fn reader_id(&self) -> u64;

    /// Fill exactly `request.required_bytes()` with interleaved signed-i16 IQ
    /// samples or return an error. `report` must describe what the reader
    /// observed, including optional clock, stream/generation/sequence, and
    /// overrun metadata. Missing fields remain unknown.
    /// `acquire_into` clears the caller buffer on errors or inconsistent reports.
    fn read_i16_iq(
        &mut self,
        request: SdrReadRequest,
        cancelled: &AtomicBool,
        output: &mut [u8],
    ) -> Result<SdrReadReport, ReaderError>;
}

/// A successful bounded IQ block with only host-declared provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdrFrame<'a> {
    reader_id: u64,
    requested_tuning: RfTuning,
    applied_tuning: AppliedTuning,
    sample_count: usize,
    byte_order: IqByteOrder,
    timestamp: Option<ClockStamp>,
    timestamp_anchor: TimestampAnchor,
    overrun: OverrunReport,
    trust: CaptureTrust,
    bytes: &'a [u8],
    stream_id: Option<u64>,
    generation: Option<u32>,
    sequence: Option<u64>,
}

impl<'a> SdrFrame<'a> {
    pub const fn reader_id(self) -> u64 {
        self.reader_id
    }
    pub const fn requested_tuning(self) -> RfTuning {
        self.requested_tuning
    }
    pub const fn applied_tuning(self) -> AppliedTuning {
        self.applied_tuning
    }
    pub const fn sample_count(self) -> usize {
        self.sample_count
    }
    pub const fn byte_order(self) -> IqByteOrder {
        self.byte_order
    }
    pub const fn timestamp(self) -> Option<ClockStamp> {
        self.timestamp
    }
    pub const fn timestamp_anchor(self) -> TimestampAnchor {
        self.timestamp_anchor
    }
    pub const fn overrun(self) -> OverrunReport {
        self.overrun
    }
    pub const fn trust(self) -> CaptureTrust {
        self.trust
    }
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
    pub const fn stream_id(self) -> Option<u64> {
        self.stream_id
    }
    pub const fn generation(self) -> Option<u32> {
        self.generation
    }
    pub const fn sequence(self) -> Option<u64> {
        self.sequence
    }

    /// Describe the borrowed payload using Etalon's existing complex-i16
    /// tensor layout contract. This adds no physical interpretation or unit.
    pub fn sample_layout(self) -> Result<SampleLayout, SdrError> {
        let count = u32::try_from(self.sample_count).map_err(|_| SdrError::InvalidSampleCount)?;
        SampleLayout::new(
            1,
            [count, 0, 0, 0],
            [BYTES_PER_IQ_SAMPLE as u64, 0, 0, 0],
            SampleEncoding::ComplexI16,
        )
        .map_err(|_| SdrError::InvalidSampleLayout)
    }
}

/// Execute one bounded reader callback into caller-owned storage.
///
/// `output.len()` must exactly match the requested block size. The returned
/// view is created only after the reader identity, exact byte/sample counts,
/// clock declaration, tuning declaration type, and overrun declaration have
/// been checked. A reader cannot claim verified trust through this contract.
/// This function does not itself enforce authorization; the host must make the
/// call only after its policy and purpose checks, and must retain its policy
/// decision separately from this unverified capture declaration. Reader errors
/// and inconsistent reports clear `output`; this cannot undo radio-side reads
/// or copies retained by the driver.
pub fn acquire_into<'a, R: SdrReader>(
    reader: &mut R,
    request: SdrReadRequest,
    cancelled: &AtomicBool,
    output: &'a mut [u8],
) -> Result<SdrFrame<'a>, SdrError> {
    let reader_id = reader.reader_id();
    if reader_id == 0 || reader_id != request.reader_id {
        return Err(SdrError::ReaderMismatch);
    }
    if output.len() != request.required_bytes() {
        return Err(SdrError::BufferLength);
    }
    if cancelled.load(Ordering::Acquire) {
        output.fill(0);
        return Err(SdrError::Cancelled);
    }

    let report = match reader.read_i16_iq(request, cancelled, output) {
        Ok(report) => report,
        Err(error) => {
            output.fill(0);
            return Err(SdrError::Reader(error));
        }
    };
    if cancelled.load(Ordering::Acquire) {
        output.fill(0);
        return Err(SdrError::Cancelled);
    }
    if report.bytes_written != output.len() || report.sample_count != request.sample_count {
        output.fill(0);
        return Err(SdrError::IncompleteRead);
    }
    if matches!(report.overrun, OverrunReport::LostSamples(0)) {
        output.fill(0);
        return Err(SdrError::InvalidReport);
    }
    if report.timestamp.is_none() && report.timestamp_anchor != TimestampAnchor::Unknown {
        output.fill(0);
        return Err(SdrError::InvalidReport);
    }

    Ok(SdrFrame {
        reader_id: request.reader_id,
        requested_tuning: request.tuning,
        applied_tuning: report.applied_tuning,
        sample_count: report.sample_count,
        byte_order: request.byte_order,
        timestamp: report.timestamp,
        timestamp_anchor: report.timestamp_anchor,
        overrun: report.overrun,
        trust: CaptureTrust::HostReportedUnverified,
        bytes: output,
        stream_id: report.stream_id,
        generation: report.generation,
        sequence: report.sequence,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReaderError {
    Unavailable,
    Io,
    TuneRejected,
    ClockUnavailable,
    Overrun,
    Timeout,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdrError {
    InvalidReader,
    ReaderMismatch,
    InvalidTuning,
    InvalidSampleCount,
    InvalidReadBudget,
    Cancelled,
    CapacityExceeded,
    BufferLength,
    IncompleteRead,
    InvalidReport,
    InvalidSampleLayout,
    Reader(ReaderError),
    SignalExecution(ExecutionError),
}

/// Caller-buffered derived I/Q power paired with its complete SDR declaration.
/// All metadata remains host-reported and unverified, including after CPU or
/// registered-device processing. The values are scaled I² + Q² and have no
/// calibrated physical unit such as watts or dBm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdrPowerFrame<'a> {
    reader_id: u64,
    requested_tuning: RfTuning,
    applied_tuning: AppliedTuning,
    sample_count: usize,
    byte_order: IqByteOrder,
    timestamp: Option<ClockStamp>,
    timestamp_anchor: TimestampAnchor,
    overrun: OverrunReport,
    trust: CaptureTrust,
    power_values: &'a [f64],
    stream_id: Option<u64>,
    generation: Option<u32>,
    sequence: Option<u64>,
}

impl<'a> SdrPowerFrame<'a> {
    pub const fn reader_id(self) -> u64 {
        self.reader_id
    }
    pub const fn requested_tuning(self) -> RfTuning {
        self.requested_tuning
    }
    pub const fn applied_tuning(self) -> AppliedTuning {
        self.applied_tuning
    }
    pub const fn sample_count(self) -> usize {
        self.sample_count
    }
    pub const fn byte_order(self) -> IqByteOrder {
        self.byte_order
    }
    pub const fn timestamp(self) -> Option<ClockStamp> {
        self.timestamp
    }
    pub const fn timestamp_anchor(self) -> TimestampAnchor {
        self.timestamp_anchor
    }
    pub const fn overrun(self) -> OverrunReport {
        self.overrun
    }
    pub const fn trust(self) -> CaptureTrust {
        self.trust
    }
    pub const fn power_values(self) -> &'a [f64] {
        self.power_values
    }
    pub const fn stream_id(self) -> Option<u64> {
        self.stream_id
    }
    pub const fn generation(self) -> Option<u32> {
        self.generation
    }
    pub const fn sequence(self) -> Option<u64> {
        self.sequence
    }
}

/// Execute the existing Etalon signed-i16 I/Q power pipeline with only
/// caller-owned workspace/output buffers, preserving all source-frame
/// declarations in the returned result. This performs no authorization or
/// provider attestation. This compatibility entry point imposes no allocation
/// admission and does not imply a zero-heap provider.
pub fn compute_iq_power_into<'a>(
    frame: SdrFrame<'_>,
    target: ExecutionTarget,
    provider: Option<&mut dyn RegisteredSignalProvider>,
    scale: f32,
    workspace: &mut [f32],
    staged_output: &mut [f64],
    output: &'a mut [f64],
) -> Result<SdrPowerFrame<'a>, SdrError> {
    compute_iq_power_into_with_admission(
        frame,
        target,
        provider,
        scale,
        workspace,
        staged_output,
        output,
        AllocationAdmission::Unrestricted,
    )
}

/// Compute power with an explicit signal-provider allocation admission.
/// Rejected strict admissions occur before provider dispatch or output writes;
/// provider class declarations are not independently measured or verified.
#[allow(clippy::too_many_arguments)]
pub fn compute_iq_power_into_with_admission<'a>(
    frame: SdrFrame<'_>,
    target: ExecutionTarget,
    provider: Option<&mut dyn RegisteredSignalProvider>,
    scale: f32,
    workspace: &mut [f32],
    staged_output: &mut [f64],
    output: &'a mut [f64],
    admission: AllocationAdmission,
) -> Result<SdrPowerFrame<'a>, SdrError> {
    let byte_order = match frame.byte_order {
        IqByteOrder::LittleEndianIq => ByteOrder::LittleEndian,
        IqByteOrder::BigEndianIq => ByteOrder::BigEndian,
    };
    let count = execute_i16_iq_power_into_with_admission(
        target,
        provider,
        frame.bytes,
        byte_order,
        scale,
        workspace,
        staged_output,
        output,
        admission,
    )
    .map_err(SdrError::SignalExecution)?;
    if count != frame.sample_count {
        return Err(SdrError::SignalExecution(ExecutionError::Capacity));
    }
    Ok(SdrPowerFrame {
        reader_id: frame.reader_id,
        requested_tuning: frame.requested_tuning,
        applied_tuning: frame.applied_tuning,
        sample_count: frame.sample_count,
        byte_order: frame.byte_order,
        timestamp: frame.timestamp,
        timestamp_anchor: frame.timestamp_anchor,
        overrun: frame.overrun,
        trust: frame.trust,
        power_values: output,
        stream_id: frame.stream_id,
        generation: frame.generation,
        sequence: frame.sequence,
    })
}

#[cfg(test)]
mod tests;
