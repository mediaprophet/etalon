use etalon_sense_core::{
    adapter::{AdapterError, AuthorizedExtendedCaptureBuffer},
    buffer::SampleLayout,
    extended_capability::ExtendedCapability,
    time::ClockStamp,
    ContractError,
};

use crate::format::WasapiFormat;

pub(super) fn stable_source_id(endpoint: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in endpoint {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    if hash == 0 { 1 } else { hash }
}

pub(super) fn validate_capability(capability: ExtendedCapability, format: WasapiFormat) -> Result<(), AdapterError> {
    if capability.encoding() != format.encoding() || capability.max_extents()[1] < u32::from(format.channels()) {
        return Err(AdapterError::Unsupported);
    }
    if let Some(max_rate) = capability.max_sample_rate() {
        if f64::from(format.sample_rate_hz()) > max_rate.hz() { return Err(AdapterError::Unsupported); }
    }
    Ok(())
}

pub(super) fn validate_layout(
    layout: &SampleLayout,
    format: WasapiFormat,
    max_frame_bytes: usize,
) -> Result<(), AdapterError> {
    if layout.rank() != 2 || *layout.encoding() != format.encoding()
        || layout.extents()[1] != u32::from(format.channels())
        || layout.strides_bytes()[0] != u64::from(format.block_align())
        || layout.strides_bytes()[1] != u64::from(format.bits_per_sample() / 8)
    {
        return Err(AdapterError::Unsupported);
    }
    let span = usize::try_from(layout.required_byte_span()?)
        .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
    if span == 0 || span > max_frame_bytes { return Err(AdapterError::Buffer(ContractError::OutOfBounds)); }
    Ok(())
}

pub(super) fn clear_prefix(
    output: &mut AuthorizedExtendedCaptureBuffer<'_>,
    frames: u32,
    block_align: u16,
) {
    let bytes = output.writable_bytes();
    let count = usize::try_from(frames).ok()
        .and_then(|frame_count| frame_count.checked_mul(usize::from(block_align)))
        .unwrap_or(bytes.len()).min(bytes.len());
    bytes[..count].fill(0);
}

pub(super) fn clear_all(output: &mut AuthorizedExtendedCaptureBuffer<'_>) {
    output.writable_bytes().fill(0);
}

pub(super) const fn packet_fits_whole(packet_frames: u32, remaining_frames: u32) -> bool {
    packet_frames != 0 && packet_frames <= remaining_frames
}

/// Bound each OS wait by the remaining budget measured with the host's trusted
/// monotonic clock. Uncertainty is fail-closed: earliest possible start and
/// latest possible current time determine whether any budget remains.
pub(super) fn capture_wait_timeout_ms(
    started: ClockStamp,
    now: ClockStamp,
    max_duration_ms: u32,
    max_wait_ms: u32,
) -> Result<u32, AdapterError> {
    if started.clock_id() != now.clock_id() || started.restart_epoch() != now.restart_epoch() {
        return Err(AdapterError::ClockDomainMismatch);
    }
    if max_duration_ms == 0 || max_wait_ms == 0 || now.ticks_ns() < started.ticks_ns() {
        return Err(AdapterError::BackendFailure);
    }
    let duration_ns = u64::from(max_duration_ms)
        .checked_mul(1_000_000)
        .ok_or(AdapterError::BackendFailure)?;
    let deadline_ns = started.interval().lower_ns()
        .checked_add(duration_ns)
        .ok_or(AdapterError::BackendFailure)?;
    let remaining_ns = deadline_ns.checked_sub(now.interval().upper_ns())
        .ok_or(AdapterError::BackendFailure)?;
    let remaining_ms = u32::try_from(remaining_ns / 1_000_000)
        .map_err(|_| AdapterError::BackendFailure)?;
    if remaining_ms == 0 {
        return Err(AdapterError::BackendFailure);
    }
    Ok(max_wait_ms.min(remaining_ms))
}

pub(super) fn timeout_clock_advanced(
    previous: ClockStamp,
    current: ClockStamp,
) -> Result<bool, AdapterError> {
    if previous.clock_id() != current.clock_id()
        || previous.restart_epoch() != current.restart_epoch()
    {
        return Err(AdapterError::ClockDomainMismatch);
    }
    if current.ticks_ns() < previous.ticks_ns() {
        return Err(AdapterError::BackendFailure);
    }
    Ok(current.ticks_ns() > previous.ticks_ns())
}

pub(super) fn finish_stop_state<T>(
    stop_result: Result<(), AdapterError>,
    active: &mut Option<T>,
    generation: &mut u32,
    faulted: &mut bool,
) -> Result<(), AdapterError> {
    if stop_result.is_err() {
        *faulted = true;
        return Err(AdapterError::BackendFailure);
    }
    *active = None;
    match generation.checked_add(1) {
        Some(next_generation) => {
            *generation = next_generation;
            *faulted = false;
            Ok(())
        }
        None => {
            *faulted = true;
            Err(AdapterError::BackendFailure)
        }
    }
}
