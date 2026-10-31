use etalon_sense_core::{
    adapter::{AdapterError, AdapterId},
    extended_capability::ExtendedCapability,
    quantity::SampleEncoding,
};

/// Adapter configuration. Capability IDs are host declarations and are not
/// verified here. Each capture has an explicit total duration bound measured
/// by the supplied host clock, separately from individual event wait slices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WasapiConfig {
    pub(super) adapter_id: AdapterId,
    pub(super) modality_id: u64,
    pub(super) capability: ExtendedCapability,
    pub(super) max_frame_bytes: usize,
    pub(super) wait_timeout_ms: u32,
    pub(super) max_capture_duration_ms: u32,
}

impl WasapiConfig {
    /// `max_capture_duration_ms` is capped at 60 seconds. Repeated OS wait
    /// timeouts with a stalled host clock fail closed. A blocked OS call itself
    /// cannot be preempted by this synchronous adapter.
    pub fn new(
        adapter_id: AdapterId,
        modality_id: u64,
        capability: ExtendedCapability,
        max_frame_bytes: usize,
        wait_timeout_ms: u32,
        max_capture_duration_ms: u32,
    ) -> Result<Self, AdapterError> {
        if modality_id == 0
            || modality_id != capability.modality_id()
            || capability.rank() != 2
            || max_frame_bytes == 0
            || wait_timeout_ms == 0
            || wait_timeout_ms > 1_000
            || max_capture_duration_ms == 0
            || max_capture_duration_ms > 60_000
            || !matches!(
                capability.encoding(),
                SampleEncoding::I16 | SampleEncoding::F32
            )
        {
            return Err(AdapterError::Unsupported);
        }
        Ok(Self {
            adapter_id,
            modality_id,
            capability,
            max_frame_bytes,
            wait_timeout_ms,
            max_capture_duration_ms,
        })
    }
}
