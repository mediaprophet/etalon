//! Associate captured Windows frame metadata with the common camera envelope.
use etalon_sense_camera::{
    FrameDeclaration, FrameTimestamp, HostPolicyClaim, NativeTimestamp, NegotiatedFrame,
    ObservationEnvelope, ObservationError, ObservationQuantityMapping,
};

use crate::winrt::WindowsCapturedFrame;

/// The host supplies stream identity, source semantics and IDs for Windows'
/// raw SystemRelativeTime clock and 100 ns unit. This does not authenticate the
/// frame, source, policy claim, timestamp, or radiometric declaration.
/// A reported native time requires both IDs; it is never silently discarded or
/// interpreted as UTC. No synchronized clock mapping is inferred.
#[allow(clippy::too_many_arguments)]
pub fn associate_windows_frame(
    declaration: FrameDeclaration,
    frame: WindowsCapturedFrame,
    stream_id: u64,
    generation: u64,
    quantity: ObservationQuantityMapping,
    native_time_ids: Option<(u64, u64)>,
    policy_claim: Option<HostPolicyClaim>,
) -> Result<ObservationEnvelope, ObservationError> {
    if !matches!(frame.metadata.timestamp, FrameTimestamp::Unknown) {
        return Err(ObservationError::InvalidTimestamp);
    }
    let raw_time = match frame.system_relative_time_100ns {
        Some(ticks) => {
            let (clock_domain_id, unit_id) =
                native_time_ids.ok_or(ObservationError::InvalidTimestamp)?;
            Some(NativeTimestamp::new(ticks, clock_domain_id, unit_id)?)
        }
        None => None,
    };
    let negotiated = NegotiatedFrame {
        source_id: declaration.source_id(),
        modality: declaration.modality(),
        format: frame.format,
        radiometric: declaration.radiometric(),
    };
    ObservationEnvelope::new(
        declaration,
        negotiated,
        frame.metadata,
        stream_id,
        generation,
        quantity,
        raw_time,
        None,
        policy_claim,
    )
}
