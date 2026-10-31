#![deny(unsafe_code)]
//! Host-fed image-frame validation for visible, near-infrared, and thermal
//! sensors. This crate does not open a device or implement a camera driver.
//!
//! A `FrameProvider` supplies explicit negotiated format/plane layout and
//! writes into a caller-owned byte slice. Format IDs, modality, units, sequence,
//! timestamps, gaps, and calibration are host/provider declarations; this
//! boundary checks shape and association only. Thermal image bytes are never
//! interpreted as temperature. `PolicyMode::HostManaged` is an explicit
//! composition choice for contexts whose policy is managed outside this crate;
//! it does not create or assert authorization.

mod format;
mod ingest;
mod observation;

pub use format::{
    CameraModality, FrameDeclaration, FrameFormat, FrameFormatError, FramePixelFormat, FramePlane,
    RadiometricStatus, MAX_FRAME_PLANES,
};
pub use ingest::{
    FrameError, FrameIngestAdapter, FrameMetadata, FramePolicyGate, FramePolicyMode,
    FramePolicyOperation, FramePolicyScope, FrameProvider, FrameProviderError, FrameTimestamp,
    NegotiatedFrame, PolicyDecision, ProviderFrame,
};
pub use observation::{
    HostPolicyClaim, HostPolicyOutcome, NativeTimestamp, ObservationEnvelope, ObservationError,
    ObservationQuantityMapping,
};

#[cfg(test)]
mod tests;
