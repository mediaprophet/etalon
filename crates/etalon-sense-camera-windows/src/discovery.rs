//! Bounded enumeration and selection within one initialized MediaCapture.
use crate::winrt::{CameraOperation, CameraPolicy, NativeError};
use ::windows::{
    core::HSTRING, Media::Capture::Frames::MediaFrameSource, Media::Capture::MediaCapture,
};
use etalon_sense_camera::{CameraModality, FrameDeclaration};

pub const MAX_CAMERA_SOURCES: usize = 64;
pub const MAX_CAMERA_SOURCE_ID_UNITS: usize = 1024;

/// Retains the opaque native identifier and raw OS categories. These fields
/// may be identifying; no source ownership or physical modality is inferred.
#[derive(Clone, Debug)]
pub struct CameraSourceCandidate {
    native_id: HSTRING,
    source_kind: i32,
    media_stream_type: i32,
}

impl CameraSourceCandidate {
    pub fn native_id_utf16(&self) -> &[u16] {
        &self.native_id
    }
    pub const fn source_kind(&self) -> i32 {
        self.source_kind
    }
    pub const fn media_stream_type(&self) -> i32 {
        self.media_stream_type
    }
}

/// Host binding for the exact native source, independently of the generic
/// camera-operation policy. The host decides applicable policy and namespace.
pub trait CameraSourceAuthority {
    fn permits(
        &mut self,
        declared_source_id: u64,
        modality: CameraModality,
        operation: CameraOperation,
        candidate: &CameraSourceCandidate,
    ) -> bool;
}

fn operation_allowed(
    policy: &mut impl CameraPolicy,
    declaration: FrameDeclaration,
    operation: CameraOperation,
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        policy.permit(declaration.source_id(), declaration.modality(), operation)
    }))
    .unwrap_or(false)
}

fn candidate_allowed(
    authority: &mut impl CameraSourceAuthority,
    declaration: FrameDeclaration,
    operation: CameraOperation,
    candidate: &CameraSourceCandidate,
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        authority.permits(
            declaration.source_id(),
            declaration.modality(),
            operation,
            candidate,
        )
    }))
    .unwrap_or(false)
}

fn describe(id: HSTRING, source: &MediaFrameSource) -> Result<CameraSourceCandidate, NativeError> {
    if id.is_empty() || id.len() > MAX_CAMERA_SOURCE_ID_UNITS {
        return Err(NativeError::InvalidSource);
    }
    let info = source.Info()?;
    if info.Id()? != id {
        return Err(NativeError::InvalidSource);
    }
    Ok(CameraSourceCandidate {
        native_id: id,
        source_kind: info.SourceKind()?.0,
        media_stream_type: info.MediaStreamType()?.0,
    })
}

pub(crate) fn list_sources(
    capture: &MediaCapture,
    declaration: FrameDeclaration,
    policy: &mut impl CameraPolicy,
    authority: &mut impl CameraSourceAuthority,
    output: &mut [Option<CameraSourceCandidate>],
) -> Result<usize, NativeError> {
    if !operation_allowed(policy, declaration, CameraOperation::EnumerateSources) {
        return Err(NativeError::PolicyDenied);
    }
    let sources = capture.FrameSources()?;
    let count = sources.Size()? as usize;
    if count > MAX_CAMERA_SOURCES {
        return Err(NativeError::SourceLimitExceeded);
    }
    let mut staged: [Option<CameraSourceCandidate>; MAX_CAMERA_SOURCES] =
        core::array::from_fn(|_| None);
    let iterator = sources.First()?;
    let mut index = 0;
    while iterator.HasCurrent()? {
        if index >= count {
            return Err(NativeError::SourceListChanged);
        }
        if !operation_allowed(policy, declaration, CameraOperation::EnumerateSources) {
            return Err(NativeError::PolicyDenied);
        }
        let pair = iterator.Current()?;
        let candidate = describe(pair.Key()?, &pair.Value()?)?;
        if !candidate_allowed(
            authority,
            declaration,
            CameraOperation::EnumerateSources,
            &candidate,
        ) {
            return Err(NativeError::PolicyDenied);
        }
        staged[index] = Some(candidate);
        index += 1;
        iterator.MoveNext()?;
    }
    if index != count || sources.Size()? as usize != count {
        return Err(NativeError::SourceListChanged);
    }
    if !operation_allowed(policy, declaration, CameraOperation::EnumerateSources) {
        return Err(NativeError::PolicyDenied);
    }
    for candidate in staged.iter().take(count) {
        let candidate = candidate.as_ref().ok_or(NativeError::SourceListChanged)?;
        if !candidate_allowed(
            authority,
            declaration,
            CameraOperation::EnumerateSources,
            candidate,
        ) {
            return Err(NativeError::PolicyDenied);
        }
    }
    if !operation_allowed(policy, declaration, CameraOperation::EnumerateSources) {
        return Err(NativeError::PolicyDenied);
    }
    if output.len() < count {
        return Err(NativeError::BufferTooSmall { required: count });
    }
    for (dst, src) in output.iter_mut().zip(staged.iter_mut()).take(count) {
        *dst = src.take();
    }
    Ok(count)
}

pub(crate) fn resolve_candidate(
    capture: &MediaCapture,
    declaration: FrameDeclaration,
    candidate: &CameraSourceCandidate,
    policy: &mut impl CameraPolicy,
    authority: &mut impl CameraSourceAuthority,
) -> Result<MediaFrameSource, NativeError> {
    if !operation_allowed(policy, declaration, CameraOperation::SelectSource) {
        return Err(NativeError::PolicyDenied);
    }
    let source = capture.FrameSources()?.Lookup(&candidate.native_id)?;
    let fresh = describe(candidate.native_id.clone(), &source)?;
    if fresh.source_kind != candidate.source_kind
        || fresh.media_stream_type != candidate.media_stream_type
    {
        return Err(NativeError::SourceListChanged);
    }
    if !candidate_allowed(
        authority,
        declaration,
        CameraOperation::SelectSource,
        &fresh,
    ) || !operation_allowed(policy, declaration, CameraOperation::SelectSource)
    {
        return Err(NativeError::PolicyDenied);
    }
    Ok(source)
}
