use etalon_sense_core::{extended_capability::ExtendedCapability, quantity::SampleEncoding};

pub const MAX_FRAME_PLANES: usize = 4;

/// Declared camera measurement family; it does not prove the connected
/// hardware's sensor type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraModality {
    Visible,
    NearInfrared,
    Thermal,
}

/// Data reduction/calibration status supplied by the host. A host declaration
/// is not verified by this crate and never turns pixel codes into temperature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadiometricStatus {
    NotApplicable,
    Unknown,
    HostDeclaredUnverified { calibration_id: u64 },
}

/// Explicit pixel packing for camera buffers. Channel order is named but no
/// color-space normalization, transfer function, or thermal calibration is
/// applied. `OpaqueSamples` preserves the original generic frame API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramePixelFormat {
    OpaqueSamples(SampleEncoding),
    Gray8,
    Gray16,
    Bgra8,
    Rgba8,
}

impl FramePixelFormat {
    pub const fn sample_encoding(self) -> SampleEncoding {
        match self {
            Self::OpaqueSamples(encoding) => encoding,
            Self::Gray8 | Self::Bgra8 | Self::Rgba8 => SampleEncoding::U8,
            Self::Gray16 => SampleEncoding::U16,
        }
    }

    pub const fn channels(self) -> u8 {
        match self {
            Self::Bgra8 | Self::Rgba8 => 4,
            Self::OpaqueSamples(_) | Self::Gray8 | Self::Gray16 => 1,
        }
    }

    pub const fn bytes_per_pixel(self) -> u32 {
        self.sample_encoding().element_bytes() * self.channels() as u32
    }

    fn minimum_dense_bytes(self, width: u32, height: u32) -> Result<u64, FrameFormatError> {
        u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(u64::from(self.bytes_per_pixel())))
            .ok_or(FrameFormatError::ArithmeticOverflow)
    }
}

/// Opaque byte layout for one image plane. `offset_bytes` and `row_stride_bytes`
/// are measured from the start of the complete frame buffer. The format ID
/// gives the host's encoding semantics; this crate does not decode pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FramePlane {
    pub offset_bytes: u32,
    pub row_stride_bytes: u32,
    pub row_bytes: u32,
    pub rows: u32,
}

impl FramePlane {
    pub const fn new(offset_bytes: u32, row_stride_bytes: u32, row_bytes: u32, rows: u32) -> Self {
        Self {
            offset_bytes,
            row_stride_bytes,
            row_bytes,
            rows,
        }
    }

    fn end_byte(self) -> Result<u32, FrameFormatError> {
        if self.row_bytes == 0 || self.rows == 0 || self.row_stride_bytes < self.row_bytes {
            return Err(FrameFormatError::InvalidPlane);
        }
        let prior_rows = self
            .rows
            .checked_sub(1)
            .ok_or(FrameFormatError::InvalidPlane)?;
        let body = prior_rows
            .checked_mul(self.row_stride_bytes)
            .and_then(|bytes| bytes.checked_add(self.row_bytes))
            .ok_or(FrameFormatError::ArithmeticOverflow)?;
        self.offset_bytes
            .checked_add(body)
            .ok_or(FrameFormatError::ArithmeticOverflow)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFormatError {
    InvalidIdentifier,
    InvalidExtent,
    InvalidPlane,
    OverlappingPlanes,
    ArithmeticOverflow,
}

/// Validated host-declared format and plane arrangement. Trailing unused plane
/// entries must be zero-valued. Padding and plane gaps remain part of the frame
/// byte extent and are passed through unchanged. Plane bounding spans, including
/// row padding, must not overlap; interleaved plane arrangements are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameFormat {
    format_id: u64,
    width: u32,
    height: u32,
    frame_bytes: u32,
    encoding: SampleEncoding,
    pixel_format: FramePixelFormat,
    planes: [FramePlane; MAX_FRAME_PLANES],
    plane_count: u8,
}

impl FrameFormat {
    pub fn new(
        format_id: u64,
        width: u32,
        height: u32,
        frame_bytes: u32,
        encoding: SampleEncoding,
        planes: [FramePlane; MAX_FRAME_PLANES],
        plane_count: u8,
    ) -> Result<Self, FrameFormatError> {
        Self::new_with_pixel_format(
            format_id,
            width,
            height,
            frame_bytes,
            FramePixelFormat::OpaqueSamples(encoding),
            planes,
            plane_count,
        )
    }

    /// Construct a frame with an explicit camera pixel packing. The explicit
    /// BGRA/RGBA and gray formats require one full-height plane whose row bytes
    /// cover every pixel; padding is permitted and included in `frame_bytes`.
    pub fn new_with_pixel_format(
        format_id: u64,
        width: u32,
        height: u32,
        frame_bytes: u32,
        pixel_format: FramePixelFormat,
        planes: [FramePlane; MAX_FRAME_PLANES],
        plane_count: u8,
    ) -> Result<Self, FrameFormatError> {
        let count = usize::from(plane_count);
        if format_id == 0 {
            return Err(FrameFormatError::InvalidIdentifier);
        }
        if width == 0 || height == 0 || frame_bytes == 0 || count == 0 || count > MAX_FRAME_PLANES {
            return Err(FrameFormatError::InvalidExtent);
        }
        if planes[count..]
            .iter()
            .any(|plane| *plane != FramePlane::default())
        {
            return Err(FrameFormatError::InvalidPlane);
        }

        if !matches!(pixel_format, FramePixelFormat::OpaqueSamples(_)) {
            let min_row_bytes = width
                .checked_mul(pixel_format.bytes_per_pixel())
                .ok_or(FrameFormatError::ArithmeticOverflow)?;
            if count != 1 || planes[0].rows != height || planes[0].row_bytes < min_row_bytes {
                return Err(FrameFormatError::InvalidPlane);
            }
        }

        let mut ends = [0u32; MAX_FRAME_PLANES];
        for index in 0..count {
            let plane = planes[index];
            if plane.rows > height {
                return Err(FrameFormatError::InvalidPlane);
            }
            let end = plane.end_byte()?;
            if end > frame_bytes {
                return Err(FrameFormatError::InvalidExtent);
            }
            for previous in 0..index {
                let previous_plane = planes[previous];
                if plane.offset_bytes < ends[previous] && previous_plane.offset_bytes < end {
                    return Err(FrameFormatError::OverlappingPlanes);
                }
            }
            ends[index] = end;
        }

        Ok(Self {
            format_id,
            width,
            height,
            frame_bytes,
            encoding: pixel_format.sample_encoding(),
            pixel_format,
            planes,
            plane_count,
        })
    }

    pub const fn format_id(self) -> u64 {
        self.format_id
    }
    pub const fn width(self) -> u32 {
        self.width
    }
    pub const fn height(self) -> u32 {
        self.height
    }
    pub const fn frame_bytes(self) -> u32 {
        self.frame_bytes
    }
    pub const fn encoding(self) -> SampleEncoding {
        self.encoding
    }
    pub const fn pixel_format(self) -> FramePixelFormat {
        self.pixel_format
    }
    pub const fn planes(self) -> [FramePlane; MAX_FRAME_PLANES] {
        self.planes
    }
    pub const fn plane_count(self) -> u8 {
        self.plane_count
    }
}

/// Host declaration bounding a source and its camera frame capability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameDeclaration {
    source_id: u64,
    modality: CameraModality,
    capability: ExtendedCapability,
    pixel_format: FramePixelFormat,
    max_width: u32,
    max_height: u32,
    max_frame_bytes: u32,
    radiometric: RadiometricStatus,
}

impl FrameDeclaration {
    pub fn new(
        source_id: u64,
        modality: CameraModality,
        capability: ExtendedCapability,
        max_width: u32,
        max_height: u32,
        radiometric: RadiometricStatus,
    ) -> Result<Self, FrameFormatError> {
        let max_frame_bytes = u32::try_from(capability.max_payload_bytes())
            .map_err(|_| FrameFormatError::ArithmeticOverflow)?;
        Self::new_with_pixel_format(
            source_id,
            modality,
            capability,
            FramePixelFormat::OpaqueSamples(capability.encoding()),
            max_width,
            max_height,
            max_frame_bytes,
            radiometric,
        )
    }

    /// Declare pixel packing and the maximum whole-frame bytes, including row
    /// padding. Capability extents are `[width,height]` for scalar pixels and
    /// `[width,height,channels]` for interleaved multi-channel pixels.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_pixel_format(
        source_id: u64,
        modality: CameraModality,
        capability: ExtendedCapability,
        pixel_format: FramePixelFormat,
        max_width: u32,
        max_height: u32,
        max_frame_bytes: u32,
        radiometric: RadiometricStatus,
    ) -> Result<Self, FrameFormatError> {
        if source_id == 0 || max_width == 0 || max_height == 0 {
            return Err(FrameFormatError::InvalidIdentifier);
        }
        if capability.max_payload_bytes() == 0 || max_frame_bytes == 0 {
            return Err(FrameFormatError::InvalidExtent);
        }
        if pixel_format.sample_encoding() != capability.encoding()
            || capability.max_extents()[0] != max_width
            || capability.max_extents()[1] != max_height
        {
            return Err(FrameFormatError::InvalidExtent);
        }
        let channels = pixel_format.channels();
        if capability.rank() != (if channels == 1 { 2 } else { 3 })
            || (channels > 1 && capability.max_extents()[2] != u32::from(channels))
        {
            return Err(FrameFormatError::InvalidExtent);
        }
        let minimum_bytes = pixel_format.minimum_dense_bytes(max_width, max_height)?;
        if u64::from(max_frame_bytes) < minimum_bytes {
            return Err(FrameFormatError::InvalidExtent);
        }
        if let RadiometricStatus::HostDeclaredUnverified { calibration_id: 0 } = radiometric {
            return Err(FrameFormatError::InvalidIdentifier);
        }
        Ok(Self {
            source_id,
            modality,
            capability,
            pixel_format,
            max_width,
            max_height,
            max_frame_bytes,
            radiometric,
        })
    }

    pub const fn source_id(self) -> u64 {
        self.source_id
    }
    pub const fn modality(self) -> CameraModality {
        self.modality
    }
    pub const fn capability(self) -> ExtendedCapability {
        self.capability
    }
    pub const fn pixel_format(self) -> FramePixelFormat {
        self.pixel_format
    }
    pub const fn max_width(self) -> u32 {
        self.max_width
    }
    pub const fn max_height(self) -> u32 {
        self.max_height
    }
    pub const fn max_frame_bytes(self) -> u32 {
        self.max_frame_bytes
    }
    pub const fn radiometric(self) -> RadiometricStatus {
        self.radiometric
    }
}

pub(super) fn validate_negotiated_format(
    declaration: FrameDeclaration,
    format: FrameFormat,
    radiometric: RadiometricStatus,
) -> Result<(), FrameFormatError> {
    if format.width > declaration.max_width || format.height > declaration.max_height {
        return Err(FrameFormatError::InvalidExtent);
    }
    if format.encoding != declaration.capability.encoding()
        || format.pixel_format != declaration.pixel_format
        || format.width > declaration.capability.max_extents()[0]
        || format.height > declaration.capability.max_extents()[1]
    {
        return Err(FrameFormatError::InvalidExtent);
    }
    if format
        .pixel_format
        .minimum_dense_bytes(format.width, format.height)?
        > declaration.capability.max_payload_bytes()
        || format.frame_bytes > declaration.max_frame_bytes
    {
        return Err(FrameFormatError::InvalidExtent);
    }
    if radiometric != declaration.radiometric {
        return Err(FrameFormatError::InvalidPlane);
    }
    Ok(())
}
