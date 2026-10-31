use super::*;

#[cfg(feature = "unqualified-frame-copy")]
pub(super) fn copy_bitmap(
    bitmap: &SoftwareBitmap,
    declaration: FrameDeclaration,
    byte_budget: u64,
    output: &mut [u8],
) -> Result<FrameFormat, NativeError> {
    let width = u32::try_from(bitmap.PixelWidth()?).map_err(|_| NativeError::InvalidFrame)?;
    let height = u32::try_from(bitmap.PixelHeight()?).map_err(|_| NativeError::InvalidFrame)?;
    let pixel_format = match bitmap.BitmapPixelFormat()? {
        value if value == BitmapPixelFormat::Gray8 => FramePixelFormat::Gray8,
        value if value == BitmapPixelFormat::Gray16 => FramePixelFormat::Gray16,
        value if value == BitmapPixelFormat::Bgra8 => FramePixelFormat::Bgra8,
        value if value == BitmapPixelFormat::Rgba8 => FramePixelFormat::Rgba8,
        _ => return Err(NativeError::UnsupportedPixelFormat),
    };
    if pixel_format != declaration.pixel_format()
        || width == 0 || height == 0
        || width > declaration.max_width() || height > declaration.max_height()
    {
        return Err(NativeError::InvalidFrame);
    }

    // Lock only long enough to read the platform plane descriptor, then close
    // it before asking SoftwareBitmap to copy into a separate bounded buffer.
    let lock = bitmap.LockBuffer(BitmapBufferAccessMode::Read)?;
    let plane_count = lock.GetPlaneCount()?;
    if plane_count != 1 { let _ = lock.Close(); return Err(NativeError::UnsupportedPixelFormat); }
    let plane = match lock.GetPlaneDescription(0) {
        Ok(plane) => plane,
        Err(error) => { let _ = lock.Close(); return Err(error.into()); }
    };
    lock.Close()?;
    let plane_width = u32::try_from(plane.Width).map_err(|_| NativeError::InvalidFrame)?;
    let plane_height = u32::try_from(plane.Height).map_err(|_| NativeError::InvalidFrame)?;
    let offset = u32::try_from(plane.StartIndex).map_err(|_| NativeError::InvalidFrame)?;
    let stride = u32::try_from(plane.Stride).map_err(|_| NativeError::InvalidFrame)?;
    let row_bytes = width.checked_mul(pixel_format.bytes_per_pixel()).ok_or(NativeError::ArithmeticOverflow)?;
    if plane_width != width || plane_height != height {
        return Err(NativeError::InvalidFrame);
    }
    if !copy_to_buffer_layout_supported(offset, stride, row_bytes) {
        // The projection confirms that CopyToBuffer accepts an IBuffer but does
        // not establish its behavior for padded rows or nonzero plane starts.
        // Reject those layouts instead of guessing how offsets/strides map.
        return Err(NativeError::UnsupportedLayout);
    }
    let frame_bytes = row_bytes
        .checked_mul(height)
        .ok_or(NativeError::ArithmeticOverflow)?;
    if u64::from(frame_bytes) > byte_budget {
        return Err(NativeError::ColdAllocationBudgetExceeded {
            required_bytes: u64::from(frame_bytes),
            budget_bytes: byte_budget,
        });
    }
    if frame_bytes > declaration.max_frame_bytes() || output.len() < frame_bytes as usize {
        return Err(NativeError::BufferTooSmall { required: frame_bytes as usize });
    }

    let mut planes = [FramePlane::default(); 4];
    planes[0] = FramePlane::new(offset, stride, row_bytes, height);
    let format_id = match pixel_format {
        FramePixelFormat::Gray8 => 0x4554_4c4e_0000_0001,
        FramePixelFormat::Gray16 => 0x4554_4c4e_0000_0002,
        FramePixelFormat::Bgra8 => 0x4554_4c4e_0000_0003,
        FramePixelFormat::Rgba8 => 0x4554_4c4e_0000_0004,
        FramePixelFormat::OpaqueSamples(_) => return Err(NativeError::UnsupportedPixelFormat),
    };
    let validated = FrameFormat::new_with_pixel_format(
        format_id, width, height, frame_bytes, pixel_format, planes, 1,
    ).map_err(|_| NativeError::InvalidFrame)?;
    // `max_payload_bytes` bounds dense samples. Padding and nonzero plane
    // offsets have already been rejected because CopyToBuffer's mapping for
    // those layouts is not established here.
    let dense = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(u64::from(pixel_format.bytes_per_pixel())))
        .ok_or(NativeError::ArithmeticOverflow)?;
    if dense > declaration.capability().max_payload_bytes()
        || validated.frame_bytes() > declaration.max_frame_bytes()
    {
        return Err(NativeError::InvalidFrame);
    }

    let buffer = Buffer::Create(frame_bytes)?;
    buffer.SetLength(frame_bytes)?;
    bitmap.CopyToBuffer(&buffer)?;
    if buffer.Length()? < frame_bytes { return Err(NativeError::InvalidFrame); }
    let reader = DataReader::FromBuffer(&buffer)?;
    if reader.UnconsumedBufferLength()? < frame_bytes { return Err(NativeError::InvalidFrame); }
    reader.ReadBytes(&mut output[..frame_bytes as usize])?;
    Ok(validated)
}

pub(super) fn validate_copy_allocation(
    allocation: CopyAllocationClass,
    declared_max_frame_bytes: u64,
) -> Result<u64, NativeError> {
    match allocation {
        CopyAllocationClass::HotZeroHeap => Err(NativeError::HotZeroHeapDenied),
        CopyAllocationClass::ColdBounded { byte_budget }
            if byte_budget >= declared_max_frame_bytes => Ok(byte_budget),
        CopyAllocationClass::ColdBounded { byte_budget } => {
            Err(NativeError::ColdAllocationBudgetExceeded {
                required_bytes: declared_max_frame_bytes,
                budget_bytes: byte_budget,
            })
        }
    }
}

#[cfg(any(feature = "unqualified-frame-copy", test))]
fn copy_to_buffer_layout_supported(offset: u32, stride: u32, row_bytes: u32) -> bool {
    offset == 0 && row_bytes != 0 && stride == row_bytes
}

#[cfg(test)]
mod tests {
    use super::{copy_to_buffer_layout_supported, validate_copy_allocation, CopyAllocationClass, NativeError};

    #[test]
    fn copy_buffer_accepts_only_tight_single_plane_layouts() {
        assert!(copy_to_buffer_layout_supported(0, 640, 640));
        assert!(!copy_to_buffer_layout_supported(1, 640, 640));
        assert!(!copy_to_buffer_layout_supported(0, 672, 640));
        assert!(!copy_to_buffer_layout_supported(0, 0, 0));
    }

    #[test]
    fn allocation_admission_rejects_hot_and_requires_declared_cold_budget() {
        assert_eq!(
            validate_copy_allocation(CopyAllocationClass::HotZeroHeap, 4096),
            Err(NativeError::HotZeroHeapDenied),
        );
        assert_eq!(
            validate_copy_allocation(CopyAllocationClass::ColdBounded { byte_budget: 4095 }, 4096),
            Err(NativeError::ColdAllocationBudgetExceeded { required_bytes: 4096, budget_bytes: 4095 }),
        );
        assert_eq!(
            validate_copy_allocation(CopyAllocationClass::ColdBounded { byte_budget: 4096 }, 4096),
            Ok(4096),
        );
    }
}
