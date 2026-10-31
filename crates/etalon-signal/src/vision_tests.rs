use super::*;

#[test]
fn region_mean_excludes_row_padding() {
    let frame = [1, 2, 99, 3, 4, 99];
    assert_eq!(mean_u8_region(&frame, 2, 2, 3, ImageRegion { x: 0, y: 0, width: 2, height: 2 }), Ok(2.5));
}

#[test]
fn invalid_regions_and_short_frames_are_rejected() {
    let roi = ImageRegion { x: 1, y: 1, width: 2, height: 1 };
    assert_eq!(mean_u8_region(&[0; 4], 2, 2, 2, roi), Err(SignalError::InvalidParameter));
    let full = ImageRegion { x: 0, y: 0, width: 2, height: 2 };
    assert_eq!(mean_u8_region(&[0; 3], 2, 2, 2, full), Err(SignalError::LengthMismatch));
}
