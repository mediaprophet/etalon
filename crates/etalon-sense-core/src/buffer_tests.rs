use super::{BufferToken, SampleLayout, SampleRegion};
use crate::{quantity::SampleEncoding, ContractError};

fn token(generation: u32) -> BufferToken {
    BufferToken::new(7, generation).unwrap()
}

fn layout(
    rank: u8,
    extents: [u32; 4],
    strides: [u64; 4],
    encoding: SampleEncoding,
) -> Result<SampleLayout, ContractError> {
    SampleLayout::new(rank, extents, strides, encoding)
}

#[test]
fn tokens_reject_reserved_zero_and_keep_generations_distinct() {
    assert_eq!(BufferToken::new(0, 1), Err(ContractError::InvalidIdentifier));
    assert_eq!(BufferToken::new(1, 0), Err(ContractError::InvalidIdentifier));
    assert_ne!(token(1), token(2));
    assert_eq!(token(2).allocation_id(), 7);
    assert_eq!(token(2).generation(), 2);
}

#[test]
fn layout_checks_rank_active_and_inactive_axes() {
    for rank in [0, 5] {
        assert_eq!(
            layout(rank, [1, 0, 0, 0], [1, 0, 0, 0], SampleEncoding::U8),
            Err(ContractError::InvalidExtent)
        );
    }
    assert_eq!(
        layout(2, [2, 0, 0, 0], [1, 0, 0, 0], SampleEncoding::U8),
        Err(ContractError::InvalidExtent)
    );
    assert_eq!(
        layout(1, [1, 1, 0, 0], [0, 0, 0, 0], SampleEncoding::U8),
        Err(ContractError::InvalidExtent)
    );
    assert_eq!(
        layout(1, [1, 0, 0, 0], [0, 1, 0, 0], SampleEncoding::U8),
        Err(ContractError::InvalidExtent)
    );
    assert!(layout(1, [1, 0, 0, 0], [0; 4], SampleEncoding::U8).is_ok());
}

#[test]
fn layout_rejects_zero_stride_for_repeated_axes_and_aliasing() {
    assert_eq!(
        layout(1, [2, 0, 0, 0], [0, 0, 0, 0], SampleEncoding::I16),
        Err(ContractError::InvalidStride)
    );
    assert_eq!(
        layout(2, [2, 2, 0, 0], [4, 4, 0, 0], SampleEncoding::F32),
        Err(ContractError::InvalidStride)
    );
    assert_eq!(
        layout(1, [2, 0, 0, 0], [1, 0, 0, 0], SampleEncoding::I16),
        Err(ContractError::InvalidStride)
    );
}

#[test]
fn padded_and_transposed_layouts_have_checked_spans() {
    let padded = layout(2, [2, 3, 0, 0], [16, 4, 0, 0], SampleEncoding::F32).unwrap();
    assert_eq!(padded.required_byte_span(), Ok(28));

    let transposed = layout(2, [2, 3, 0, 0], [4, 8, 0, 0], SampleEncoding::F32).unwrap();
    assert_eq!(transposed.required_byte_span(), Ok(24));

    let complex = layout(1, [3, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::ComplexF32).unwrap();
    assert_eq!(complex.required_byte_span(), Ok(24));
}

#[test]
fn layout_reports_u32_extents_and_u64_stride_overflow() {
    let large = layout(
        1,
        [u32::MAX, 0, 0, 0],
        [1, 0, 0, 0],
        SampleEncoding::U8,
    )
    .unwrap();
    assert_eq!(large.required_byte_span(), Ok(u64::from(u32::MAX)));

    assert_eq!(
        layout(
            1,
            [u32::MAX, 0, 0, 0],
            [u64::MAX, 0, 0, 0],
            SampleEncoding::U8,
        ),
        Err(ContractError::ArithmeticOverflow)
    );
}

#[test]
fn region_checks_alignment_nonzero_length_and_checked_capacity() {
    assert_eq!(
        SampleRegion::new(token(1), 0, 4, 4, 0),
        Err(ContractError::InvalidAlignment)
    );
    assert_eq!(
        SampleRegion::new(token(1), 0, 4, 4, 3),
        Err(ContractError::InvalidAlignment)
    );
    assert_eq!(
        SampleRegion::new(token(1), 2, 4, 8, 4),
        Err(ContractError::InvalidAlignment)
    );
    assert_eq!(
        SampleRegion::new(token(1), 0, 0, 8, 4),
        Err(ContractError::InvalidExtent)
    );
    assert_eq!(
        SampleRegion::new(token(1), u64::MAX - 3, 8, u64::MAX, 1),
        Err(ContractError::ArithmeticOverflow)
    );
    assert_eq!(
        SampleRegion::new(token(1), 8, 4, 11, 4),
        Err(ContractError::OutOfBounds)
    );
}

#[test]
fn region_accepts_exact_boundary_and_rejects_short_layout_capacity() {
    let sample_layout = layout(2, [2, 3, 0, 0], [16, 4, 0, 0], SampleEncoding::F32).unwrap();
    let exact = SampleRegion::new(token(1), 16, 28, 44, 4).unwrap();
    assert_eq!(exact.validate_layout(&sample_layout), Ok(()));
    assert_eq!(exact.token(), token(1));
    assert_eq!(exact.offset_bytes(), 16);
    assert_eq!(exact.length_bytes(), 28);

    let short = SampleRegion::new(token(1), 16, 27, 43, 4).unwrap();
    assert_eq!(
        short.validate_layout(&sample_layout),
        Err(ContractError::OutOfBounds)
    );
}

#[test]
fn region_requires_compatible_scalar_alignment() {
    let sample_layout = layout(1, [2, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::F32).unwrap();
    let unaligned_for_scalar = SampleRegion::new(token(1), 2, 8, 10, 2).unwrap();
    assert_eq!(
        unaligned_for_scalar.validate_layout(&sample_layout),
        Err(ContractError::InvalidAlignment)
    );
}
