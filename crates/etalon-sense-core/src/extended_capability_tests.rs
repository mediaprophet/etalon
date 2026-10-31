use super::*;

#[test]
fn extensible_stream_retains_unknown_unit_and_checks_shape() {
    let cap =
        ExtendedCapability::new(101, 202, None, SampleEncoding::I16, [8, 0, 0, 0], None).unwrap();
    let layout = SampleLayout::new(1, [8, 0, 0, 0], [2, 0, 0, 0], SampleEncoding::I16).unwrap();
    assert_eq!(cap.unit_id(), None);
    assert_eq!(cap.modality_id(), 101);
    assert_eq!(cap.quantity_id(), 202);
    assert_eq!(cap.encoding(), SampleEncoding::I16);
    assert_eq!(cap.max_extents(), [8, 0, 0, 0]);
    assert_eq!(cap.rank(), 1);
    assert_eq!(cap.max_sample_rate(), None);
    assert_eq!(cap.max_payload_bytes(), 16);
    assert_eq!(cap.validate_layout(&layout, None), Ok(()));
}

#[test]
fn rejects_empty_ids_and_noncontiguous_shape() {
    assert_eq!(
        ExtendedCapability::new(0, 1, None, SampleEncoding::U8, [1, 0, 0, 0], None),
        Err(ContractError::InvalidIdentifier)
    );
    assert_eq!(
        ExtendedCapability::new(1, 2, None, SampleEncoding::U8, [1, 0, 2, 0], None),
        Err(ContractError::InvalidExtent)
    );
}
