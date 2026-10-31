use super::*;

#[test]
fn native_runtime_is_not_admitted_as_zero_heap() {
    let descriptor = make_descriptor(7, 1, 2).unwrap();
    assert_eq!(descriptor.allocation_class(), AllocationClass::ColdUnbounded);
    assert_eq!(descriptor.workspace_bytes(), 2 * wire::HEADER_BYTES as u64 + 24);
    assert_eq!(descriptor.latency_bound_ns(), None);
}

#[test]
fn caller_scratch_is_cleared_on_ordinary_and_unwind_cleanup() {
    let mut bytes = [0xaa; 20];
    { let _scratch = Scratch(&mut bytes); }
    assert_eq!(bytes, [0; 20]);
    bytes.fill(0xaa);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _scratch = Scratch(&mut bytes);
        panic!("planned cleanup probe");
    }));
    assert!(result.is_err());
    assert_eq!(bytes, [0; 20]);
}
