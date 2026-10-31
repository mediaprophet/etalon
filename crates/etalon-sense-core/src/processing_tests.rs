use super::*;

const PROCESSORS: [ProcessorKind; 5] = [
    ProcessorKind::Cpu,
    ProcessorKind::Gpu,
    ProcessorKind::Dsp,
    ProcessorKind::Npu,
    ProcessorKind::DeviceFixedFunction,
];

#[test]
fn required_processor_never_accepts_a_different_processor() {
    for required in PROCESSORS {
        for actual in PROCESSORS {
            assert_eq!(
                ProcessorPreference::Require(required).permits(actual),
                required == actual,
            );
        }
    }
}

#[test]
fn preferred_processor_allows_an_explicitly_reported_fallback() {
    for preferred in PROCESSORS {
        for actual in PROCESSORS {
            assert!(ProcessorPreference::Prefer(preferred).permits(actual));
            assert!(ProcessorPreference::AnySupported.permits(actual));
        }
    }
}

#[test]
fn operation_identity_requires_both_identifier_and_revision() {
    assert_eq!(OperationId::new(0, 1), Err(ContractError::InvalidIdentifier));
    assert_eq!(OperationId::new(1, 0), Err(ContractError::InvalidIdentifier));
    let op = OperationId::new(u64::MAX, u32::MAX).unwrap();
    assert_eq!(op.id(), u64::MAX);
    assert_eq!(op.revision(), u32::MAX);
}

#[test]
fn workspace_limit_is_exact_and_zero_workspace_is_valid() {
    let op = OperationId::new(1, 1).unwrap();
    let request = ProcessingRequest::new(op, ProcessorPreference::AnySupported, 8, None).unwrap();
    assert!(request.check_workspace(8).is_ok());
    assert_eq!(request.check_workspace(9), Err(ContractError::OutOfBounds));
    let zero = ProcessingRequest::new(op, ProcessorPreference::AnySupported, 0, None).unwrap();
    assert!(zero.check_workspace(0).is_ok());
    assert_eq!(zero.check_workspace(1), Err(ContractError::OutOfBounds));
}

#[test]
fn deadline_and_request_fields_are_preserved_without_clock_inference() {
    let op = OperationId::new(7, 3).unwrap();
    assert_eq!(
        ProcessingRequest::new(op, ProcessorPreference::AnySupported, 0, Some(0)),
        Err(ContractError::InvalidTime),
    );
    let preference = ProcessorPreference::Require(ProcessorKind::Dsp);
    let request = ProcessingRequest::new(op, preference, u64::MAX, Some(90)).unwrap();
    assert_eq!(request.operation(), op);
    assert_eq!(request.preference(), preference);
    assert_eq!(request.max_workspace_bytes(), u64::MAX);
    assert_eq!(request.latency_budget_ns(), Some(90));
}
