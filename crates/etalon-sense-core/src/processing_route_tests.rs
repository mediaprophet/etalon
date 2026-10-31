use super::*;

fn candidate(id: u64, kind: ProcessorKind, operation: OperationId) -> BackendCandidate {
    BackendCandidate {
        provider_id: id,
        processor: kind,
        operation,
        required_workspace_bytes: 8,
        estimated_latency_ns: Some(10),
        available: true,
    }
}

#[test]
fn preference_selects_eligible_accelerator_and_falls_back_explicitly() {
    let op = OperationId::new(1, 1).unwrap();
    let cpu = candidate(1, ProcessorKind::Cpu, op);
    let gpu = candidate(2, ProcessorKind::Gpu, op);
    let req = ProcessingRequest::new(
        op,
        ProcessorPreference::Prefer(ProcessorKind::Gpu),
        8,
        Some(10),
    )
    .unwrap();
    assert_eq!(select_backend(req, &[cpu, gpu]).unwrap().provider_id, 2);
    assert_eq!(
        select_backend(req, &[cpu]).unwrap().processor,
        ProcessorKind::Cpu
    );
    let required = ProcessingRequest::new(
        op,
        ProcessorPreference::Require(ProcessorKind::Gpu),
        8,
        None,
    )
    .unwrap();
    assert_eq!(select_backend(required, &[cpu]), None);
}

#[test]
fn no_estimate_or_excess_workspace_fails_closed() {
    let op = OperationId::new(1, 1).unwrap();
    let req = ProcessingRequest::new(op, ProcessorPreference::AnySupported, 7, Some(10)).unwrap();
    let mut cpu = candidate(1, ProcessorKind::Cpu, op);
    assert_eq!(select_backend(req, &[cpu]), None);
    cpu.required_workspace_bytes = 7;
    cpu.estimated_latency_ns = None;
    assert_eq!(select_backend(req, &[cpu]), None);
}
