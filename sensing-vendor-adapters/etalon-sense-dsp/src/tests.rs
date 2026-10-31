use super::*;
use core::cell::Cell;
use etalon_sense_core::{
    processing::OperationId,
    processing_runtime::{AllocationClass, ProviderDispatchGrant, ProviderReport},
    quantity::SampleEncoding,
};

struct TestProvider<'a> { generation: &'a Cell<u32>, processor: ProcessorKind, calls: usize, fail: bool, operation: u64, revision: u32 }
impl LiveProcessingProvider for TestProvider<'_> {
    fn descriptor(&self) -> LiveProviderDescriptor {
        LiveProviderDescriptor::new(7, self.generation.get(), self.processor,
            OperationId::new(self.operation, self.revision).unwrap(), SampleEncoding::U8, SampleEncoding::U8,
            1, [4, 0, 0, 0], 0, AllocationClass::HotZeroHeap, None).unwrap()
    }
    fn lifecycle(&self) -> ProviderLifecycle { ProviderLifecycle::Ready }
    fn dispatch(&mut self, grant: ProviderDispatchGrant, _: &SampleLayout, input: &[u8],
        _: &SampleLayout, output: &mut [u8], _: &mut [u8]) -> Result<ProviderReport, ContractError> {
        self.calls += 1;
        output.copy_from_slice(input);
        if self.fail { return Err(ContractError::InvalidValue); }
        Ok(ProviderReport::new(grant.provider_id(), grant.generation(), grant.operation(),
            input.len() as u64, output.len() as u64))
    }
}

fn request() -> ProcessingRequest {
    ProcessingRequest::new(OperationId::new(3, 1).unwrap(), ProcessorPreference::AnySupported, 0, None).unwrap()
}
fn layout() -> SampleLayout { SampleLayout::new(1, [4, 0, 0, 0], [1, 0, 0, 0], SampleEncoding::U8).unwrap() }
fn constraints() -> DispatchConstraints { DispatchConstraints::new(AllocationClass::HotZeroHeap, 0) }
fn target(id: u64) -> DspTarget { DspTarget::new(1, 2, id).unwrap() }

#[test]
fn exact_target_routes_one_provider_and_missing_target_never_falls_back() {
    let generation = Cell::new(1);
    let mut a = TestProvider { generation: &generation, processor: ProcessorKind::Dsp, calls: 0, fail: false, operation: 3, revision: 1 };
    let mut b = TestProvider { generation: &generation, processor: ProcessorKind::Dsp, calls: 0, fail: false, operation: 3, revision: 1 };
    let mut output = [9; 4];
    {
        let mut bindings = [DspBinding::new(target(1), &mut a).unwrap(), DspBinding::new(target(2), &mut b).unwrap()];
        let receipt = dispatch_dsp(target(2), request(), constraints(), &mut bindings,
            &layout(), &[1, 2, 3, 4], &layout(), &mut output, &mut []).unwrap();
        assert_eq!(receipt.target, target(2));
        assert_eq!(output, [1, 2, 3, 4]);
        assert_eq!(dispatch_dsp(target(3), request(), constraints(), &mut bindings,
            &layout(), &[1; 4], &layout(), &mut output, &mut []), Err(DspError::Unavailable));
        assert_eq!(output, [0; 4]);
    }
    assert_eq!(a.calls, 0);
    assert_eq!(b.calls, 1);
}

#[test]
fn cpu_provider_cannot_register_as_dsp() {
    let generation = Cell::new(1);
    let mut provider = TestProvider { generation: &generation, processor: ProcessorKind::Cpu, calls: 0, fail: false, operation: 3, revision: 1 };
    assert!(matches!(DspBinding::new(target(1), &mut provider), Err(DspError::WrongProcessor)));
    assert_eq!(provider.calls, 0);
}

#[test]
fn one_target_can_register_distinct_operations() {
    let generation = Cell::new(1);
    let mut a = TestProvider { generation: &generation, processor: ProcessorKind::Dsp,
        calls: 0, fail: false, operation: 3, revision: 1 };
    let mut b = TestProvider { generation: &generation, processor: ProcessorKind::Dsp,
        calls: 0, fail: false, operation: 4, revision: 1 };
    let mut output = [9; 4];
    {
        let mut bindings = [DspBinding::new(target(1), &mut a).unwrap(),
            DspBinding::new(target(1), &mut b).unwrap()];
        let second = ProcessingRequest::new(OperationId::new(4, 1).unwrap(),
            ProcessorPreference::Require(ProcessorKind::Dsp), 0, None).unwrap();
        dispatch_dsp(target(1), second, constraints(), &mut bindings,
            &layout(), &[7; 4], &layout(), &mut output, &mut []).unwrap();
        assert_eq!(output, [7; 4]);
    }
    assert_eq!(a.calls, 0);
    assert_eq!(b.calls, 1);
}

#[test]
fn incompatible_cpu_requirement_is_rejected_without_dispatch() {
    let generation = Cell::new(1);
    let mut provider = TestProvider { generation: &generation, processor: ProcessorKind::Dsp,
        calls: 0, fail: false, operation: 3, revision: 1 };
    let mut output = [9; 4];
    {
        let mut bindings = [DspBinding::new(target(1), &mut provider).unwrap()];
        let cpu = ProcessingRequest::new(OperationId::new(3, 1).unwrap(),
            ProcessorPreference::Require(ProcessorKind::Cpu), 0, None).unwrap();
        assert_eq!(dispatch_dsp(target(1), cpu, constraints(), &mut bindings,
            &layout(), &[1; 4], &layout(), &mut output, &mut []), Err(DspError::WrongProcessor));
    }
    assert_eq!(provider.calls, 0);
    assert_eq!(output, [0; 4]);
}

#[test]
fn one_target_can_register_distinct_revisions_of_an_operation() {
    let generation = Cell::new(1);
    let mut a = TestProvider { generation: &generation, processor: ProcessorKind::Dsp,
        calls: 0, fail: false, operation: 3, revision: 1 };
    let mut b = TestProvider { generation: &generation, processor: ProcessorKind::Dsp,
        calls: 0, fail: false, operation: 3, revision: 2 };
    let mut output = [9; 4];
    {
        let mut bindings = [DspBinding::new(target(1), &mut a).unwrap(),
            DspBinding::new(target(1), &mut b).unwrap()];
        let revised = ProcessingRequest::new(OperationId::new(3, 2).unwrap(),
            ProcessorPreference::Require(ProcessorKind::Dsp), 0, None).unwrap();
        dispatch_dsp(target(1), revised, constraints(), &mut bindings,
            &layout(), &[7; 4], &layout(), &mut output, &mut []).unwrap();
    }
    assert_eq!(a.calls, 0);
    assert_eq!(b.calls, 1);
}

#[test]
fn restart_rejects_old_binding_and_clears_output() {
    let generation = Cell::new(1);
    let mut provider = TestProvider { generation: &generation, processor: ProcessorKind::Dsp, calls: 0, fail: false, operation: 3, revision: 1 };
    let mut bindings = [DspBinding::new(target(1), &mut provider).unwrap()];
    generation.set(2);
    let mut output = [9; 4];
    assert_eq!(dispatch_dsp(target(1), request(), constraints(), &mut bindings,
        &layout(), &[1; 4], &layout(), &mut output, &mut []), Err(DspError::StaleBinding));
    assert_eq!(output, [0; 4]);
}

#[test]
fn duplicate_registry_and_provider_error_clear_output() {
    let generation = Cell::new(1);
    let mut a = TestProvider { generation: &generation, processor: ProcessorKind::Dsp, calls: 0, fail: true, operation: 3, revision: 1 };
    let mut b = TestProvider { generation: &generation, processor: ProcessorKind::Dsp, calls: 0, fail: false, operation: 3, revision: 1 };
    let mut output = [9; 4];
    {
        let mut bindings = [DspBinding::new(target(1), &mut a).unwrap(), DspBinding::new(target(1), &mut b).unwrap()];
        assert_eq!(dispatch_dsp(target(1), request(), constraints(), &mut bindings,
            &layout(), &[1; 4], &layout(), &mut output, &mut []), Err(DspError::DuplicateBinding));
        assert_eq!(output, [0; 4]);
    }
    let mut bindings = [DspBinding::new(target(1), &mut a).unwrap()];
    assert_eq!(dispatch_dsp(target(1), request(), constraints(), &mut bindings,
        &layout(), &[1; 4], &layout(), &mut output, &mut []), Err(DspError::Contract(ContractError::InvalidValue)));
    assert_eq!(output, [0; 4]);
}

struct RacingProvider {
    reads: Cell<usize>,
    generation: Cell<u32>,
    restart_at_read: usize,
    restart_in_dispatch: bool,
    calls: usize,
}
impl LiveProcessingProvider for RacingProvider {
    fn descriptor(&self) -> LiveProviderDescriptor {
        let reads = self.reads.get() + 1;
        self.reads.set(reads);
        if reads >= self.restart_at_read { self.generation.set(2); }
        LiveProviderDescriptor::new(7, self.generation.get(), ProcessorKind::Dsp,
            OperationId::new(3, 1).unwrap(), SampleEncoding::U8, SampleEncoding::U8,
            1, [4, 0, 0, 0], 0, AllocationClass::HotZeroHeap, None).unwrap()
    }
    fn lifecycle(&self) -> ProviderLifecycle { ProviderLifecycle::Ready }
    fn dispatch(&mut self, grant: ProviderDispatchGrant, _: &SampleLayout, input: &[u8],
        _: &SampleLayout, output: &mut [u8], _: &mut [u8]) -> Result<ProviderReport, ContractError> {
        self.calls += 1;
        output.copy_from_slice(input);
        if self.restart_in_dispatch { self.generation.set(2); }
        Ok(ProviderReport::new(grant.provider_id(), grant.generation(), grant.operation(),
            input.len() as u64, output.len() as u64))
    }
}

#[test]
fn restart_between_binding_check_and_core_admission_does_not_issue_new_generation_grant() {
    for restart_at_read in [3, 4, 5] {
        let mut provider = RacingProvider { reads: Cell::new(0), generation: Cell::new(1),
            restart_at_read, restart_in_dispatch: false, calls: 0 };
        let mut output = [9; 4];
        {
            let mut bindings = [DspBinding::new(target(1), &mut provider).unwrap()];
            assert!(dispatch_dsp(target(1), request(), constraints(), &mut bindings,
                &layout(), &[1; 4], &layout(), &mut output, &mut []).is_err());
        }
        assert_eq!(provider.calls, 0);
        assert_eq!(output, [0; 4]);
    }
}

#[test]
fn restart_during_provider_dispatch_rejects_completion_and_clears_output() {
    let mut provider = RacingProvider { reads: Cell::new(0), generation: Cell::new(1),
        restart_at_read: usize::MAX, restart_in_dispatch: true, calls: 0 };
    let mut output = [9; 4];
    {
        let mut bindings = [DspBinding::new(target(1), &mut provider).unwrap()];
        assert_eq!(dispatch_dsp(target(1), request(), constraints(), &mut bindings,
            &layout(), &[1; 4], &layout(), &mut output, &mut []),
            Err(DspError::Contract(ContractError::InvalidSource)));
    }
    assert_eq!(provider.calls, 1);
    assert_eq!(output, [0; 4]);
}
