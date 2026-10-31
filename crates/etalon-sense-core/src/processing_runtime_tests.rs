use super::*;

struct FakeProvider {
    descriptor: LiveProviderDescriptor,
    lifecycle: ProviderLifecycle,
    calls: usize,
    bump_generation: bool,
    bad_report: bool,
}

impl LiveProcessingProvider for FakeProvider {
    fn descriptor(&self) -> LiveProviderDescriptor { self.descriptor }
    fn lifecycle(&self) -> ProviderLifecycle { self.lifecycle }

    fn dispatch(
        &mut self,
        grant: ProviderDispatchGrant,
        _input_layout: &SampleLayout,
        input: &[u8],
        _output_layout: &SampleLayout,
        output: &mut [u8],
        _workspace: &mut [u8],
    ) -> Result<ProviderReport, ContractError> {
        assert_eq!(grant.provider_id(), self.descriptor.provider_id());
        assert_eq!(grant.generation(), self.descriptor.generation());
        assert_eq!(grant.operation(), self.descriptor.operation());
        self.calls += 1;
        for (dst, src) in output.iter_mut().zip(input.iter()) { *dst = *src; }
        if self.bump_generation {
            self.descriptor = descriptor(ProcessorKind::Cpu, self.descriptor.generation() + 1);
        }
        let generation = if self.bad_report { self.descriptor.generation() + 1 } else { grant.generation() };
        Ok(ProviderReport::new(
            grant.provider_id(), generation, grant.operation(),
            if self.bad_report { 0 } else { input.len() as u64 },
            output.len() as u64,
        ))
    }
}

fn descriptor(processor: ProcessorKind, generation: u32) -> LiveProviderDescriptor {
    LiveProviderDescriptor::new(
        11, generation, processor, OperationId::new(7, 2).unwrap(),
        SampleEncoding::F32, SampleEncoding::F32, 1, [8, 0, 0, 0],
        4, AllocationClass::HotZeroHeap, Some(100),
    ).unwrap()
}

fn layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::F32).unwrap()
}

fn request(preference: ProcessorPreference) -> ProcessingRequest {
    ProcessingRequest::new(OperationId::new(7, 2).unwrap(), preference, 4, Some(100)).unwrap()
}

fn provider(processor: ProcessorKind) -> FakeProvider {
    FakeProvider {
        descriptor: descriptor(processor, 3), lifecycle: ProviderLifecycle::Ready,
        calls: 0, bump_generation: false, bad_report: false,
    }
}

#[test]
fn live_dispatch_checks_buffers_and_returns_provider_assertion() {
    let mut cpu = provider(ProcessorKind::Cpu);
    let mut list: [&mut dyn LiveProcessingProvider; 1] = [&mut cpu];
    let input = [1, 2, 3, 4, 5, 6, 7, 8, 99];
    let mut output = [0xFF; 10];
    let mut workspace = [0; 4];
    let receipt = dispatch_live(
        request(ProcessorPreference::AnySupported),
        DispatchConstraints::new(AllocationClass::HotZeroHeap, 4),
        &mut list, &layout(), &input, &layout(), &mut output, &mut workspace,
    ).unwrap();
    drop(list);
    assert_eq!(receipt.provider_id, 11);
    assert_eq!(receipt.generation, 3);
    assert_eq!(receipt.processor, ProcessorKind::Cpu);
    assert_eq!(&output[..8], &input[..8]);
    assert_eq!(output[8..], [0xFF; 2]);
    assert_eq!(cpu.calls, 1);
}

#[test]
fn stopped_or_wrong_required_processor_is_never_invoked() {
    let mut cpu = provider(ProcessorKind::Cpu);
    cpu.lifecycle = ProviderLifecycle::Stopped;
    let mut list: [&mut dyn LiveProcessingProvider; 1] = [&mut cpu];
    let mut output = [0xFF; 8];
    let err = dispatch_live(
        request(ProcessorPreference::Require(ProcessorKind::Dsp)),
        DispatchConstraints::new(AllocationClass::HotZeroHeap, 4),
        &mut list, &layout(), &[0; 8], &layout(), &mut output, &mut [0; 4],
    ).unwrap_err();
    drop(list);
    assert_eq!(err, ContractError::UnsupportedOperation);
    assert_eq!(cpu.calls, 0);
    assert_eq!(output, [0; 8]);
}

#[test]
fn restarted_generation_invalidates_completion_and_clears_output() {
    let mut cpu = provider(ProcessorKind::Cpu);
    cpu.bump_generation = true;
    let mut list: [&mut dyn LiveProcessingProvider; 1] = [&mut cpu];
    let mut output = [0xFF; 8];
    let err = dispatch_live(
        request(ProcessorPreference::AnySupported),
        DispatchConstraints::new(AllocationClass::HotZeroHeap, 4),
        &mut list, &layout(), &[1; 8], &layout(), &mut output, &mut [0; 4],
    ).unwrap_err();
    drop(list);
    assert_eq!(err, ContractError::InvalidSource);
    assert_eq!(output, [0; 8]);
}

#[test]
fn malformed_provider_report_cannot_publish_output() {
    let mut cpu = provider(ProcessorKind::Cpu);
    cpu.bad_report = true;
    let mut list: [&mut dyn LiveProcessingProvider; 1] = [&mut cpu];
    let mut output = [0xFF; 8];
    let err = dispatch_live(
        request(ProcessorPreference::AnySupported),
        DispatchConstraints::new(AllocationClass::HotZeroHeap, 4),
        &mut list, &layout(), &[1; 8], &layout(), &mut output, &mut [0; 4],
    ).unwrap_err();
    drop(list);
    assert_eq!(err, ContractError::InvalidSource);
    assert_eq!(output, [0; 8]);
}

#[test]
fn allocation_class_shape_and_latency_are_admission_constraints() {
    let mut cpu = provider(ProcessorKind::Cpu);
    let mut list: [&mut dyn LiveProcessingProvider; 1] = [&mut cpu];
    let mut output = [0xFF; 8];
    let err = dispatch_live(
        request(ProcessorPreference::AnySupported),
        DispatchConstraints::new(AllocationClass::ColdBounded, 4),
        &mut list, &layout(), &[0; 8], &layout(), &mut output, &mut [0; 4],
    ).unwrap_err();
    drop(list);
    assert_eq!(err, ContractError::UnsupportedOperation);
    assert_eq!(cpu.calls, 0);
}

#[test]
fn unbounded_provider_requires_explicit_unbounded_admission() {
    let mut gpu = provider(ProcessorKind::Gpu);
    gpu.descriptor = LiveProviderDescriptor::new(
        11, 3, ProcessorKind::Gpu, OperationId::new(7, 2).unwrap(),
        SampleEncoding::F32, SampleEncoding::F32, 1, [8, 0, 0, 0],
        4, AllocationClass::ColdUnbounded, Some(100),
    ).unwrap();
    let mut output = [0xff; 8];
    for class in [AllocationClass::HotZeroHeap, AllocationClass::ColdBounded] {
        let mut list: [&mut dyn LiveProcessingProvider; 1] = [&mut gpu];
        assert_eq!(dispatch_live(request(ProcessorPreference::AnySupported),
            DispatchConstraints::new(class, 4), &mut list, &layout(), &[1; 8],
            &layout(), &mut output, &mut [0; 4]).unwrap_err(), ContractError::UnsupportedOperation);
    }
    assert_eq!(gpu.calls, 0);
    let mut list: [&mut dyn LiveProcessingProvider; 1] = [&mut gpu];
    assert!(dispatch_live(request(ProcessorPreference::AnySupported),
        DispatchConstraints::new(AllocationClass::ColdUnbounded, 4), &mut list, &layout(),
        &[1; 8], &layout(), &mut output, &mut [0; 4]).is_ok());
}

#[test]
fn short_buffer_fails_before_provider_call() {
    let mut cpu = provider(ProcessorKind::Cpu);
    let mut list: [&mut dyn LiveProcessingProvider; 1] = [&mut cpu];
    let mut output = [0xFF; 4];
    let err = dispatch_live(
        request(ProcessorPreference::AnySupported),
        DispatchConstraints::new(AllocationClass::HotZeroHeap, 4),
        &mut list, &layout(), &[0; 8], &layout(), &mut output, &mut [0; 4],
    ).unwrap_err();
    drop(list);
    assert_eq!(err, ContractError::OutOfBounds);
    assert_eq!(cpu.calls, 0);
    assert_eq!(output, [0; 4]);
}
