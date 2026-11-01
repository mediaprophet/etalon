use super::*;
use etalon_sense_core::processing_runtime::AllocationClass;
use etalon_signal::execution::DeviceId;

#[test]
fn prepared_contract_has_real_workspace_and_no_unmeasured_hot_promise() {
    assert_eq!(setup::buffer_bytes(1).unwrap(), 48);
    assert_eq!(
        setup::buffer_bytes(crate::MAX_IQ_PAIRS).unwrap(),
        16_777_248
    );
    assert!(setup::buffer_bytes(0).is_err());
    assert!(setup::buffer_bytes(crate::MAX_IQ_PAIRS + 1).is_err());
    let desc = setup::descriptor(DeviceId::new(7).unwrap(), 2, 4).unwrap();
    assert_eq!(desc.workspace_bytes(), 16);
    assert_eq!(desc.allocation_class(), AllocationClass::ColdUnbounded);
}

#[test]
fn malformed_requests_are_rejected_without_device_access() {
    assert_eq!(execution::preflight(&[0; 8], 1.0, 8, 16, 2), Ok(2));
    assert_eq!(
        execution::preflight(&[0; 7], 1.0, 8, 16, 2),
        Err(PreparedPowerError::InvalidInput)
    );
    assert_eq!(
        execution::preflight(&[0; 8], f32::NAN, 8, 16, 2),
        Err(PreparedPowerError::InvalidScale)
    );
    assert_eq!(
        execution::preflight(&[0; 8], 1.0, 7, 16, 2),
        Err(PreparedPowerError::Capacity)
    );
    assert_eq!(
        execution::preflight(&[0; 8], 1.0, 8, 15, 2),
        Err(PreparedPowerError::Capacity)
    );
}

#[test]
fn prepared_shader_matches_uniform_contract_and_validates() {
    let module = naga::front::wgsl::parse_str(setup::SHADER).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let uniform = module
        .global_variables
        .iter()
        .find(|(_, v)| {
            v.binding
                .as_ref()
                .is_some_and(|b| b.group == 0 && b.binding == 2)
        })
        .unwrap()
        .1;
    assert_eq!(uniform.space, naga::AddressSpace::Uniform);
}

struct Permit;
impl PreparedPowerGate for Permit {
    fn permit_preparation(
        &mut self,
        _: &etalon::Adapter,
        _: PreparedPowerScope,
        _: PreparedPowerStage,
    ) -> bool {
        true
    }
    fn permit(&mut self, _: PreparedPowerScope, _: PreparedPowerStage) -> bool {
        true
    }
}

#[test]
#[ignore = "explicit native Vulkan target qualification"]
fn prepared_iq_power_matches_reference_and_reuses_private_lane() {
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::VULKAN,
        flags: etalon::InstanceFlags::empty(),
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&etalon::RequestAdapterOptions::default()))
            .unwrap();
    let mut gate = Permit;
    let mut provider = PreparedIqPower::new(
        &adapter,
        DeviceId::new(7).unwrap(),
        1,
        2,
        64,
        std::time::Duration::from_secs(5),
        &mut gate,
    )
    .unwrap();
    let mut scratch = [0; 8];
    let mut output = [0; 16];
    provider
        .compute_into(
            &[3, 0, 4, 0, 0, 128, 0, 128],
            etalon_signal::decode::ByteOrder::LittleEndian,
            1.0,
            &mut scratch,
            &mut output,
        )
        .unwrap();
    assert_eq!(f64::from_le_bytes(output[..8].try_into().unwrap()), 25.0);
    assert_eq!(
        f64::from_le_bytes(output[8..].try_into().unwrap()),
        2_147_483_648.0
    );
    provider
        .compute_into(
            &[0, 3, 0, 4],
            etalon_signal::decode::ByteOrder::BigEndian,
            0.5,
            &mut scratch,
            &mut output[..8],
        )
        .unwrap();
    assert_eq!(f64::from_le_bytes(output[..8].try_into().unwrap()), 6.25);
    assert_eq!(scratch, [0; 8]);
    assert_eq!(provider.completed_runs(), 2);
    provider.retire().unwrap();
}

#[test]
fn live_layout_extent_is_bounded_before_host_width_arithmetic() {
    use etalon_sense_core::{buffer::SampleLayout, quantity::SampleEncoding, ContractError};
    let input = SampleLayout::new(
        1,
        [0x4000_0001, 0, 0, 0],
        [4, 0, 0, 0],
        SampleEncoding::ComplexI16,
    )
    .unwrap();
    let output =
        SampleLayout::new(1, [0x4000_0001, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::F64).unwrap();
    assert_eq!(
        execution::validate_layouts(&input, 4, &output, 8, crate::MAX_IQ_PAIRS),
        Err(ContractError::OutOfBounds)
    );
    let input =
        SampleLayout::new(1, [1, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::ComplexI16).unwrap();
    let output = SampleLayout::new(1, [1, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::F64).unwrap();
    assert_eq!(
        execution::validate_layouts(&input, 4, &output, 8, 1),
        Ok(())
    );
    assert_eq!(
        execution::validate_layouts(&input, 4, &output, 7, 1),
        Err(ContractError::OutOfBounds)
    );
}
