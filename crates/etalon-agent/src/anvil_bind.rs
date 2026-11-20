//! N3: bind live Vulkan `deviceUUID` into Anvil.
//!
//! QualiaDB `probe_host` fills CPU/IOMMU facts first. This module appends the
//! GPU die UUID Etalon can actually read. DXGI LUID and WebGPU adapter.info
//! are not written into the 16-byte slot.

use anvil_core::AnvilContext;
use anvil_gpu::{bind_gpu_into_profile, etalon_bind_gpu, AdapterReport};
use etalon_resources::device_caps::GpuAdapterCaps;

/// Snapshot + bind the selected adapter into an Anvil context.
pub fn bind_adapter(ctx: &mut AnvilContext, caps: &GpuAdapterCaps) -> AdapterReport {
    let report = caps.anvil_adapter_report();
    etalon_bind_gpu(ctx, &report);
    report
}

/// Overlay a live UUID onto a NodeProfile64 (file rewrite is the caller's job).
pub fn bind_uuid_into_profile(profile: &mut anvil_core::NodeProfile64, uuid: [u8; 16]) {
    let report = AdapterReport::from_pci_uuid(profile.vendor_id, profile.device_id, uuid);
    bind_gpu_into_profile(profile, &report);
}

/// Vulkan `deviceUUID` for this adapter, or zeros on other backends.
pub fn uuid_from_adapter(adapter: &etalon::Adapter) -> [u8; 16] {
    adapter.vulkan_device_uuid()
}

/// When the selected adapter is not Vulkan, enumerate a Vulkan-only instance
/// and take `deviceUUID` from the physical device whose PCI ids match.
///
/// Returns zeros if Vulkan is missing or no PCI match exists. Does not guess
/// across iGPU/dGPU.
pub async fn live_device_uuid(adapter: &etalon::Adapter) -> [u8; 16] {
    let direct = adapter.vulkan_device_uuid();
    if uuid_nonzero(&direct) {
        return direct;
    }
    let info = adapter.get_info();
    probe_vulkan_device_uuid(info.vendor as u16, info.device as u16).await
}

/// Enumerate Vulkan adapters and return the UUID of the PCI match.
pub async fn probe_vulkan_device_uuid(vendor_id: u16, device_id: u16) -> [u8; 16] {
    let mut desc = etalon::InstanceDescriptor::new_without_display_handle();
    desc.backends = etalon::Backends::VULKAN;
    let instance = etalon::Instance::new(desc);
    let adapters = instance
        .enumerate_adapters(etalon::Backends::VULKAN)
        .await;
    for adapter in adapters {
        let info = adapter.get_info();
        if info.vendor as u16 == vendor_id && info.device as u16 == device_id {
            let uuid = adapter.vulkan_device_uuid();
            if uuid_nonzero(&uuid) {
                return uuid;
            }
        }
    }
    [0; 16]
}

fn uuid_nonzero(uuid: &[u8; 16]) -> bool {
    uuid.iter().any(|b| *b != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anvil_core::{verify_sanctuary_policy, AttestationTier, ISO_VTD_IOMMU, SanctuaryPolicy};

    #[test]
    fn bind_adapter_fills_context_uuid() {
        let mut caps = dummy_caps();
        caps.device_uuid = [0xAA; 16];
        let mut ctx = AnvilContext::empty();
        ctx.capabilities.isolation = ISO_VTD_IOMMU;
        ctx.attestation_tier = AttestationTier::DieSerial;
        let report = bind_adapter(&mut ctx, &caps);
        ctx.seal_machine_anchor();
        assert_eq!(report.device_uuid, [0xAA; 16]);
        assert!(ctx.has_gpu_uuid);
        assert!(verify_sanctuary_policy(
            &ctx,
            SanctuaryPolicy::hardware_isolated()
        ));
    }

    #[test]
    fn spoofed_mac_does_not_enter_report() {
        let mut caps = dummy_caps();
        caps.device_uuid = [0x11; 16];
        let mut ctx = AnvilContext::empty();
        ctx.set_ephemeral_mac([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        let report = bind_adapter(&mut ctx, &caps);
        assert_eq!(report.device_uuid, [0x11; 16]);
        assert_eq!(ctx.ephemeral_mac, [0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert_ne!(&report.device_uuid[..6], &ctx.ephemeral_mac);
    }

    #[test]
    fn uuid_drift_after_seal_trips_sanctuary() {
        let mut caps = dummy_caps();
        caps.device_uuid = [0xAA; 16];
        let mut ctx = AnvilContext::empty();
        ctx.capabilities.isolation = ISO_VTD_IOMMU;
        ctx.attestation_tier = AttestationTier::DieSerial;
        bind_adapter(&mut ctx, &caps);
        ctx.seal_machine_anchor();
        caps.device_uuid = [0x55; 16];
        bind_adapter(&mut ctx, &caps);
        assert!(!verify_sanctuary_policy(
            &ctx,
            SanctuaryPolicy::hardware_isolated()
        ));
    }

    #[test]
    fn bind_uuid_into_profile_round_trip() {
        let mut profile = anvil_core::NodeProfile64::empty();
        bind_uuid_into_profile(&mut profile, [0x55; 16]);
        let parsed = anvil_core::NodeProfile64::parse(&profile.to_bytes()).unwrap();
        assert!(parsed.has_flag(anvil_core::PROFILE_HAS_GPU_UUID));
        assert_eq!(parsed.gpu_device_uuid, [0x55; 16]);
    }

    fn dummy_caps() -> GpuAdapterCaps {
        GpuAdapterCaps {
            name: String::from("Device_001"),
            backend: etalon::Backend::Vulkan,
            device_type: etalon::DeviceType::DiscreteGpu,
            vendor: 0x10DE,
            device: 0x2531,
            driver: String::new(),
            driver_info: String::new(),
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            cooperative_matrix_tile_count: 0,
            features: Default::default(),
            limits: Default::default(),
            device_uuid: [0; 16],
        }
    }
}
