use core::ffi::c_void;
use core::mem::size_of;

use ::windows::core::{s, w, Interface, GUID, HRESULT};
use ::windows::Win32::Foundation::{FreeLibrary, HMODULE, LUID};
use ::windows::Win32::Graphics::DXCore::{
    DriverDescription, DriverVersion, IDXCoreAdapter, IDXCoreAdapterFactory, IDXCoreAdapterList,
    InstanceLuid, IsHardware,
};
use ::windows::Win32::System::LibraryLoader::{
    GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32,
};

use super::{
    AcceleratorDevice, AcceleratorKind, NpuProbeReport, ProbeStatus, MAX_DEVICES,
    MAX_PROPERTY_BYTES,
};

// DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU from Microsoft's dxcore_interface.h.
const NPU_ATTRIBUTE: GUID = GUID::from_u128(0xd46140c4_add7_451b_9e56_06fe8c3b58ed);

struct DxCoreLibrary(HMODULE);

impl Drop for DxCoreLibrary {
    fn drop(&mut self) {
        let _ = unsafe { FreeLibrary(self.0) };
    }
}

fn description(adapter: &IDXCoreAdapter) -> Result<String, String> {
    let size =
        unsafe { adapter.GetPropertySize(DriverDescription) }.map_err(|error| error.to_string())?;
    if size == 0 || size > MAX_PROPERTY_BYTES {
        return Err("DXCore description exceeds the property byte limit or is empty".into());
    }
    let mut bytes = vec![0u8; size];
    unsafe { adapter.GetProperty(DriverDescription, size, bytes.as_mut_ptr().cast()) }
        .map_err(|error| error.to_string())?;
    decode_description(&bytes)
}

fn decode_description(bytes: &[u8]) -> Result<String, String> {
    if bytes.is_empty() || bytes.len() > MAX_PROPERTY_BYTES {
        return Err("DXCore description exceeds the property byte limit or is empty".into());
    }
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .ok_or("DXCore description is not terminated")?;
    let name = std::str::from_utf8(&bytes[..end]).map_err(|error| error.to_string())?;
    if name.is_empty() {
        return Err("DXCore description is empty".into());
    }
    Ok(name.to_owned())
}

pub(super) fn probe() -> NpuProbeReport {
    let mut report = NpuProbeReport::new("windows", "dxcore");
    let library =
        match unsafe { LoadLibraryExW(w!("dxcore.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32) } {
            Ok(library) => DxCoreLibrary(library),
            Err(error) => {
                report.status = ProbeStatus::Unavailable;
                report.issue("load_dxcore", error.to_string());
                return report;
            }
        };
    let export = match unsafe { GetProcAddress(library.0, s!("DXCoreCreateAdapterFactory")) } {
        Some(export) => export,
        None => {
            report.status = ProbeStatus::Unavailable;
            report.issue(
                "factory_export",
                "DXCoreCreateAdapterFactory is unavailable",
            );
            return report;
        }
    };
    type CreateFactory = unsafe extern "system" fn(*const GUID, *mut *mut c_void) -> HRESULT;
    // The system export has DXCore's documented factory ABI.
    let create_factory: CreateFactory = unsafe { std::mem::transmute(export) };
    let mut raw_factory = core::ptr::null_mut();
    if let Err(error) =
        unsafe { create_factory(&IDXCoreAdapterFactory::IID, &mut raw_factory) }.ok()
    {
        report.status = ProbeStatus::Failed;
        report.issue("create_factory", error.to_string());
        return report;
    }
    if raw_factory.is_null() {
        report.status = ProbeStatus::Failed;
        report.issue("create_factory", "DXCore returned a null factory");
        return report;
    }
    let factory = unsafe { IDXCoreAdapterFactory::from_raw(raw_factory) };
    let list: IDXCoreAdapterList = match unsafe { factory.CreateAdapterList(&[NPU_ATTRIBUTE]) } {
        Ok(list) => list,
        Err(error) => {
            report.status = ProbeStatus::Failed;
            report.issue("enumerate_npu_attribute", error.to_string());
            return report;
        }
    };
    report.status = ProbeStatus::Complete;
    let count = unsafe { list.GetAdapterCount() };
    if count as usize > MAX_DEVICES {
        report.status = ProbeStatus::Partial;
        report.issue("device_limit", "DXCore accelerator device limit reached");
    }
    for index in 0..count.min(MAX_DEVICES as u32) {
        let adapter: IDXCoreAdapter = match unsafe { list.GetAdapter(index) } {
            Ok(adapter) => adapter,
            Err(error) => {
                report.status = ProbeStatus::Partial;
                report.issue("get_adapter", error.to_string());
                continue;
            }
        };
        if !unsafe { adapter.IsValid() && adapter.IsAttributeSupported(&NPU_ATTRIBUTE) } {
            report.status = ProbeStatus::Partial;
            report.issue(
                "adapter_classification",
                "adapter invalidated or lacks the NPU hardware attribute",
            );
            continue;
        }
        let mut hardware = 0u8;
        if let Err(error) = unsafe {
            adapter.GetProperty(
                IsHardware,
                size_of::<u8>(),
                (&mut hardware as *mut u8).cast(),
            )
        } {
            report.status = ProbeStatus::Partial;
            report.issue("hardware_property", error.to_string());
            continue;
        }
        if hardware != 1 {
            report.status = ProbeStatus::Partial;
            report.issue(
                "hardware_property",
                "NPU adapter is not reported as hardware",
            );
            continue;
        }
        let mut luid = LUID::default();
        let id = match unsafe {
            adapter.GetProperty(
                InstanceLuid,
                size_of::<LUID>(),
                (&mut luid as *mut LUID).cast(),
            )
        } {
            Ok(()) => format!("dxcore:{:08x}{:08x}", luid.HighPart as u32, luid.LowPart),
            Err(error) => {
                report.status = ProbeStatus::Partial;
                report.issue("instance_luid", error.to_string());
                format!("dxcore:unidentified:{index}")
            }
        };
        let name = match description(&adapter) {
            Ok(name) => name,
            Err(error) => {
                report.status = ProbeStatus::Partial;
                report.issue("driver_description", error);
                "DXCore NPU".into()
            }
        };
        let mut device = AcceleratorDevice::detected(id, name, AcceleratorKind::Npu);
        device.driver = Some("DXCore".into());
        let mut version = 0u64;
        match unsafe {
            adapter.GetProperty(
                DriverVersion,
                size_of::<u64>(),
                (&mut version as *mut u64).cast(),
            )
        } {
            Ok(()) => device.driver_version = Some(format!("{version:016x}")),
            Err(error) => {
                report.status = ProbeStatus::Partial;
                report.issue("driver_version", error.to_string());
            }
        }
        report.devices.push(device);
    }
    if unsafe { list.IsStale() } {
        report.status = ProbeStatus::Partial;
        report.issue(
            "adapter_list",
            "adapter topology changed during enumeration",
        );
    }
    report.devices.sort_by(|left, right| left.id.cmp(&right.id));
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_driver_descriptions_are_rejected() {
        assert!(decode_description(&[]).is_err());
        assert!(decode_description(b"unterminated").is_err());
        assert!(decode_description(&[0xff, 0]).is_err());
        assert!(decode_description(&vec![0; MAX_PROPERTY_BYTES + 1]).is_err());
        assert_eq!(decode_description(b"NPU driver\0").unwrap(), "NPU driver");
    }
}
