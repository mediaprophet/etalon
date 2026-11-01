use std::fs;
use std::io::Read;
use std::path::Path;

use super::{
    npu_driver, AcceleratorDevice, AcceleratorKind, NpuProbeReport, ProbeStatus, MAX_DEVICES,
    MAX_PROPERTY_BYTES,
};

fn bounded_text(path: &Path) -> Result<String, String> {
    let file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut data = Vec::new();
    file.take((MAX_PROPERTY_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|error| error.to_string())?;
    if data.len() > MAX_PROPERTY_BYTES {
        return Err("sysfs property exceeds the probe byte limit".into());
    }
    String::from_utf8(data)
        .map(|text| text.trim().to_owned())
        .map_err(|error| error.to_string())
}

pub(super) fn probe(root: &Path) -> NpuProbeReport {
    let mut report = NpuProbeReport::new("linux", "sysfs-accel");
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            report.status = if error.kind() == std::io::ErrorKind::NotFound {
                ProbeStatus::Unavailable
            } else {
                ProbeStatus::Failed
            };
            report.issue("read_dir", error.to_string());
            return report;
        }
    };
    report.status = ProbeStatus::Complete;
    for (index, entry) in entries.enumerate() {
        if index >= MAX_DEVICES * 2 {
            report.status = ProbeStatus::Partial;
            report.issue("enumeration_limit", "sysfs directory entry limit reached");
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                report.status = ProbeStatus::Partial;
                report.issue("read_dir_entry", error.to_string());
                continue;
            }
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.strip_prefix("accel").is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        }) {
            continue;
        }
        if report.devices.len() >= MAX_DEVICES {
            report.status = ProbeStatus::Partial;
            report.issue("device_limit", "accelerator device limit reached");
            break;
        }
        let path = entry.path();
        let device_path = path.join("device");
        let id = match device_path.canonicalize() {
            Ok(id) => id.to_string_lossy().into_owned(),
            Err(error) => {
                report.status = ProbeStatus::Partial;
                report.issue("device_path", error.to_string());
                path.to_string_lossy().into_owned()
            }
        };
        let driver = match fs::read_link(device_path.join("driver")) {
            Ok(link) => link
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned),
            Err(error) => {
                report.status = ProbeStatus::Partial;
                report.issue("driver", error.to_string());
                None
            }
        };
        let kind = if driver.as_deref().is_some_and(npu_driver) {
            AcceleratorKind::Npu
        } else {
            AcceleratorKind::OtherAccelerator
        };
        let mut device = AcceleratorDevice::detected(format!("sysfs:{id}"), name.to_owned(), kind);
        device.driver = driver;
        device.driver_version = bounded_text(&device_path.join("driver/module/version")).ok();
        report.devices.push(device);
    }
    report.devices.sort_by(|left, right| left.id.cmp(&right.id));
    report
}

#[cfg(test)]
mod tests {
    use super::super::HardwarePresence;
    use super::*;

    #[test]
    fn only_documented_npu_drivers_are_classified_as_npus() {
        for driver in [
            "ivpu",
            "amdxdna",
            "qaic",
            "ethosu",
            "ethosn",
            "edgetpu_platform",
        ] {
            assert!(npu_driver(driver));
        }
        for driver in ["fastrpc", "adsprpc", "unknown", "apusys"] {
            assert!(!npu_driver(driver));
        }
    }

    #[test]
    fn missing_kernel_interface_is_not_hardware_absence() {
        let root = tempfile::tempdir().unwrap();
        let report = probe(&root.path().join("missing"));
        assert_eq!(report.status, ProbeStatus::Unavailable);
        assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
    }

    #[test]
    fn empty_visible_interface_reports_no_detected_npu() {
        let root = tempfile::tempdir().unwrap();
        let report = probe(root.path());
        assert_eq!(report.status, ProbeStatus::Complete);
        assert_eq!(report.npu_presence(), HardwarePresence::NotDetected);
    }

    #[test]
    fn unknown_accelerator_is_not_called_an_npu() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("accel0/device")).unwrap();
        let report = probe(root.path());
        assert_eq!(report.devices.len(), 1);
        assert_eq!(report.devices[0].kind, AcceleratorKind::OtherAccelerator);
        assert_eq!(report.status, ProbeStatus::Partial);
        assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
    }

    #[test]
    fn enumeration_limit_cannot_establish_absence() {
        let root = tempfile::tempdir().unwrap();
        for index in 0..MAX_DEVICES + 1 {
            fs::create_dir_all(root.path().join(format!("accel{index}/device"))).unwrap();
        }
        let report = probe(root.path());
        assert_eq!(report.status, ProbeStatus::Partial);
        assert_eq!(report.devices.len(), MAX_DEVICES);
        assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
    }

    #[test]
    fn properties_have_a_byte_limit() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("property");
        fs::write(&path, vec![b'a'; MAX_PROPERTY_BYTES + 1]).unwrap();
        assert!(bounded_text(&path).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn documented_driver_identifies_hardware_without_verifying_execution() {
        let root = tempfile::tempdir().unwrap();
        let device = root.path().join("accel0/device");
        fs::create_dir_all(&device).unwrap();
        std::os::unix::fs::symlink("/sys/bus/pci/drivers/ivpu", device.join("driver")).unwrap();
        let report = probe(root.path());
        assert_eq!(report.npu_presence(), HardwarePresence::Detected);
        assert_eq!(report.devices[0].driver.as_deref(), Some("ivpu"));
        assert!(!report.devices[0].is_verified());
    }
}
