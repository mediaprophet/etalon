mod sysfs;
#[cfg(test)]
mod tests;

use super::{
    AcceleratorDevice, AcceleratorKind, HardwarePresence, NpuProbeReport, ProbeStatus, MAX_DEVICES,
};
use std::{fs, path::Path};
use sysfs::Scan;

const PLATFORM_DRIVERS: &[&str] = &["edgetpu_platform", "ethosu", "ethosn"];
const GOOGLE_NODES: &[&str] = &["abrolhos", "janeiro", "edgetpu-soc"];

pub(super) fn probe(sys: &Path, dev: &Path) -> NpuProbeReport {
    let mut scan = Scan::new(sys);
    for (class, known_npu) in [
        ("accel", false),
        ("edgetpu", true),
        ("ethosu", true),
        ("ethosn", true),
    ] {
        for path in scan.entries(&sys.join("class").join(class)) {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !known_npu
                && !name
                    .strip_prefix("accel")
                    .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            {
                continue;
            }
            if let Some(device) = scan.canonical(&path.join("device")) {
                if !device.is_dir() {
                    scan.issue("device_path", "sysfs class device is not a directory");
                    continue;
                }
                let driver = scan.driver(&device);
                add_device(&mut scan, &device, name, driver, known_npu);
            }
        }
    }
    for path in scan.entries(&sys.join("bus/platform/drivers")) {
        let Some(driver) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !PLATFORM_DRIVERS.contains(&driver) {
            continue;
        }
        for bound in scan.entries(&path) {
            match fs::symlink_metadata(&bound) {
                Ok(metadata) if metadata.file_type().is_symlink() => {}
                Ok(_) => continue,
                Err(error) => {
                    scan.issue("bound_device", error.to_string());
                    continue;
                }
            }
            if bound.file_name().is_some_and(|name| name == "module") {
                continue;
            }
            let Some(device) = scan.canonical(&bound) else {
                continue;
            };
            if !device.is_dir() {
                continue;
            }
            let actual_driver = scan.driver(&device);
            if actual_driver.as_deref() != Some(driver) {
                scan.issue(
                    "driver_identity",
                    "platform device does not confirm its bound driver",
                );
                continue;
            }
            let name = device
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(driver);
            add_device(&mut scan, &device, name, actual_driver, true);
        }
    }
    probe_node_metadata(&mut scan, dev);
    if scan.directories_read == 0 && scan.report.devices.is_empty() {
        scan.report.status = ProbeStatus::Unavailable;
    }
    scan.report
        .devices
        .sort_by(|left, right| left.id.cmp(&right.id));
    scan.report
}

fn add_device(scan: &mut Scan, path: &Path, name: &str, driver: Option<String>, known_npu: bool) {
    let id = format!("sysfs:{}", path.display());
    if scan.report.devices.iter().any(|device| device.id == id) {
        return;
    }
    if scan.report.devices.len() >= MAX_DEVICES {
        scan.issue("device_limit", "Android accelerator device limit reached");
        return;
    }
    let kind = if known_npu || driver.as_deref().is_some_and(super::linux::npu_driver) {
        AcceleratorKind::Npu
    } else {
        if driver.is_none() {
            scan.issue("driver_identity", "accelerator driver is not visible");
        }
        AcceleratorKind::OtherAccelerator
    };
    let mut device = AcceleratorDevice::detected(id, name.to_owned(), kind);
    device.driver = driver;
    device.driver_version = scan.text(&path.join("driver/module/version"));
    scan.report.devices.push(device);
}

fn probe_node_metadata(scan: &mut Scan, root: &Path) {
    for name in GOOGLE_NODES {
        let path = root.join(name);
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                scan.issue("device_metadata", format!("{}: {error}", path.display()));
                continue;
            }
        };
        if !is_char_device(&metadata) {
            scan.issue(
                "device_metadata",
                "known TPU device path is not a character device",
            );
            continue;
        }
        if scan.report.npu_presence() == HardwarePresence::Detected {
            continue;
        }
        if scan.report.devices.len() >= MAX_DEVICES {
            scan.issue("device_limit", "Android accelerator device limit reached");
            break;
        }
        let mut device = AcceleratorDevice::detected(
            format!("devnode:{}", path.display()),
            (*name).to_owned(),
            AcceleratorKind::Npu,
        );
        // A device-node name alone does not establish that a physical TPU is bound.
        device.hardware = HardwarePresence::Unknown;
        scan.report.devices.push(device);
        scan.issue(
            "device_metadata",
            "TPU character node is visible; bound hardware remains unverified",
        );
    }
}

#[cfg(unix)]
fn is_char_device(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::FileTypeExt;
    metadata.file_type().is_char_device()
}
#[cfg(not(unix))]
fn is_char_device(_: &fs::Metadata) -> bool {
    false
}
