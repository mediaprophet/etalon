use super::super::{ExecutionStatus, RuntimeAvailability, WorkloadSupport};
use super::*;

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    for path in [
        "sys/class/accel",
        "sys/class/edgetpu",
        "sys/class/ethosu",
        "sys/class/ethosn",
        "sys/bus/platform/drivers",
        "sys/devices",
        "dev",
    ] {
        fs::create_dir_all(root.path().join(path)).unwrap();
    }
    root
}

fn run(root: &Path) -> NpuProbeReport {
    probe(&root.join("sys"), &root.join("dev"))
}

#[test]
fn android_platform_and_runtime_status_are_honest() {
    let root = fixture();
    let device = root.path().join("sys/class/edgetpu/janeiro/device");
    fs::create_dir_all(&device).unwrap();
    fs::write(device.join("uevent"), "DRIVER=edgetpu_platform\n").unwrap();
    let report = run(root.path());
    assert_eq!(report.platform, "android");
    assert_eq!(report.npu_presence(), HardwarePresence::Detected);
    let device = &report.devices[0];
    assert_eq!(device.driver.as_deref(), Some("edgetpu_platform"));
    assert_eq!(device.runtime, RuntimeAvailability::NotProbed);
    assert_eq!(device.workload, WorkloadSupport::NotProbed);
    assert_eq!(device.execution, ExecutionStatus::NotRun);
    assert!(!device.is_verified());
}

#[test]
fn missing_interfaces_are_unknown_even_with_an_empty_accel_class() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("sys/class/accel")).unwrap();
    let report = run(root.path());
    assert_eq!(report.status, ProbeStatus::Partial);
    assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
    let report = run(&root.path().join("missing"));
    assert_eq!(report.status, ProbeStatus::Unavailable);
    assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
}

#[test]
fn unreadable_interface_is_not_hardware_absence() {
    let root = fixture();
    fs::remove_dir(root.path().join("sys/class/edgetpu")).unwrap();
    fs::write(root.path().join("sys/class/edgetpu"), "not a directory").unwrap();
    let report = run(root.path());
    assert_eq!(report.status, ProbeStatus::Partial);
    assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.operation == "read_dir"));
}

#[test]
fn fastrpc_transport_is_not_npu_evidence() {
    let root = fixture();
    let device = root.path().join("sys/class/accel/accel0/device");
    fs::create_dir_all(&device).unwrap();
    fs::write(device.join("uevent"), "DRIVER=fastrpc\n").unwrap();
    let report = run(root.path());
    assert_eq!(report.devices[0].kind, AcceleratorKind::OtherAccelerator);
    assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
}

#[test]
fn registered_driver_without_bound_devices_is_not_detection() {
    let root = fixture();
    fs::create_dir(
        root.path()
            .join("sys/bus/platform/drivers/edgetpu_platform"),
    )
    .unwrap();
    assert_eq!(
        run(root.path()).npu_presence(),
        HardwarePresence::NotDetected
    );
}

#[test]
fn fake_device_node_is_not_a_tpu() {
    let root = fixture();
    fs::write(root.path().join("dev/abrolhos"), "ordinary file").unwrap();
    let report = run(root.path());
    assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
    assert!(report.devices.is_empty());
}

#[test]
fn oversized_properties_and_directory_limits_fail_closed() {
    let root = fixture();
    let device = root.path().join("sys/class/accel/accel0/device");
    fs::create_dir_all(&device).unwrap();
    fs::write(
        device.join("uevent"),
        vec![b'a'; super::super::MAX_PROPERTY_BYTES + 1],
    )
    .unwrap();
    let report = run(root.path());
    assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.operation == "property_limit"));
    for index in 0..sysfs::MAX_ENTRIES + 1 {
        fs::create_dir_all(
            root.path()
                .join(format!("sys/class/accel/accel{index}/device")),
        )
        .unwrap();
    }
    let report = run(root.path());
    assert_eq!(report.status, ProbeStatus::Partial);
    assert!(report.devices.len() <= MAX_DEVICES);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.operation == "entry_limit"));
}

#[test]
fn cumulative_property_byte_budget_is_enforced() {
    let root = fixture();
    let file = root.path().join("sys/property");
    fs::write(&file, "value").unwrap();
    let mut scan = Scan::new(&root.path().join("sys"));
    for _ in 0..sysfs::MAX_BYTES / super::super::MAX_PROPERTY_BYTES + 1 {
        let _ = scan.text(&file);
    }
    assert_eq!(scan.report.status, ProbeStatus::Partial);
    assert!(scan
        .report
        .issues
        .iter()
        .any(|issue| issue.operation == "byte_limit"));
}

#[cfg(unix)]
#[test]
fn class_and_bound_platform_links_deduplicate_the_same_device() {
    use std::os::unix::fs::symlink;
    let root = fixture();
    let device = root.path().join("sys/devices/platform/1ce00000.abrolhos");
    let driver = root
        .path()
        .join("sys/bus/platform/drivers/edgetpu_platform");
    fs::create_dir_all(&device).unwrap();
    fs::create_dir_all(&driver).unwrap();
    symlink(&driver, device.join("driver")).unwrap();
    symlink(&device, driver.join("1ce00000.abrolhos")).unwrap();
    for name in ["abrolhos", "edgetpu-soc"] {
        let class = root.path().join("sys/class/edgetpu").join(name);
        fs::create_dir(&class).unwrap();
        symlink(&device, class.join("device")).unwrap();
    }
    let report = run(root.path());
    assert_eq!(report.npu_presence(), HardwarePresence::Detected);
    assert_eq!(report.devices.len(), 1);
    assert_eq!(
        report.devices[0].driver.as_deref(),
        Some("edgetpu_platform")
    );
}

#[cfg(unix)]
#[test]
fn platform_binding_detects_tpu_without_accel_or_edgetpu_classes() {
    use std::os::unix::fs::symlink;
    let root = fixture();
    fs::remove_dir(root.path().join("sys/class/edgetpu")).unwrap();
    fs::remove_dir(root.path().join("sys/class/accel")).unwrap();
    let device = root.path().join("sys/devices/platform/janeiro");
    let driver = root
        .path()
        .join("sys/bus/platform/drivers/edgetpu_platform");
    fs::create_dir_all(&device).unwrap();
    fs::create_dir_all(&driver).unwrap();
    symlink(&driver, device.join("driver")).unwrap();
    symlink(&device, driver.join("janeiro")).unwrap();
    let report = run(root.path());
    assert_eq!(report.status, ProbeStatus::Partial);
    assert_eq!(report.npu_presence(), HardwarePresence::Detected);
}

#[cfg(unix)]
#[test]
fn sysfs_links_cannot_escape_the_configured_fixture_root() {
    use std::os::unix::fs::symlink;
    let root = fixture();
    let outside = root.path().join("external");
    fs::create_dir(&outside).unwrap();
    let class = root.path().join("sys/class/edgetpu/abrolhos");
    fs::create_dir(&class).unwrap();
    symlink(&outside, class.join("device")).unwrap();
    let report = run(root.path());
    assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.operation == "sysfs_boundary"));
}

#[cfg(unix)]
#[test]
fn class_root_links_are_checked_before_directory_enumeration() {
    use std::os::unix::fs::symlink;
    let root = fixture();
    let outside = root.path().join("external");
    fs::create_dir(&outside).unwrap();
    fs::create_dir(outside.join("janeiro")).unwrap();
    let class = root.path().join("sys/class/edgetpu");
    fs::remove_dir(&class).unwrap();
    symlink(&outside, &class).unwrap();
    let report = run(root.path());
    assert!(report.devices.is_empty());
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.operation == "sysfs_boundary"));
    assert!(!report
        .issues
        .iter()
        .any(|issue| issue.message.contains("external/janeiro")));
}
