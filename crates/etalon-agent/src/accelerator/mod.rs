#[cfg(any(target_os = "android", test))]
mod android;
#[cfg(any(target_os = "linux", target_os = "android", test))]
mod linux;
mod types;
#[cfg(target_os = "windows")]
mod windows;

pub use types::{
    AcceleratorDevice, AcceleratorKind, ExecutionStatus, HardwarePresence, NpuProbeReport,
    ProbeIssue, ProbeStatus, RuntimeAvailability, WorkloadSupport,
};

const MAX_DEVICES: usize = 64;
#[cfg(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "android",
    test
))]
const MAX_PROPERTY_BYTES: usize = 4096;

/// Enumerate OS-visible NPUs without initializing an inference provider.
pub fn probe_npus() -> NpuProbeReport {
    #[cfg(target_os = "windows")]
    {
        windows::probe()
    }
    #[cfg(target_os = "linux")]
    {
        linux::probe(std::path::Path::new("/sys/class/accel"))
    }
    #[cfg(target_os = "android")]
    {
        android::probe(std::path::Path::new("/sys"), std::path::Path::new("/dev"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "android")))]
    {
        let mut report = NpuProbeReport::new(std::env::consts::OS, "none");
        report.status = ProbeStatus::NotProbed;
        report.issue(
            "enumeration",
            "no NPU discovery implementation for this platform",
        );
        report
    }
}
