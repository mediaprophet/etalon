#[cfg(any(target_os = "linux", test))]
mod linux;
mod types;
#[cfg(target_os = "windows")]
mod windows;

pub use types::{
    AcceleratorDevice, AcceleratorKind, ExecutionStatus, HardwarePresence, NpuProbeReport,
    ProbeIssue, ProbeStatus, RuntimeAvailability, WorkloadSupport,
};

const MAX_DEVICES: usize = 64;
#[cfg(any(target_os = "windows", target_os = "linux", test))]
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
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
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
