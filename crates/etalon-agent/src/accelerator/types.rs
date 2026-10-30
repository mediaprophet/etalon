#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeStatus {
    NotProbed,
    Complete,
    Partial,
    Unavailable,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcceleratorKind {
    Cpu,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    SoftwareGpu,
    Npu,
    OtherAccelerator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwarePresence {
    Unknown,
    NotDetected,
    Detected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeAvailability {
    NotProbed,
    Unavailable,
    Available,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadSupport {
    NotProbed,
    Unsupported,
    Supported,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionStatus {
    NotRun,
    Verified,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceleratorDevice {
    pub id: String,
    pub name: String,
    pub kind: AcceleratorKind,
    pub driver: Option<String>,
    pub driver_version: Option<String>,
    pub hardware: HardwarePresence,
    pub runtime: RuntimeAvailability,
    pub workload: WorkloadSupport,
    pub execution: ExecutionStatus,
}

impl AcceleratorDevice {
    pub fn detected(id: String, name: String, kind: AcceleratorKind) -> Self {
        Self {
            id,
            name,
            kind,
            driver: None,
            driver_version: None,
            hardware: HardwarePresence::Detected,
            runtime: RuntimeAvailability::NotProbed,
            workload: WorkloadSupport::NotProbed,
            execution: ExecutionStatus::NotRun,
        }
    }

    pub fn is_verified(&self) -> bool {
        self.hardware == HardwarePresence::Detected
            && self.runtime == RuntimeAvailability::Available
            && self.workload == WorkloadSupport::Supported
            && self.execution == ExecutionStatus::Verified
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeIssue {
    pub operation: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NpuProbeReport {
    pub platform: &'static str,
    pub source: &'static str,
    pub status: ProbeStatus,
    pub devices: Vec<AcceleratorDevice>,
    pub issues: Vec<ProbeIssue>,
}

impl NpuProbeReport {
    pub fn new(platform: &'static str, source: &'static str) -> Self {
        Self {
            platform,
            source,
            status: ProbeStatus::NotProbed,
            devices: Vec::new(),
            issues: Vec::new(),
        }
    }

    /// A negative result describes OS-visible devices, not physically absent silicon.
    pub fn npu_presence(&self) -> HardwarePresence {
        if self.devices.iter().any(|device| {
            device.kind == AcceleratorKind::Npu && device.hardware == HardwarePresence::Detected
        }) {
            HardwarePresence::Detected
        } else if self.status == ProbeStatus::Complete && self.devices.is_empty() {
            HardwarePresence::NotDetected
        } else {
            HardwarePresence::Unknown
        }
    }

    pub(super) fn issue(&mut self, operation: &'static str, message: impl AsRef<str>) {
        if self.issues.len() < super::MAX_DEVICES + 4 {
            self.issues.push(ProbeIssue {
                operation,
                message: message.as_ref().chars().take(1024).collect(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_does_not_establish_runtime_or_execution() {
        let mut device =
            AcceleratorDevice::detected("npu:1".into(), "NPU".into(), AcceleratorKind::Npu);
        assert!(!device.is_verified());
        device.execution = ExecutionStatus::Verified;
        assert!(!device.is_verified());
        device.runtime = RuntimeAvailability::Available;
        assert!(!device.is_verified());
        device.workload = WorkloadSupport::Supported;
        assert!(device.is_verified());
        device.execution = ExecutionStatus::Failed;
        assert!(!device.is_verified());
    }

    #[test]
    fn an_incomplete_probe_cannot_establish_absence() {
        let mut report = NpuProbeReport::new("fixture", "fixture");
        for status in [
            ProbeStatus::NotProbed,
            ProbeStatus::Partial,
            ProbeStatus::Unavailable,
            ProbeStatus::Failed,
        ] {
            report.status = status;
            assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
        }
        report.status = ProbeStatus::Complete;
        assert_eq!(report.npu_presence(), HardwarePresence::NotDetected);
        report.devices.push(AcceleratorDevice::detected(
            "unknown:1".into(),
            "unknown".into(),
            AcceleratorKind::OtherAccelerator,
        ));
        assert_eq!(report.npu_presence(), HardwarePresence::Unknown);
        report.devices.push(AcceleratorDevice::detected(
            "npu:1".into(),
            "NPU".into(),
            AcceleratorKind::Npu,
        ));
        report.status = ProbeStatus::Partial;
        assert_eq!(report.npu_presence(), HardwarePresence::Detected);
    }
}
