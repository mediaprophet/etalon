use serde::{Deserialize, Serialize};

/// A compute circuit's class in the capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitKind {
    DiscreteGpu,
    IntegratedGpu,
    Cpu,
    Npu,
    Other,
}

impl CircuitKind {
    #[cfg(feature = "gpu")]
    pub(crate) fn from_wgpu(t: etalon::DeviceType) -> Self {
        match t {
            etalon::DeviceType::DiscreteGpu => Self::DiscreteGpu,
            etalon::DeviceType::IntegratedGpu => Self::IntegratedGpu,
            etalon::DeviceType::Cpu => Self::Cpu,
            _ => Self::Other,
        }
    }
}

/// One benchmarked circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBench {
    pub label: String,
    pub kind: CircuitKind,
    pub backend: String,
    /// Milliseconds per GEMV dispatch (lower is faster).
    pub ms_per_gemv: f64,
    /// Effective GFLOP/s for the GEMV (2·n·n flops / time).
    pub gflops: f64,
    /// Host→device upload bandwidth (GB/s) for a representative buffer. The **transfer axis** (D31):
    /// PCIe for a discrete GPU; staging-path for an iGPU (wgpu can't show true zero-copy → relative
    /// signal); `f64::INFINITY` for the CPU (data is already in its pool — no transfer). Decode that
    /// streams weights to a device pays this every token; in-pool compute does not.
    pub upload_gbps: f64,
    /// Relative score in \[0,1\]: fastest circuit = 1.0, others = fastest_ms / this_ms
    /// (or highest decode_proxy_tok_s when decode ranking is active).
    pub rel_score: f64,
    /// Optional real-decode proxy (tok/s) from a short resident decode on a small model.
    /// When present for ≥1 GPU circuit, passport ranking prefers this over GEMV µs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_proxy_tok_s: Option<f64>,
}

impl CircuitBench {
    pub fn timing_source(&self) -> crate::measurement::TimingSource {
        if self.backend == "native" && self.kind == CircuitKind::Cpu {
            crate::measurement::TimingSource::HostWallClock
        } else {
            crate::measurement::TimingSource::SubmitAndCompletionWallClock
        }
    }
}

/// Stable process boundary for one physical adapter/backend benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceBenchmarkRequest {
    pub backend: String,
    pub vendor: u32,
    pub device: u32,
    pub gemv_n: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceBenchmarkResponse {
    pub request: DeviceBenchmarkRequest,
    pub bench: Option<CircuitBench>,
    pub error: Option<String>,
}
