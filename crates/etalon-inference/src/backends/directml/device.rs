use super::*;
// ─── DmlDevice ───────────────────────────────────────────────────────────────

pub struct DmlDevice {
    pub(super) provider_generation: u64,
    pub d3d12: ID3D12Device,
    pub dml: IDMLDevice,
    pub queue: ID3D12CommandQueue,
    pub adapter_desc: String,
    pub dedicated_vram_bytes: u64,
    pub local_budget_bytes: u64,
    pub local_usage_bytes: u64,
    pub shared_budget_bytes: u64,
    pub shared_usage_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct AdapterMemoryInfo {
    pub adapter_desc: String,
    pub dedicated_vram_bytes: u64,
    pub dedicated_system_memory_bytes: u64,
    pub shared_system_memory_bytes: u64,
    pub local_budget_bytes: u64,
    pub local_usage_bytes: u64,
    pub shared_budget_bytes: u64,
    pub shared_usage_bytes: u64,
}

impl AdapterMemoryInfo {
    /// Available local (dedicated) VRAM.
    pub fn available_local_bytes(&self) -> u64 {
        self.local_budget_bytes
            .saturating_sub(self.local_usage_bytes)
    }

    /// Available shared system memory.
    pub fn available_shared_bytes(&self) -> u64 {
        self.shared_budget_bytes
            .saturating_sub(self.shared_usage_bytes)
    }
}

impl DmlDevice {
    /// Create a D3D12 + DirectML device on the highest-VRAM adapter.
    pub fn new() -> windows::core::Result<Self> {
        unsafe {
            #[cfg(debug_assertions)]
            {
                let mut debug: Option<ID3D12Debug> = None;
                let _ = D3D12GetDebugInterface(&mut debug);
                if let Some(d) = debug {
                    d.EnableDebugLayer();
                }
            }

            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let adapter = Self::best_adapter(&factory)?;
            let memory = Self::adapter_memory(&adapter);
            let adapter_desc = memory.adapter_desc.clone();
            log::info!(
                "DirectML device initialization: using adapter {}",
                adapter_desc
            );

            let mut d3d12_opt: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut d3d12_opt)?;
            let d3d12 = d3d12_opt.unwrap();

            let queue: ID3D12CommandQueue =
                d3d12.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    Priority: D3D12_COMMAND_QUEUE_PRIORITY_HIGH.0,
                    Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                    NodeMask: 0,
                })?;

            #[cfg(debug_assertions)]
            let dml_flags = DML_CREATE_DEVICE_FLAGS(DML_CREATE_DEVICE_FLAG_DEBUG.0);
            #[cfg(not(debug_assertions))]
            let dml_flags = DML_CREATE_DEVICE_FLAGS(DML_CREATE_DEVICE_FLAG_NONE.0);

            let mut dml_opt: Option<IDMLDevice> = None;
            DMLCreateDevice(&d3d12, dml_flags, &mut dml_opt)?;
            let dml = dml_opt.unwrap();
            log::info!("DirectML device initialization: device ready");

            Ok(Self {
                provider_generation: {
                    static NEXT_PROVIDER: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(1);
                    NEXT_PROVIDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                },
                d3d12,
                dml,
                queue,
                adapter_desc,
                dedicated_vram_bytes: memory.dedicated_vram_bytes,
                local_budget_bytes: memory.local_budget_bytes,
                local_usage_bytes: memory.local_usage_bytes,
                shared_budget_bytes: memory.shared_budget_bytes,
                shared_usage_bytes: memory.shared_usage_bytes,
            })
        }
    }

    unsafe fn best_adapter(factory: &IDXGIFactory4) -> windows::core::Result<IDXGIAdapter1> {
        let mut best: Option<IDXGIAdapter1> = None;
        let mut best_vram = 0u64;
        let mut idx = 0u32;
        loop {
            match factory.EnumAdapters1(idx) {
                Ok(a) => {
                    if let Ok(desc) = a.GetDesc1() {
                        // DXGI_ADAPTER_FLAG_SOFTWARE = 4
                        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                            let vram = desc.DedicatedVideoMemory as u64;
                            if vram > best_vram {
                                best_vram = vram;
                                best = Some(a);
                            }
                        }
                    }
                    idx += 1;
                }
                Err(_) => break,
            }
        }
        best.map_or_else(|| factory.EnumAdapters1(0), Ok)
    }

    unsafe fn adapter_name(adapter: &IDXGIAdapter1) -> String {
        if let Ok(desc) = adapter.GetDesc1() {
            return String::from_utf16_lossy(&desc.Description)
                .trim_end_matches('\0')
                .to_string();
        }
        "Unknown GPU".into()
    }

    unsafe fn adapter_memory(adapter: &IDXGIAdapter1) -> AdapterMemoryInfo {
        let desc = adapter.GetDesc1().ok();
        let local = Self::query_video_memory(adapter, DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
        let shared = Self::query_video_memory(adapter, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
        AdapterMemoryInfo {
            adapter_desc: Self::adapter_name(adapter),
            dedicated_vram_bytes: desc
                .as_ref()
                .map(|d| d.DedicatedVideoMemory as u64)
                .unwrap_or(0),
            dedicated_system_memory_bytes: desc
                .as_ref()
                .map(|d| d.DedicatedSystemMemory as u64)
                .unwrap_or(0),
            shared_system_memory_bytes: desc
                .as_ref()
                .map(|d| d.SharedSystemMemory as u64)
                .unwrap_or(0),
            local_budget_bytes: local.as_ref().map(|m| m.Budget).unwrap_or(0),
            local_usage_bytes: local.as_ref().map(|m| m.CurrentUsage).unwrap_or(0),
            shared_budget_bytes: shared.as_ref().map(|m| m.Budget).unwrap_or(0),
            shared_usage_bytes: shared.as_ref().map(|m| m.CurrentUsage).unwrap_or(0),
        }
    }

    unsafe fn query_video_memory(
        adapter: &IDXGIAdapter1,
        group: DXGI_MEMORY_SEGMENT_GROUP,
    ) -> Option<DXGI_QUERY_VIDEO_MEMORY_INFO> {
        let adapter3: IDXGIAdapter3 = adapter.cast().ok()?;
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        if adapter3.QueryVideoMemoryInfo(0, group, &mut info).is_ok() {
            Some(info)
        } else {
            None
        }
    }
}

/// Probes the best DML adapter's memory; `Err` when none.
pub fn probe_best_adapter_memory() -> windows::core::Result<AdapterMemoryInfo> {
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter = DmlDevice::best_adapter(&factory)?;
        Ok(DmlDevice::adapter_memory(&adapter))
    }
}

/// Probes the NPU adapter's memory; `Err` when none.
pub fn probe_npu_adapter_memory() -> windows::core::Result<AdapterMemoryInfo> {
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let mut idx = 0u32;
        loop {
            match factory.EnumAdapters1(idx) {
                Ok(a) => {
                    let name = DmlDevice::adapter_name(&a).to_uppercase();
                    if name.contains("NPU") || name.contains("NEURAL") {
                        return Ok(DmlDevice::adapter_memory(&a));
                    }
                    idx += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }
}
