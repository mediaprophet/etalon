// Construction stays here; execution, completion, and pipeline ownership live in child modules.
mod lifecycle;
mod operations;
mod pipeline;
mod readback;

use crate::numerical::profiling::TimestampResources;
use std::cell::RefCell;
use std::collections::HashMap;

use super::compute::QualiaCompute;
#[cfg(not(target_arch = "wasm32"))]
use super::memory::MemoryTopology;
use super::memory::{BindingUsage, BufferView, DEFAULT_BINDING_ALIGNMENT, QualiaSlabAllocator};
use super::oracle_ctx::OracleContext;
use crate::numerical::{
    AdapterConstraints, AdapterIdentity, ForgeError, HardwareProfile, KernelSpec, Schedule,
    TargetBackend, emit_shader,
};
#[cfg(not(target_arch = "wasm32"))]
use etalon_resources::device_caps::GpuAdapterCaps;

/// Create a wgpu instance respecting `QUALIA_WGPU_BACKEND`.
///
/// On Windows the default is DX12, but cooperative matrix (`VK_KHR_cooperative_matrix`)
/// is only exposed on the Vulkan backend for NVIDIA hardware. Setting
/// `QUALIA_WGPU_BACKEND=vulkan` routes the forge through Vulkan, un-gating coopmat.
#[cfg(not(target_arch = "wasm32"))]
fn create_instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    if let Some(backends) = crate::numerical::device_options::backend_override() {
        desc.backends = backends;
    } else if cfg!(all(target_os = "windows", feature = "gpu-native-dx12")) {
        desc.backends = wgpu::Backends::DX12;
    } else if cfg!(all(target_os = "windows", not(feature = "gpu-native-dx12"))) {
        desc.backends = wgpu::Backends::GL;
    }
    wgpu::Instance::new(desc)
}

/// Like [`create_instance`] but forces Vulkan when the default backend lacks coopmat.
///
/// This is the un-gating path: on Windows/DX12 where `EXPERIMENTAL_COOPERATIVE_MATRIX`
/// is not advertised, we try Vulkan explicitly. NVIDIA Vulkan drivers expose
/// `VK_KHR_cooperative_matrix` which wgpu maps to `EXPERIMENTAL_COOPERATIVE_MATRIX`.
#[cfg(not(target_arch = "wasm32"))]
fn create_instance_for_coopmat() -> wgpu::Instance {
    // First try the user's explicit backend choice.
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    if let Some(backends) = crate::numerical::device_options::backend_override() {
        desc.backends = backends;
    } else {
        // No explicit override: try Vulkan first (coopmat is available there),
        // falling back to all backends if Vulkan isn't present.
        desc.backends = wgpu::Backends::VULKAN;
    }
    wgpu::Instance::new(desc)
}

/// Find the best adapter for cooperative matrix work.
///
/// Enumerates adapters on the given instance and returns the first that
/// advertises `EXPERIMENTAL_COOPERATIVE_MATRIX`, preferring discrete GPUs.
/// Returns `None` if no adapter has coopmat.
#[cfg(not(target_arch = "wasm32"))]
fn find_coopmat_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    let backends = if let Some(b) = crate::numerical::device_options::backend_override() {
        b
    } else {
        wgpu::Backends::all()
    };
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    // First pass: look for a discrete GPU with coopmat.
    for adapter in &adapters {
        let info = adapter.get_info();
        if info.device_type != wgpu::DeviceType::DiscreteGpu {
            continue;
        }
        let features = adapter.features();
        if features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return Some(adapter.clone());
        }
    }
    // Second pass: any adapter with coopmat.
    for adapter in &adapters {
        let features = adapter.features();
        if features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return Some(adapter.clone());
        }
    }
    None
}

pub struct WgpuComputeContext {
    #[cfg(not(target_arch = "wasm32"))]
    pub provider_lease: etalon_resources::device_lease::DeviceLease,
    #[cfg(not(target_arch = "wasm32"))]
    pub provider_caps: GpuAdapterCaps,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: AdapterIdentity,
    pub constraints: AdapterConstraints,
    /// Rich topology description for `profile-hardware` and cache keying.
    pub profile: HardwareProfile,
    pub allocator: QualiaSlabAllocator,
    /// Backs read-only storage and uniform views (both non-exclusive usages,
    /// so they may share one buffer).
    pub slab: wgpu::Buffer,
    /// Backs read-write storage outputs. wgpu treats read-write storage as an
    /// exclusive usage, so it cannot share a buffer with the read-only inputs in
    /// the same dispatch.
    pub out_slab: wgpu::Buffer,
    /// Backs the **persistent weight region** ([`BindingUsage::StorageReadResident`]): big,
    /// upload-once matrices (a decode layer's projection / FFN weights) that are referenced by
    /// offset across many `run`s instead of being re-uploaded each call. Separate buffer from the
    /// transient ring so [`Self::clear_transient_allocations`] never recycles it.
    pub weight_slab: wgpu::Buffer,
    /// Write-once bump cursor into `weight_slab` (bytes, kept 256-aligned). Weights are never
    /// freed individually; [`Self::clear_weights`] resets it to reuse the region for a new model.
    weight_cursor: usize,
    pub timestamp_supported: bool,
    pub timestamp_period_ns: f32,
    timestamp_resources: Option<TimestampResources>,
    /// Process-lifetime cache of compiled compute pipelines, keyed by `entry\0source`.
    /// Pipeline creation (shader compile + PSO build) is the dominant per-node cost once
    /// the device + slab are reused; a graph re-run over the same kernels (e.g. one decode
    /// block per generated token) then pays zero compile after the first. Survives
    /// [`clear_transient_allocations`](Self::clear_transient_allocations) — pipelines reference
    /// the shader + bind-group *layout*, never the slab buffers (bind groups are rebuilt per
    /// call). `RefCell` because [`compile_pipeline_cached`](Self::compile_pipeline_cached) takes
    /// `&self`; the context is used single-threaded per dispatch (shared dispatch is serialized
    /// behind a `Mutex` by the dispatcher).
    pipeline_cache: RefCell<HashMap<String, wgpu::ComputePipeline>>,
}

impl WgpuComputeContext {
    #[cfg(target_arch = "wasm32")]
    pub fn new(_capacity_bytes: usize) -> Result<Self, ForgeError> {
        Err(ForgeError::GpuUnavailable(
            "browser GPU contexts require asynchronous device admission".into(),
        ))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_for_coopmat(_capacity_bytes: usize) -> Result<Self, ForgeError> {
        Err(ForgeError::GpuUnavailable(
            "native cooperative-matrix device creation is unavailable in browsers".into(),
        ))
    }

    /// Creates `WgpuComputeContext` with the given state.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(capacity_bytes: usize) -> Result<Self, ForgeError> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(prepared) = etalon_resources::provider_scope::prepared_device(false) {
            let owner = prepared.map_err(|error| ForgeError::GpuUnavailable(error.into()))?;
            return Self::from_device(owner.device, owner.queue, &owner.caps, capacity_bytes);
        }
        // Respect QUALIA_WGPU_BACKEND so the forge can use Vulkan (which exposes
        // cooperative matrix on NVIDIA) instead of the Windows DX12 default.
        let instance = create_instance();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|error| ForgeError::GpuUnavailable(error.to_string()))?;
        let info = adapter.get_info();
        let available_features = adapter.features();
        // Request timestamp queries plus subgroup + cooperative-matrix support
        // when the adapter offers them (cooperative-matrix kernels need both).
        let mut wanted =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16;
        if crate::numerical::device_options::experimental_features_allowed() {
            wanted |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
                | wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        }
        let required_features = available_features & wanted;
        let timestamp_supported = required_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let limits = adapter.limits();
        // Populate intrinsic-capability flags from the adapter's real feature set
        // so the tuner can prune schedules that rely on absent hardware (plan §6).
        let mut constraints = AdapterConstraints::from_wgpu_limits(&limits);
        constraints.supports_subgroups = available_features.contains(wgpu::Features::SUBGROUP);
        constraints.supports_coopmat =
            available_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        constraints.supports_rt_cores =
            available_features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        // Warp/wavefront width by vendor: AMD wavefronts are 64, others 32.
        constraints.warp_size = if info.vendor == 0x1002 { 64 } else { 32 };
        // The ray-tracing acceleration-structure limits default to 0, which forbids
        // BLAS/TLAS creation even with EXPERIMENTAL_RAY_QUERY enabled. Raise them to
        // the adapter's supported values (a no-op on adapters that report 0).
        let required_limits = wgpu::Limits {
            max_blas_primitive_count: limits.max_blas_primitive_count,
            max_blas_geometry_count: limits.max_blas_geometry_count,
            max_tlas_instance_count: limits.max_tlas_instance_count,
            max_acceleration_structures_per_shader_stage: limits
                .max_acceleration_structures_per_shader_stage,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits,
            // The cooperative-matrix feature is gated behind wgpu's experimental
            // token. Only requested (above) when the adapter advertises it; the
            // token is harmless when no experimental feature is actually enabled.
            // Safety: we only use it for the cooperative-matrix matmul kernel.
            experimental_features: if required_features.intersects(
                wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
                    | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ) {
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            },
            ..Default::default()
        }))
        .map_err(|error| ForgeError::GpuUnavailable(error.to_string()))?;
        let timestamp_period_ns = if timestamp_supported {
            queue.get_timestamp_period()
        } else {
            0.0
        };

        // Determine topology (unified vs discrete)
        let topology = if info.device_type == wgpu::DeviceType::IntegratedGpu
            || info.device_type == wgpu::DeviceType::Cpu
        {
            MemoryTopology::Unified { zero_copy: true }
        } else {
            MemoryTopology::Discrete {
                staging_required: true,
            }
        };
        let memory_class = match topology {
            MemoryTopology::Unified { .. } => "unified",
            MemoryTopology::Discrete { .. } => "discrete",
        }
        .to_string();

        #[cfg(not(target_arch = "wasm32"))]
        let provider_caps = GpuAdapterCaps::from_adapter(&adapter);
        let adapter = AdapterIdentity {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
        };
        let profile = HardwareProfile {
            adapter: adapter.clone(),
            constraints,
            memory_class,
            supports_timestamp_query: timestamp_supported,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
        };

        let allocator = QualiaSlabAllocator::new(topology, capacity_bytes);

        let slab = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forge-slab"),
            size: capacity_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_slab = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forge-out-slab"),
            size: capacity_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let weight_slab = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forge-weight-slab"),
            size: capacity_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timestamp_resources = timestamp_supported.then(|| TimestampResources::new(&device));

        Ok(Self {
            #[cfg(not(target_arch = "wasm32"))]
            provider_lease: etalon_resources::device_lease::DeviceLease::new(&device, &queue)
                .map_err(ForgeError::GpuUnavailable)?,
            #[cfg(not(target_arch = "wasm32"))]
            provider_caps,
            device,
            queue,
            adapter,
            constraints,
            profile,
            allocator,
            slab,
            out_slab,
            weight_slab,
            weight_cursor: 0,
            timestamp_supported,
            timestamp_period_ns,
            timestamp_resources,
            pipeline_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Like `new()` but tries the Vulkan backend first to find a cooperative-matrix-
    /// capable adapter. On Windows/DX12, `EXPERIMENTAL_COOPERATIVE_MATRIX` is not
    /// advertised, but the same NVIDIA GPU exposes `VK_KHR_cooperative_matrix` via
    /// the Vulkan driver. This constructor:
    ///
    /// 1. Creates a Vulkan-only instance (unless `QUALIA_WGPU_BACKEND` overrides).
    /// 2. Enumerates adapters, looking for one with `EXPERIMENTAL_COOPERATIVE_MATRIX`.
    /// 3. If found, builds the context on that adapter (un-gating coopmat).
    /// 4. If not found, falls back to `new()` (which uses the default backend).
    ///
    /// This is the primary un-gating path for the HLSL WaveMatrix / WGSL coopmat
    /// tensor-core emitters on NVIDIA hardware where DX12 doesn't expose coopmat.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_for_coopmat(capacity_bytes: usize) -> Result<Self, ForgeError> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(prepared) = etalon_resources::provider_scope::prepared_device(true) {
            let owner = prepared.map_err(|error| ForgeError::GpuUnavailable(error.into()))?;
            return Self::from_device(owner.device, owner.queue, &owner.caps, capacity_bytes);
        }
        // Step 1: Try Vulkan instance.
        let vk_instance = create_instance_for_coopmat();

        // Step 2: Find a coopmat-capable adapter.
        if let Some(adapter) = find_coopmat_adapter(&vk_instance) {
            let info = adapter.get_info();
            log::info!(
                "forge|coopmat_adapter|{}|backend={:?}|vendor=0x{:04x}:0x{:04x}",
                info.name,
                info.backend,
                info.vendor,
                info.device
            );
            let available_features = adapter.features();
            let mut wanted = wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::SUBGROUP
                | wgpu::Features::SHADER_F16;
            if crate::numerical::device_options::experimental_features_allowed() {
                wanted |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
                    | wgpu::Features::EXPERIMENTAL_RAY_QUERY;
            }
            let required_features = available_features & wanted;
            let timestamp_supported = required_features.contains(wgpu::Features::TIMESTAMP_QUERY);
            let limits = adapter.limits();
            let mut constraints = AdapterConstraints::from_wgpu_limits(&limits);
            constraints.supports_subgroups = available_features.contains(wgpu::Features::SUBGROUP);
            constraints.supports_coopmat =
                available_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
            constraints.supports_rt_cores =
                available_features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
            constraints.warp_size = if info.vendor == 0x1002 { 64 } else { 32 };
            let required_limits = wgpu::Limits {
                max_blas_primitive_count: limits.max_blas_primitive_count,
                max_blas_geometry_count: limits.max_blas_geometry_count,
                max_tlas_instance_count: limits.max_tlas_instance_count,
                max_acceleration_structures_per_shader_stage: limits
                    .max_acceleration_structures_per_shader_stage,
                ..wgpu::Limits::default()
            };
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_features,
                    required_limits,
                    experimental_features: if required_features.intersects(
                        wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
                            | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                    ) {
                        unsafe { wgpu::ExperimentalFeatures::enabled() }
                    } else {
                        wgpu::ExperimentalFeatures::disabled()
                    },
                    ..Default::default()
                }))
                .map_err(|error| ForgeError::GpuUnavailable(error.to_string()))?;
            let timestamp_period_ns = if timestamp_supported {
                queue.get_timestamp_period()
            } else {
                0.0
            };
            let topology = if info.device_type == wgpu::DeviceType::IntegratedGpu
                || info.device_type == wgpu::DeviceType::Cpu
            {
                MemoryTopology::Unified { zero_copy: true }
            } else {
                MemoryTopology::Discrete {
                    staging_required: true,
                }
            };
            let memory_class = match topology {
                MemoryTopology::Unified { .. } => "unified",
                MemoryTopology::Discrete { .. } => "discrete",
            }
            .to_string();
            let adapter_id = AdapterIdentity {
                name: info.name,
                vendor: info.vendor,
                device: info.device,
                device_type: format!("{:?}", info.device_type),
                backend: format!("{:?}", info.backend),
                driver: info.driver,
                driver_info: info.driver_info,
            };
            let profile = HardwareProfile {
                adapter: adapter_id.clone(),
                constraints,
                memory_class,
                supports_timestamp_query: timestamp_supported,
                max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
                min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
            };
            let allocator = QualiaSlabAllocator::new(topology, capacity_bytes);
            let slab = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("forge-slab-coopmat"),
                size: capacity_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let out_slab = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("forge-out-slab-coopmat"),
                size: capacity_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let weight_slab = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("forge-weight-slab-coopmat"),
                size: capacity_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let timestamp_resources = timestamp_supported.then(|| TimestampResources::new(&device));
            return Ok(Self {
                #[cfg(not(target_arch = "wasm32"))]
                provider_lease: etalon_resources::device_lease::DeviceLease::new(&device, &queue)
                    .map_err(ForgeError::GpuUnavailable)?,
                #[cfg(not(target_arch = "wasm32"))]
                provider_caps: GpuAdapterCaps::from_adapter(&adapter),
                device,
                queue,
                adapter: adapter_id,
                constraints,
                profile,
                allocator,
                slab,
                out_slab,
                weight_slab,
                weight_cursor: 0,
                timestamp_supported,
                timestamp_period_ns,
                timestamp_resources,
                pipeline_cache: RefCell::new(HashMap::new()),
            });
        }

        // Step 3: No coopmat adapter found — fall back to the default backend.
        log::info!("forge|coopmat_adapter|none_found|falling_back_to_default");
        Self::new(capacity_bytes)
    }

    /// Build a forge context on an **already-existing** `wgpu::Device` + `Queue` (e.g. the
    /// process-wide [`etalon_resources::device_caps::shared_gpu`]) instead of requesting a *second* adapter
    /// and device the way [`Self::new`] does. wgpu `Device`/`Queue` are cheap `Arc` clones, so the
    /// forge then runs on the **same** device that owns the resident LLM weights + KV cache — the
    /// device-unification keystone for running decode on the forge (LLM-on-forge plan, Phase 1a).
    ///
    /// Adapter identity / constraints / hardware profile are reconstructed from the **live**
    /// `device.limits()` + `device.features()` plus the caller's [`GpuAdapterCaps`] snapshot,
    /// because the original `wgpu::Adapter` is consumed at shared-gpu init and not retained.
    ///
    /// Honest boundary: this inherits the *host* device's negotiated features and limits verbatim.
    /// In particular, if the shared device was created without the ray-tracing acceleration-structure
    /// limits raised (as `shared_gpu` currently does), RT-core Neighbor cannot create BLAS/TLAS on
    /// this context even when `supports_rt_cores` is true — `from_device` does not silently widen the
    /// host device. The decode path (matmul/elementwise/reduce) needs none of that.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        caps: &GpuAdapterCaps,
        capacity_bytes: usize,
    ) -> Result<Self, ForgeError> {
        if !etalon_resources::provider_scope::admits_device(&device, &queue) {
            return Err(ForgeError::GpuUnavailable(
                "device was not admitted during preparation".into(),
            ));
        }
        match device.owns_queue(&queue) {
            Some(true) => {}
            Some(false) => {
                return Err(ForgeError::GpuValidation(
                    "queue belongs to a different device".into(),
                ));
            }
            None => {
                return Err(ForgeError::GpuValidation(
                    "device/queue ownership cannot be verified".into(),
                ));
            }
        }
        if device.is_valid_state() == Some(false) {
            return Err(ForgeError::DeviceLost("device is no longer live".into()));
        }
        let features = device.features();
        let limits = device.limits();
        let timestamp_supported = features.contains(wgpu::Features::TIMESTAMP_QUERY);

        // Mirror `new()`'s capability derivation, but from the live device + caps snapshot.
        let mut constraints = AdapterConstraints::from_wgpu_limits(&limits);
        constraints.supports_subgroups = features.contains(wgpu::Features::SUBGROUP);
        constraints.supports_coopmat =
            features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        constraints.supports_rt_cores = features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        // AMD wavefronts are 64-wide; others 32. Vendor id 0x1002 = AMD.
        constraints.warp_size = if caps.vendor == 0x1002 { 64 } else { 32 };

        let topology = if matches!(
            caps.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
        ) {
            MemoryTopology::Unified { zero_copy: true }
        } else {
            MemoryTopology::Discrete {
                staging_required: true,
            }
        };
        let memory_class = match topology {
            MemoryTopology::Unified { .. } => "unified",
            MemoryTopology::Discrete { .. } => "discrete",
        }
        .to_string();

        let adapter = AdapterIdentity {
            name: caps.name.clone(),
            vendor: caps.vendor,
            device: caps.device,
            device_type: format!("{:?}", caps.device_type),
            backend: format!("{:?}", caps.backend),
            driver: caps.driver.clone(),
            driver_info: caps.driver_info.clone(),
        };
        let timestamp_period_ns = if timestamp_supported {
            queue.get_timestamp_period()
        } else {
            0.0
        };
        let profile = HardwareProfile {
            adapter: adapter.clone(),
            constraints,
            memory_class,
            supports_timestamp_query: timestamp_supported,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
        };

        let allocator = QualiaSlabAllocator::new(topology, capacity_bytes);
        let slab = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forge-slab"),
            size: capacity_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_slab = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forge-out-slab"),
            size: capacity_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let weight_slab = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forge-weight-slab"),
            size: capacity_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timestamp_resources = timestamp_supported.then(|| TimestampResources::new(&device));

        Ok(Self {
            provider_lease: etalon_resources::device_lease::DeviceLease::new(&device, &queue)
                .map_err(ForgeError::GpuUnavailable)?,
            provider_caps: caps.clone(),
            device,
            queue,
            adapter,
            constraints,
            profile,
            allocator,
            slab,
            out_slab,
            weight_slab,
            weight_cursor: 0,
            timestamp_supported,
            timestamp_period_ns,
            timestamp_resources,
            pipeline_cache: RefCell::new(HashMap::new()),
        })
    }
}

impl OracleContext for WgpuComputeContext {
    fn allocate_and_write(
        &mut self,
        data: &[u8],
        binding: u32,
        group: u32,
        usage: BindingUsage,
    ) -> Result<BufferView, ForgeError> {
        WgpuComputeContext::allocate_and_write(self, data, binding, group, usage)
    }

    fn allocate_transient(
        &mut self,
        size_bytes: usize,
        binding: u32,
        group: u32,
        usage: BindingUsage,
    ) -> Result<BufferView, ForgeError> {
        WgpuComputeContext::allocate_transient(self, size_bytes, binding, group, usage)
    }

    fn read_buffer_f32(&self, view: &BufferView) -> Result<Vec<f32>, ForgeError> {
        WgpuComputeContext::read_buffer_f32(self, view)
    }

    fn clear_transient_allocations(&mut self) {
        WgpuComputeContext::clear_transient_allocations(self);
    }

    fn adapter(&self) -> &AdapterIdentity {
        &self.adapter
    }

    fn constraints(&self) -> &AdapterConstraints {
        &self.constraints
    }

    fn timestamp_supported(&self) -> bool {
        self.timestamp_supported
    }

    /// Emit the kernel's WGSL, compile it, then run the warmup + timed-sample
    /// dispatch loop — byte-for-byte the loop the wgpu oracle evaluators ran inline
    /// (warmups untimed, then `samples` timed dispatches via [`QualiaCompute::dispatch`]).
    fn run_kernel(
        &mut self,
        kernel: &KernelSpec,
        schedule: &Schedule,
        buffers: &[BufferView],
        element_count: usize,
        warmups: usize,
        samples: usize,
    ) -> Result<Vec<u64>, ForgeError> {
        let generated = emit_shader(kernel, *schedule, TargetBackend::Wgsl)?;
        let pipeline = WgpuPipeline::compile(self, &generated.source, &kernel.entry_point)?;

        for _ in 0..warmups {
            pipeline.dispatch(buffers, schedule, element_count)?;
        }
        let mut timing_samples = Vec::with_capacity(samples);
        for _ in 0..samples {
            timing_samples.push(pipeline.dispatch(buffers, schedule, element_count)?);
        }
        Ok(timing_samples)
    }
}

/// One recorded graph node, ready for [`WgpuComputeContext::submit_graph`] to play into a
/// shared command encoder: the compiled pipeline, its bind group, the workgroup count, and
/// the optional GPU→GPU hand-off copy (`src` in the read_write slab → `dst` in the read
/// slab) emitted after the node's compute pass. The pipeline and bind group are owned so the
/// executor can build them all up front and submit the whole graph in one go.
pub struct GraphPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
    pub workgroups: u32,
    pub copy: Option<(BufferView, BufferView)>,
}

pub struct WgpuPipeline<'a> {
    context: &'a WgpuComputeContext,
    pipeline: wgpu::ComputePipeline,
}
