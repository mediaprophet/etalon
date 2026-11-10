//! Native wgpu adapter capability reporting.
//!
//! This module is intentionally diagnostic/policy-facing rather than hot-path code:
//! it records what the selected adapter actually exposes so benchmark rows, logs,
//! and future feature negotiation can agree on the same facts.
//!
//! Compiles only with `gpu-runtime` (the feature that pulls in the wgpu crate).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuFeatureCaps {
    pub timestamp_query: bool,
    pub timestamp_query_inside_passes: bool,
    pub pipeline_statistics_query: bool,
    pub pipeline_cache: bool,
    pub shader_f16: bool,
    pub shader_int64: bool,
    pub subgroup: bool,
    pub subgroup_barrier: bool,
    pub cooperative_matrix: bool,
    pub ray_query: bool,
}

impl GpuFeatureCaps {
    /// Builds a caps snapshot from the adapter's `etalon::Features`.
    pub fn from_features(features: etalon::Features) -> Self {
        Self {
            timestamp_query: features.contains(etalon::Features::TIMESTAMP_QUERY),
            timestamp_query_inside_passes: features
                .contains(etalon::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
            pipeline_statistics_query: features
                .contains(etalon::Features::PIPELINE_STATISTICS_QUERY),
            pipeline_cache: features.contains(etalon::Features::PIPELINE_CACHE),
            shader_f16: features.contains(etalon::Features::SHADER_F16),
            shader_int64: features.contains(etalon::Features::SHADER_INT64),
            subgroup: features.contains(etalon::Features::SUBGROUP),
            subgroup_barrier: features.contains(etalon::Features::SUBGROUP_BARRIER),
            cooperative_matrix: features
                .contains(etalon::Features::EXPERIMENTAL_COOPERATIVE_MATRIX),
            ray_query: features.contains(etalon::Features::EXPERIMENTAL_RAY_QUERY),
        }
    }

    /// One-line compact flag string of enabled features.
    pub fn compact_flags(&self) -> String {
        let mut out = String::with_capacity(96);
        push_flag(&mut out, "ts", self.timestamp_query);
        push_flag(&mut out, "ts-pass", self.timestamp_query_inside_passes);
        push_flag(&mut out, "stats", self.pipeline_statistics_query);
        push_flag(&mut out, "cache", self.pipeline_cache);
        push_flag(&mut out, "f16", self.shader_f16);
        push_flag(&mut out, "i64", self.shader_int64);
        push_flag(&mut out, "subgroup", self.subgroup);
        push_flag(&mut out, "subgroup-barrier", self.subgroup_barrier);
        push_flag(&mut out, "coop-matrix", self.cooperative_matrix);
        push_flag(&mut out, "ray-query", self.ray_query);
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuLimitCaps {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

impl GpuLimitCaps {
    /// Builds a caps snapshot from `etalon::Limits`.
    pub fn from_limits(limits: &etalon::Limits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuAdapterCaps {
    pub name: String,
    pub backend: etalon::Backend,
    pub device_type: etalon::DeviceType,
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
    pub driver_info: String,
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
    pub cooperative_matrix_tile_count: usize,
    pub features: GpuFeatureCaps,
    pub limits: GpuLimitCaps,
}

impl GpuAdapterCaps {
    /// Collects adapter info + features + limits into a caps record.
    pub fn from_adapter(adapter: &etalon::Adapter) -> Self {
        let info = adapter.get_info();
        let features = adapter.features();
        let limits = {
            #[cfg(target_arch = "wasm32")]
            {
                let _ = adapter;
                webgpu_minimum_limits()
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                adapter.limits()
            }
        };
        Self {
            name: info.name,
            backend: info.backend,
            device_type: info.device_type,
            vendor: info.vendor,
            device: info.device,
            driver: info.driver,
            driver_info: info.driver_info,
            subgroup_min_size: info.subgroup_min_size,
            subgroup_max_size: info.subgroup_max_size,
            cooperative_matrix_tile_count: adapter.cooperative_matrix_properties().len(),
            features: GpuFeatureCaps::from_features(features),
            limits: GpuLimitCaps::from_limits(&limits),
        }
    }

    /// Anvil ontology workgroup size for this PCI id.
    ///
    /// Etalon injects this as a Naga/WGSL override. Values match
    /// `anvil-gpu::hints_for_pci` (Xe-LP 64, Ampere 256). Anvil remains a
    /// side-by-side crate; this table is the local handshake until etalon
    /// pins `mediaprophet/anvil`.
    pub fn anvil_workgroup_size(&self) -> u32 {
        match (self.vendor as u16, self.device as u16) {
            (0x8086, 0xA788) => 64,  // Xe-LP 32 EU
            (0x8086, 0x5917) => 64,  // UHD 620
            (0x10DE, 0x2531) => 256, // RTX A2000
            (0x106B, 0x0001) => 256, // Apple Silicon unified
            (0x13B5, 0x0001) => 64,  // Mali-class mobile
            _ => {
                if self.is_integrated_gpu() {
                    64
                } else {
                    256
                }
            }
        }
    }

    #[inline]
    /// `true` when the adapter reports an integrated GPU.
    pub fn is_integrated_gpu(&self) -> bool {
        matches!(self.device_type, etalon::DeviceType::IntegratedGpu)
    }

    #[inline]
    /// Static name of the wgpu backend (vulkan/dx12/metal/gl/…).
    pub fn backend_label(&self) -> &'static str {
        match self.backend {
            etalon::Backend::Noop => "noop",
            etalon::Backend::Vulkan => "vulkan",
            etalon::Backend::Metal => "metal",
            etalon::Backend::Dx12 => "dx12",
            etalon::Backend::Gl => "gl",
            etalon::Backend::BrowserWebGpu => "browser-webgpu",
        }
    }

    #[inline]
    /// Static name of the adapter device type.
    pub fn device_type_label(&self) -> &'static str {
        match self.device_type {
            etalon::DeviceType::Other => "other",
            etalon::DeviceType::IntegratedGpu => "integrated",
            etalon::DeviceType::DiscreteGpu => "discrete",
            etalon::DeviceType::VirtualGpu => "virtual",
            etalon::DeviceType::Cpu => "cpu",
        }
    }

    /// One-line human summary of adapter identity and key limits.
    pub fn summary_line(&self) -> String {
        format!(
            "{} | backend={} | type={} | vendor=0x{:04x} | device=0x{:04x} | driver={} {}",
            self.name,
            self.backend_label(),
            self.device_type_label(),
            self.vendor,
            self.device,
            self.driver,
            self.driver_info
        )
    }

    /// One-line summary of LLM-relevant features (subgroups, f16, timestamps…).
    pub fn llm_feature_line(&self) -> String {
        format!(
            "features=[{}] subgroup={}..{} coop_tiles={} max_storage_binding={}MiB max_buffer={}MiB",
            self.features.compact_flags(),
            self.subgroup_min_size,
            self.subgroup_max_size,
            self.cooperative_matrix_tile_count,
            self.limits.max_storage_buffer_binding_size / (1024 * 1024),
            self.limits.max_buffer_size / (1024 * 1024)
        )
    }
}

fn push_flag(out: &mut String, label: &str, enabled: bool) {
    if !out.is_empty() {
        out.push(' ');
    }
    out.push_str(label);
    out.push('=');
    out.push_str(if enabled { "1" } else { "0" });
}

/// WebGPU minimum limits from the spec. On wasm, `Adapter::limits` reads
/// `GPUAdapter.limits`. A null adapter throws `TypeError` there and the
/// canvas stays empty. These minimums never touch that getter. Native code
/// still asks the adapter.
#[cfg(target_arch = "wasm32")]
pub fn webgpu_minimum_limits() -> etalon::Limits {
    etalon::Limits {
        max_texture_dimension_1d: 8192,
        max_texture_dimension_2d: 8192,
        max_texture_dimension_3d: 2048,
        max_texture_array_layers: 256,
        max_bind_groups: 4,
        max_bindings_per_bind_group: 1000,
        max_dynamic_uniform_buffers_per_pipeline_layout: 8,
        max_dynamic_storage_buffers_per_pipeline_layout: 4,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_textures_per_shader_stage: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 64 << 10,
        max_storage_buffer_binding_size: 128 << 20,
        max_vertex_buffers: 8,
        max_buffer_size: 256 << 20,
        max_vertex_attributes: 16,
        max_vertex_buffer_array_stride: 2048,
        min_uniform_buffer_offset_alignment: 256,
        min_storage_buffer_offset_alignment: 256,
        max_inter_stage_shader_variables: 16,
        max_color_attachments: 8,
        max_color_attachment_bytes_per_sample: 32,
        max_compute_workgroup_storage_size: 16384,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_workgroups_per_dimension: 65535,
        ..etalon::Limits::downlevel_defaults()
    }
}
