use super::*;

pub(super) fn map_wgt_limits(limits: webgpu_sys::GpuSupportedLimits) -> wgt::Limits {
    let optional_limit = |name: &str, value: u32, fallback: u32| {
        if js_sys::Reflect::has(limits.as_ref(), &JsValue::from_str(name)).unwrap_or(true) {
            value
        } else {
            fallback
        }
    };
    wgt::Limits {
        max_texture_dimension_1d: limits.max_texture_dimension_1d(),
        max_texture_dimension_2d: limits.max_texture_dimension_2d(),
        max_texture_dimension_3d: limits.max_texture_dimension_3d(),
        max_texture_array_layers: limits.max_texture_array_layers(),
        max_bind_groups: limits.max_bind_groups(),
        max_bind_groups_plus_vertex_buffers: optional_limit(
            "maxBindGroupsPlusVertexBuffers",
            limits.max_bind_groups_plus_vertex_buffers(),
            24,
        ),
        max_bindings_per_bind_group: limits.max_bindings_per_bind_group(),
        max_dynamic_uniform_buffers_per_pipeline_layout: limits
            .max_dynamic_uniform_buffers_per_pipeline_layout(),
        max_dynamic_storage_buffers_per_pipeline_layout: limits
            .max_dynamic_storage_buffers_per_pipeline_layout(),
        max_sampled_textures_per_shader_stage: limits.max_sampled_textures_per_shader_stage(),
        max_samplers_per_shader_stage: limits.max_samplers_per_shader_stage(),
        max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage(),
        max_storage_buffers_in_vertex_stage: optional_limit(
            "maxStorageBuffersInVertexStage",
            limits.max_storage_buffers_in_vertex_stage(),
            limits.max_storage_buffers_per_shader_stage(),
        ),
        max_storage_buffers_in_fragment_stage: optional_limit(
            "maxStorageBuffersInFragmentStage",
            limits.max_storage_buffers_in_fragment_stage(),
            limits.max_storage_buffers_per_shader_stage(),
        ),
        max_storage_textures_per_shader_stage: limits.max_storage_textures_per_shader_stage(),
        max_storage_textures_in_vertex_stage: optional_limit(
            "maxStorageTexturesInVertexStage",
            limits.max_storage_textures_in_vertex_stage(),
            limits.max_storage_textures_per_shader_stage(),
        ),
        max_storage_textures_in_fragment_stage: optional_limit(
            "maxStorageTexturesInFragmentStage",
            limits.max_storage_textures_in_fragment_stage(),
            limits.max_storage_textures_per_shader_stage(),
        ),
        max_uniform_buffers_per_shader_stage: limits.max_uniform_buffers_per_shader_stage(),
        max_binding_array_elements_per_shader_stage: 0,
        max_binding_array_sampler_elements_per_shader_stage: 0,
        max_binding_array_acceleration_structure_elements_per_shader_stage: 0,
        max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size() as u64,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size() as u64,
        max_vertex_buffers: limits.max_vertex_buffers(),
        max_buffer_size: limits.max_buffer_size() as u64,
        max_vertex_attributes: limits.max_vertex_attributes(),
        max_vertex_buffer_array_stride: limits.max_vertex_buffer_array_stride(),
        max_inter_stage_shader_variables: limits.max_inter_stage_shader_variables(),
        min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment(),
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment(),
        max_color_attachments: limits.max_color_attachments(),
        max_color_attachment_bytes_per_sample: optional_limit(
            "maxColorAttachmentBytesPerSample",
            limits.max_color_attachment_bytes_per_sample(),
            32,
        ),
        max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size(),
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup(),
        max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x(),
        max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y(),
        max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z(),
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension(),
        max_immediate_size: wgt::Limits::default().max_immediate_size,
        max_non_sampler_bindings: wgt::Limits::default().max_non_sampler_bindings,

        max_task_workgroup_total_count: wgt::Limits::default().max_task_workgroup_total_count,
        max_task_workgroups_per_dimension: wgt::Limits::default().max_task_workgroups_per_dimension,
        max_mesh_workgroup_total_count: wgt::Limits::default().max_mesh_workgroup_total_count,
        max_mesh_workgroups_per_dimension: wgt::Limits::default().max_mesh_workgroups_per_dimension,
        max_task_invocations_per_workgroup: wgt::Limits::default()
            .max_task_invocations_per_workgroup,
        max_task_invocations_per_dimension: wgt::Limits::default()
            .max_task_invocations_per_dimension,
        max_mesh_invocations_per_workgroup: wgt::Limits::default()
            .max_mesh_invocations_per_workgroup,
        max_mesh_invocations_per_dimension: wgt::Limits::default()
            .max_mesh_invocations_per_dimension,
        max_task_payload_size: wgt::Limits::default().max_task_payload_size,
        max_mesh_output_vertices: wgt::Limits::default().max_mesh_output_vertices,
        max_mesh_output_primitives: wgt::Limits::default().max_mesh_output_primitives,
        max_mesh_output_layers: wgt::Limits::default().max_mesh_output_layers,
        max_mesh_multiview_view_count: wgt::Limits::default().max_mesh_multiview_view_count,

        max_blas_primitive_count: wgt::Limits::default().max_blas_primitive_count,
        max_blas_geometry_count: wgt::Limits::default().max_blas_geometry_count,
        max_tlas_instance_count: wgt::Limits::default().max_tlas_instance_count,
        max_acceleration_structures_per_shader_stage: wgt::Limits::default()
            .max_acceleration_structures_per_shader_stage,
        max_buffers_and_acceleration_structures_per_shader_stage: wgt::Limits::default()
            .max_buffers_and_acceleration_structures_per_shader_stage,

        max_multiview_view_count: wgt::Limits::default().max_multiview_view_count,

        max_ray_dispatch_count: wgt::Limits::default().max_ray_dispatch_count,
        max_ray_recursion_depth: wgt::Limits::default().max_ray_recursion_depth,
    }
}

pub(super) fn map_adapter_info(adapter_info: &webgpu_sys::GpuAdapterInfo) -> wgt::AdapterInfo {
    let vendor = adapter_info.vendor();
    let architecture = adapter_info.architecture();
    let device = adapter_info.device();
    let description = adapter_info.description();
    let name = if description.is_empty() {
        [&vendor, &architecture, &device]
            .into_iter()
            .filter(|part| !part.is_empty())
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        description
    };
    let driver_info = if vendor.is_empty() && architecture.is_empty() && device.is_empty() {
        String::new()
    } else {
        format!("WebGPU vendor={vendor}; architecture={architecture}; device={device}")
    };
    wgt::AdapterInfo {
        name,
        vendor: 0,
        device: 0,
        device_type: if adapter_info.is_fallback_adapter() {
            wgt::DeviceType::Cpu
        } else {
            wgt::DeviceType::Other
        },
        device_pci_bus_id: String::new(),
        driver: String::new(),
        driver_info,
        backend: wgt::Backend::BrowserWebGpu,
        subgroup_min_size: adapter_info.subgroup_min_size(),
        subgroup_max_size: adapter_info.subgroup_max_size(),
        transient_saves_memory: None,
        limit_bucket: None,
    }
}
