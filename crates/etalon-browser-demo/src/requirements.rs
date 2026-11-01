#[cfg(target_arch = "wasm32")]
use etalon::{Limits, TextureFormat};
#[cfg(not(target_arch = "wasm32"))]
use etalon_types::{Limits, TextureFormat};

pub(crate) fn required_limits(adapter: &Limits) -> Limits {
    Limits {
        max_texture_dimension_1d: 1,
        max_texture_dimension_2d: 64,
        max_texture_dimension_3d: 1,
        max_texture_array_layers: 1,
        max_bind_groups: 1,
        max_bind_groups_plus_vertex_buffers: 1,
        max_bindings_per_bind_group: 6,
        max_dynamic_uniform_buffers_per_pipeline_layout: 0,
        max_dynamic_storage_buffers_per_pipeline_layout: 0,
        max_sampled_textures_per_shader_stage: 0,
        max_samplers_per_shader_stage: 0,
        max_storage_buffers_per_shader_stage: 4,
        max_storage_buffers_in_vertex_stage: 0,
        max_storage_buffers_in_fragment_stage: 0,
        max_storage_textures_per_shader_stage: 0,
        max_storage_textures_in_vertex_stage: 0,
        max_storage_textures_in_fragment_stage: 0,
        max_uniform_buffers_per_shader_stage: 1,
        max_uniform_buffer_binding_size: 96,
        max_storage_buffer_binding_size: 32768,
        max_vertex_buffers: 0,
        max_buffer_size: 32768,
        max_vertex_attributes: 0,
        max_vertex_buffer_array_stride: 0,
        max_inter_stage_shader_variables: 0,
        max_color_attachments: 1,
        max_color_attachment_bytes_per_sample: TextureFormat::Rgba8Unorm
            .target_pixel_byte_cost()
            .expect("RGBA8 is a render target format"),
        max_compute_workgroup_storage_size: 12288,
        max_compute_invocations_per_workgroup: 64,
        max_compute_workgroup_size_x: 64,
        max_compute_workgroup_size_y: 1,
        max_compute_workgroup_size_z: 1,
        max_compute_workgroups_per_dimension: 17,
        max_buffers_and_acceleration_structures_per_shader_stage: 6,
        ..Limits::downlevel_webgl2_defaults().using_alignment(adapter.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_capability_limits_do_not_require_unused_desktop_resources() {
        let mut adapter = Limits::downlevel_defaults();
        adapter.max_storage_buffers_per_shader_stage = 4;
        adapter.max_storage_buffers_in_vertex_stage = 0;
        adapter.max_storage_buffers_in_fragment_stage = 0;
        adapter.max_compute_invocations_per_workgroup = 64;
        adapter.max_compute_workgroup_size_x = 64;
        adapter.max_compute_workgroup_storage_size = 12288;
        adapter.min_uniform_buffer_offset_alignment = 512;
        adapter.min_storage_buffer_offset_alignment = 512;
        let required = required_limits(&adapter);
        assert!(required.check_limits(&adapter));
        assert_eq!(required.min_storage_buffer_offset_alignment, 512);
    }

    #[test]
    fn explicit_zero_combined_binding_limit_is_rejected() {
        let mut adapter = Limits::downlevel_defaults();
        adapter.max_bind_groups_plus_vertex_buffers = 0;
        let required = required_limits(&adapter);
        assert_eq!(required.max_bind_groups_plus_vertex_buffers, 1);
        assert_eq!(required.max_bind_groups, 1);
        assert!(!required.check_limits(&adapter));
        adapter.max_bind_groups = 0;
        assert!(!required.check_limits(&adapter));
    }

    #[test]
    fn declared_render_limit_covers_rgba8unorm_pixel_cost() {
        let adapter = Limits::downlevel_defaults();
        let required = required_limits(&adapter);
        assert_eq!(required.max_bind_groups_plus_vertex_buffers, 1);
        assert_eq!(required.max_color_attachment_bytes_per_sample, 8);
        let mut insufficient = adapter;
        insufficient.max_color_attachment_bytes_per_sample = 4;
        assert!(!required.check_limits(&insufficient));
    }

    #[test]
    fn insufficient_compute_or_binding_limits_are_rejected() {
        let mut adapter = Limits::downlevel_defaults();
        adapter.max_compute_invocations_per_workgroup = 32;
        assert!(!required_limits(&adapter).check_limits(&adapter));
        adapter = Limits::downlevel_defaults();
        adapter.max_storage_buffer_binding_size = 1024;
        assert!(!required_limits(&adapter).check_limits(&adapter));
    }
}
