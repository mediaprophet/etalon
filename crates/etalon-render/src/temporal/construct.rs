use super::*;

impl<R> TemporalResolveGpu<R> {
    /// Construct the helper if its bounded history targets can be admitted.
    ///
    /// The helper is optional by design. A device that cannot admit the extra
    /// history targets keeps the existing renderer path instead of failing the
    /// whole viewport.
    pub fn try_new(
        device: &etalon::Device,
        width: u32,
        height: u32,
        scene_format: etalon::TextureFormat,
        output_format: etalon::TextureFormat,
        reserve: impl FnOnce(u64) -> Option<R>,
        exposure_scale: fn(f32) -> Option<f32>,
    ) -> Option<Self> {
        let width = width.max(1);
        let height = height.max(1);
        let reservation = reserve(target_bytes(width, height, scene_format)?)?;

        let texture_usage = etalon::TextureUsages::RENDER_ATTACHMENT
            | etalon::TextureUsages::TEXTURE_BINDING
            | etalon::TextureUsages::COPY_SRC;
        let history_texture = create_texture(
            device,
            "qualia-temporal-history-colour",
            width,
            height,
            scene_format,
            etalon::TextureUsages::TEXTURE_BINDING | etalon::TextureUsages::COPY_DST,
        );
        let history_view = history_texture.create_view(&etalon::TextureViewDescriptor::default());
        let resolved_texture = create_texture(
            device,
            "qualia-temporal-resolved-colour",
            width,
            height,
            scene_format,
            texture_usage,
        );
        let resolved_view = resolved_texture.create_view(&etalon::TextureViewDescriptor::default());
        let linear_depth_texture = create_texture(
            device,
            "qualia-temporal-current-linear-depth",
            width,
            height,
            LINEAR_DEPTH_FORMAT,
            etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TEXTURE_BINDING,
        );
        let linear_depth_view =
            linear_depth_texture.create_view(&etalon::TextureViewDescriptor::default());
        let history_depth_texture = create_texture(
            device,
            "qualia-temporal-history-linear-depth",
            width,
            height,
            DEPTH_HISTORY_FORMAT,
            etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TEXTURE_BINDING,
        );
        let history_depth_view =
            history_depth_texture.create_view(&etalon::TextureViewDescriptor::default());

        let bind_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("qualia-temporal-resolve-layout"),
            entries: &[
                float_texture_entry(0),
                float_texture_entry(1),
                float_texture_entry(2),
                float_texture_entry(3),
                float_texture_entry(4),
                float_texture_entry(5),
                uniform_entry(6, PARAMS_BYTES),
            ],
        });
        let publish_depth_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                label: Some("qualia-temporal-depth-publication-layout"),
                entries: &[float_texture_entry(7)],
            });
        let depth_producer_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                label: Some("qualia-temporal-depth-producer-layout"),
                entries: &[
                    depth_texture_entry(0),
                    uniform_entry(1, DEPTH_PRODUCER_PARAMS_BYTES),
                    sampler_entry(2),
                ],
            });
        let output_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("qualia-temporal-output-layout"),
            entries: &[
                float_texture_entry(0),
                uniform_entry(1, OUTPUT_PARAMS_BYTES),
            ],
        });

        let params_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-temporal-resolve-params"),
            size: PARAMS_BYTES,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth_producer_params_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-temporal-depth-producer-params"),
            size: DEPTH_PRODUCER_PARAMS_BYTES,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth_sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: Some("qualia-temporal-depth-sampler"),
            address_mode_u: etalon::AddressMode::ClampToEdge,
            address_mode_v: etalon::AddressMode::ClampToEdge,
            address_mode_w: etalon::AddressMode::ClampToEdge,
            mag_filter: etalon::FilterMode::Nearest,
            min_filter: etalon::FilterMode::Nearest,
            mipmap_filter: etalon::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let output_params_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-temporal-output-params"),
            size: OUTPUT_PARAMS_BYTES,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let resolve_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia-temporal-resolve-shader"),
            source: etalon::ShaderSource::Wgsl(TEMPORAL_RESOLVE_WGSL.into()),
        });
        let output_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia-temporal-output-shader"),
            source: etalon::ShaderSource::Wgsl(TEMPORAL_OUTPUT_WGSL.into()),
        });
        let resolve_pipeline = create_pipeline(
            device,
            "qualia-temporal-resolve-pipeline",
            &resolve_shader,
            &bind_layout,
            "temporal_vs",
            "temporal_fs",
            scene_format,
        );
        let depth_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia-temporal-depth-producer-shader"),
            source: etalon::ShaderSource::Wgsl(TEMPORAL_DEPTH_WGSL.into()),
        });
        let depth_producer_pipeline = create_pipeline(
            device,
            "qualia-temporal-depth-producer-pipeline",
            &depth_shader,
            &depth_producer_layout,
            "temporal_depth_vs",
            "temporal_depth_fs",
            LINEAR_DEPTH_FORMAT,
        );
        let publish_depth_pipeline = create_pipeline(
            device,
            "qualia-temporal-depth-publication-pipeline",
            &resolve_shader,
            &publish_depth_layout,
            "temporal_vs",
            "publish_depth_fs",
            DEPTH_HISTORY_FORMAT,
        );
        let output_pipeline = create_pipeline(
            device,
            "qualia-temporal-final-output-pipeline",
            &output_shader,
            &output_layout,
            "temporal_output_vs",
            "temporal_output_fs",
            output_format,
        );

        Some(Self {
            bind_layout,
            depth_producer_layout,
            publish_depth_layout,
            output_layout,
            resolve_pipeline,
            depth_producer_pipeline,
            publish_depth_pipeline,
            output_pipeline,
            params_buf,
            depth_producer_params_buf,
            depth_sampler,
            output_params_buf,
            history_texture,
            history_view,
            resolved_texture,
            resolved_view,
            _linear_depth_texture: linear_depth_texture,
            linear_depth_view,
            _history_depth_texture: history_depth_texture,
            history_depth_view,
            width,
            height,
            history_valid: false,
            resolved_ready: false,
            submission_state: TemporalSubmissionState::Idle,
            _reservation: reservation,
            exposure_scale,
        })
    }
}
