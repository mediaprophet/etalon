//! Portable frame target creation and checked readback row layout.

/// `checked_padded_bytes_per_row`.
pub fn checked_padded_bytes_per_row(width: u32) -> Option<u32> {
    let tight = width.max(1).checked_mul(4)?;
    let align = etalon::COPY_BYTES_PER_ROW_ALIGNMENT;
    tight
        .checked_add(align - 1)
        .map(|value| value / align * align)
}

/// `readback_buffer_bytes`.
pub fn readback_buffer_bytes(width: u32, height: u32) -> Option<u64> {
    u64::from(checked_padded_bytes_per_row(width)?).checked_mul(u64::from(height.max(1)))
}

/// Creates `offscreen-texture`.
pub fn create_offscreen_texture(
    device: &etalon::Device,
    format: etalon::TextureFormat,
    width: u32,
    height: u32,
) -> etalon::Texture {
    device.create_texture(&etalon::TextureDescriptor {
        label: Some("qualia-render-offscreen"),
        size: etalon::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// `padded_bytes_per_row`.
pub fn padded_bytes_per_row(width: u32) -> u32 {
    checked_padded_bytes_per_row(width)
        .unwrap_or(u32::MAX & !(etalon::COPY_BYTES_PER_ROW_ALIGNMENT - 1))
}

/// Creates `readback-buffer`.
pub fn create_readback_buffer(
    device: &etalon::Device,
    padded_bytes_per_row: u32,
    height: u32,
) -> etalon::Buffer {
    device.create_buffer(&etalon::BufferDescriptor {
        label: Some("qualia-render-readback"),
        size: padded_bytes_per_row as u64 * height.max(1) as u64,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

/// Creates `picking-texture`.
pub fn create_picking_texture(
    device: &etalon::Device,
    width: u32,
    height: u32,
) -> (etalon::Texture, etalon::TextureView) {
    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: Some("portal-picking"),
        size: etalon::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::R32Uint,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&etalon::TextureViewDescriptor::default());
    (texture, view)
}

/// `picking_color_target_state`.
pub fn picking_color_target_state() -> etalon::ColorTargetState {
    etalon::ColorTargetState {
        format: etalon::TextureFormat::R32Uint,
        blend: None,
        write_mask: etalon::ColorWrites::ALL,
    }
}

/// Creates `depth-texture`.
pub fn create_depth_texture(
    device: &etalon::Device,
    width: u32,
    height: u32,
) -> (etalon::Texture, etalon::TextureView) {
    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: Some("portal-depth"),
        size: etalon::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Depth32Float,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&etalon::TextureViewDescriptor::default());
    (texture, view)
}

/// `depth_stencil_state`.
pub fn depth_stencil_state(format: etalon::TextureFormat) -> etalon::DepthStencilState {
    etalon::DepthStencilState {
        format,
        depth_write_enabled: Some(true),
        depth_compare: Some(etalon::CompareFunction::Less),
        stencil: etalon::StencilState::default(),
        bias: etalon::DepthBiasState::default(),
    }
}

/// `depth_stencil_state_read_only`.
pub fn depth_stencil_state_read_only() -> etalon::DepthStencilState {
    etalon::DepthStencilState {
        format: etalon::TextureFormat::Depth32Float,
        depth_write_enabled: Some(false),
        depth_compare: Some(etalon::CompareFunction::Less),
        stencil: etalon::StencilState::default(),
        bias: etalon::DepthBiasState::default(),
    }
}

/// `color_target_state`.
pub fn color_target_state(format: etalon::TextureFormat) -> etalon::ColorTargetState {
    etalon::ColorTargetState {
        format,
        blend: Some(etalon::BlendState {
            color: etalon::BlendComponent {
                src_factor: etalon::BlendFactor::SrcAlpha,
                dst_factor: etalon::BlendFactor::OneMinusSrcAlpha,
                operation: etalon::BlendOperation::Add,
            },
            alpha: etalon::BlendComponent {
                src_factor: etalon::BlendFactor::One,
                dst_factor: etalon::BlendFactor::OneMinusSrcAlpha,
                operation: etalon::BlendOperation::Add,
            },
        }),
        write_mask: etalon::ColorWrites::ALL,
    }
}
