//! Portable cascaded-sun shadow resources and bounded GPU target ownership.

use crate::shadow_math::SHADOW_RESOLUTION_CANDIDATES;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniform {
    pub light_view_projection: [[[f32; 4]; 4]; SHADOW_CASCADE_COUNT],
    /// enabled, inverse resolution, constant receiver bias, camera split depth
    pub params: [f32; 4],
}

pub const SHADOW_CASCADE_COUNT: usize = 2;

pub struct ShadowTarget<R> {
    pub _textures: [etalon::Texture; SHADOW_CASCADE_COUNT],
    pub views: [etalon::TextureView; SHADOW_CASCADE_COUNT],
    pub _reservation: R,
    pub resolution: u32,
}

impl<R> ShadowTarget<R> {
    /// Try quality tiers from high to low. If even the smallest allocation is refused, return
    /// `None` so the material shader uses its explicit unshadowed fallback.
    pub fn try_new(
        device: &etalon::Device,
        mut reserve: impl FnMut(u64) -> Option<R>,
    ) -> Option<Self> {
        for resolution in SHADOW_RESOLUTION_CANDIDATES {
            let bytes = u64::from(resolution)
                .checked_mul(u64::from(resolution))?
                .checked_mul(4)?
                .checked_mul(SHADOW_CASCADE_COUNT as u64)?;
            let Some(reservation) = reserve(bytes) else {
                continue;
            };
            let textures = std::array::from_fn(|cascade| {
                device.create_texture(&etalon::TextureDescriptor {
                    label: Some(if cascade == 0 {
                        "qualia-sun-shadow-near-depth"
                    } else {
                        "qualia-sun-shadow-far-depth"
                    }),
                    size: etalon::Extent3d {
                        width: resolution,
                        height: resolution,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: etalon::TextureDimension::D2,
                    format: etalon::TextureFormat::Depth32Float,
                    usage: etalon::TextureUsages::RENDER_ATTACHMENT
                        | etalon::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            });
            let views = std::array::from_fn(|cascade| {
                textures[cascade].create_view(&etalon::TextureViewDescriptor::default())
            });
            return Some(Self {
                _textures: textures,
                views,
                _reservation: reservation,
                resolution,
            });
        }
        None
    }
}

/// `compare_sampler`.
pub fn compare_sampler(device: &etalon::Device) -> etalon::Sampler {
    device.create_sampler(&etalon::SamplerDescriptor {
        label: Some("qualia-sun-shadow-compare"),
        address_mode_u: etalon::AddressMode::ClampToEdge,
        address_mode_v: etalon::AddressMode::ClampToEdge,
        address_mode_w: etalon::AddressMode::ClampToEdge,
        mag_filter: etalon::FilterMode::Linear,
        min_filter: etalon::FilterMode::Linear,
        mipmap_filter: etalon::MipmapFilterMode::Nearest,
        compare: Some(etalon::CompareFunction::LessEqual),
        ..Default::default()
    })
}

/// `matrix_layout`.
pub fn matrix_layout(device: &etalon::Device) -> etalon::BindGroupLayout {
    device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: Some("qualia-sun-shadow-matrix-layout"),
        entries: &[etalon::BindGroupLayoutEntry {
            binding: 0,
            visibility: etalon::ShaderStages::VERTEX,
            ty: etalon::BindingType::Buffer {
                ty: etalon::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(
                    std::mem::size_of::<ShadowUniform>() as u64
                ),
            },
            count: None,
        }],
    })
}

/// 4-Cascade Cascaded Shadow Map (CSM) GPU uniform block (288 bytes) with practical split distances
/// and smooth inter-cascade blend band parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Csm4ShadowUniform {
    pub light_view_projection: [[[f32; 4]; 4]; crate::shadow_math::CSM_4_CASCADE_COUNT],
    /// Split distances `[z0, z1, z2, z3]` in camera view space.
    pub cascade_splits: [f32; crate::shadow_math::CSM_4_CASCADE_COUNT],
    /// `[enabled, inv_resolution, receiver_bias, blend_band_fraction]`
    pub params: [f32; 4],
}

const _: [(); 288] = [(); std::mem::size_of::<Csm4ShadowUniform>()];

pub fn record_cascades(
    encoder: &mut etalon::CommandEncoder,
    views: &[etalon::TextureView],
    mut draw: impl FnMut(usize, &mut etalon::RenderPass<'_>),
) {
    for (cascade, view) in views.iter().enumerate() {
        let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some(if cascade == 0 {
                "portal-sun-shadow-near-depth-pass"
            } else {
                "portal-sun-shadow-far-depth-pass"
            }),
            color_attachments: &[],
            depth_stencil_attachment: Some(etalon::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(etalon::Operations {
                    load: etalon::LoadOp::Clear(1.0),
                    store: etalon::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        draw(cascade, &mut pass);
    }
}
