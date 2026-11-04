//! Portable cascaded-sun shadow resources and bounded GPU target ownership.

use crate::shadow_math::SHADOW_RESOLUTION_CANDIDATES;
use etalon as wgpu;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniform {
    pub light_view_projection: [[[f32; 4]; 4]; SHADOW_CASCADE_COUNT],
    /// enabled, inverse resolution, constant receiver bias, camera split depth
    pub params: [f32; 4],
}

pub const SHADOW_CASCADE_COUNT: usize = 2;

pub struct ShadowTarget<R> {
    pub _textures: [wgpu::Texture; SHADOW_CASCADE_COUNT],
    pub views: [wgpu::TextureView; SHADOW_CASCADE_COUNT],
    pub _reservation: R,
    pub resolution: u32,
}

impl<R> ShadowTarget<R> {
    /// Try quality tiers from high to low. If even the smallest allocation is refused, return
    /// `None` so the material shader uses its explicit unshadowed fallback.
    pub fn try_new(
        device: &wgpu::Device,
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
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(if cascade == 0 {
                        "qualia-sun-shadow-near-depth"
                    } else {
                        "qualia-sun-shadow-far-depth"
                    }),
                    size: wgpu::Extent3d {
                        width: resolution,
                        height: resolution,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            });
            let views = std::array::from_fn(|cascade| {
                textures[cascade].create_view(&wgpu::TextureViewDescriptor::default())
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
pub fn compare_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("qualia-sun-shadow-compare"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    })
}

/// `matrix_layout`.
pub fn matrix_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("qualia-sun-shadow-matrix-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
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
    encoder: &mut wgpu::CommandEncoder,
    views: &[wgpu::TextureView],
    mut draw: impl FnMut(usize, &mut wgpu::RenderPass<'_>),
) {
    for (cascade, view) in views.iter().enumerate() {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(if cascade == 0 {
                "portal-sun-shadow-near-depth-pass"
            } else {
                "portal-sun-shadow-far-depth-pass"
            }),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
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
