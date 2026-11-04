//! Persistent material texture residency with generation-correct semantic mip uploads.
use crate::texture_layout::{
    mip_chain_layout, mip_semantic_matches, validate_upload, TextureColorSpace, TextureIdentity,
    TextureUploadError,
};
use crate::texture_mips::{AlphaCoverageDiagnostics, MipGenerator};
use etalon as wgpu;
use std::collections::BTreeMap;
#[derive(Clone, Copy)]
pub enum TextureAllocation {
    TextureResidency,
    UploadStaging,
}
pub struct ResidentTexture<R> {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub _reservation: R,
}

fn alpha_mip_scratch_bytes(width: u32, height: u32) -> Result<u64, TextureUploadError> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(TextureUploadError::ByteSizeOverflow)
}

pub type ResidentTextureMap<R> = BTreeMap<TextureIdentity, ResidentTexture<R>>;
pub fn upload_rgba8<R: Send + 'static>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mip_generator: &MipGenerator,
    resident_textures: &mut ResidentTextureMap<R>,
    source_digest: [u8; 32],
    color_space: TextureColorSpace,
    mip_semantic: crate::texture_mips::TextureMipSemantic,
    width: u32,
    height: u32,
    rgba8: &[u8],
    copy_source: bool,
    mut reserve: impl FnMut(TextureAllocation, u64) -> Option<R>,
) -> Result<Option<AlphaCoverageDiagnostics>, TextureUploadError> {
    if !mip_semantic_matches(color_space, mip_semantic) {
        return Err(TextureUploadError::InvalidMipSemantic);
    }
    let (base_bytes, bytes_per_row) = validate_upload(
        width,
        height,
        device.limits().max_texture_dimension_2d,
        rgba8.len(),
    )?;
    let (mip_level_count, resident_bytes) = mip_chain_layout(width, height)?;
    let key = TextureIdentity {
        source_digest,
        color_space,
        mip_semantic,
    };
    if resident_textures.contains_key(&key) {
        let alpha_scratch_reservation = if mip_semantic.alpha_cutoff().is_some() {
            Some(
                reserve(
                    TextureAllocation::UploadStaging,
                    alpha_mip_scratch_bytes(width, height)?,
                )
                .ok_or(TextureUploadError::UploadBudgetRefused)?,
            )
        } else {
            None
        };
        let diagnostics = mip_semantic
            .alpha_cutoff()
            .map(|cutoff| crate::texture_mips::alpha_coverage_plan(rgba8, width, height, cutoff))
            .map(AlphaCoverageDiagnostics::from);
        drop(alpha_scratch_reservation);
        return Ok(diagnostics);
    }
    let reservation = reserve(TextureAllocation::TextureResidency, resident_bytes)
        .ok_or(TextureUploadError::GpuBudgetRefused)?;
    let alpha_scratch_reservation = if mip_semantic.alpha_cutoff().is_some() {
        Some(
            reserve(
                TextureAllocation::UploadStaging,
                alpha_mip_scratch_bytes(width, height)?,
            )
            .ok_or(TextureUploadError::UploadBudgetRefused)?,
        )
    } else {
        None
    };
    let alpha_plan = mip_semantic
        .alpha_cutoff()
        .map(|cutoff| crate::texture_mips::alpha_coverage_plan(rgba8, width, height, cutoff));
    let diagnostics = alpha_plan.map(AlphaCoverageDiagnostics::from);
    let upload_reservation = reserve(TextureAllocation::UploadStaging, base_bytes)
        .ok_or(TextureUploadError::UploadBudgetRefused)?;
    let format = match color_space {
        TextureColorSpace::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureColorSpace::Linear => wgpu::TextureFormat::Rgba8Unorm,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("qualia-resident-material-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | if copy_source {
                wgpu::TextureUsages::COPY_SRC
            } else {
                wgpu::TextureUsages::empty()
            },
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba8,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    mip_generator.generate(
        device,
        queue,
        &texture,
        mip_level_count,
        mip_semantic,
        alpha_plan.as_ref(),
    );
    drop(alpha_scratch_reservation);
    queue.on_submitted_work_done(move || drop(upload_reservation));
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    resident_textures.insert(
        key,
        ResidentTexture {
            _texture: texture,
            view,
            _reservation: reservation,
        },
    );
    Ok(diagnostics)
}
/// `resident_view`.
pub fn resident_view<'a, R>(
    textures: &'a ResidentTextureMap<R>,
    digest: &[u8; 32],
    color_space: TextureColorSpace,
    mip_semantic: crate::texture_mips::TextureMipSemantic,
) -> Option<&'a wgpu::TextureView> {
    textures
        .get(&TextureIdentity {
            source_digest: *digest,
            color_space,
            mip_semantic,
        })
        .map(|resident| &resident.view)
}
