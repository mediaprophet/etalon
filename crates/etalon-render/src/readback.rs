//! Native caller-buffered offscreen mapping on the embedding's shared device.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_rgba8_into(
    device: &etalon::Device,
    queue: &etalon::Queue,
    texture: &etalon::Texture,
    staging: &etalon::Buffer,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    out: &mut [u8],
) -> Result<usize, String> {
    let need = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or("RGBA8 output size overflow")?;
    if out.len() < need {
        return Err(format!(
            "RGBA8 output buffer too small: need {need}, got {}",
            out.len()
        ));
    }
    let tight = (width as u64).checked_mul(4).ok_or("RGBA8 row overflow")?;
    if u64::from(bytes_per_row) < tight
        || u64::from(bytes_per_row)
            .checked_mul(u64::from(height))
            .filter(|size| *size <= staging.size())
            .is_none()
    {
        return Err("RGBA8 staging layout invalid".into());
    }
    let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
        label: Some("qualia-render-readback-encoder"),
    });
    encoder.copy_texture_to_buffer(
        etalon::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        etalon::TexelCopyBufferInfo {
            buffer: staging,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        etalon::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        },
    );
    queue
        .submit(std::iter::once(encoder.finish()))
        .ok_or_else(|| {
            "RGBA8 readback: native queue rejected the command submission".to_string()
        })?;

    let mut pending = etalon_wasm::readback::PendingMap::begin(
        staging.slice(..),
        etalon_wasm::readback::ready_worker(false),
    )
    .map_err(|error| format!("RGBA8 map admission: {error:?}"))?;
    device
        .poll(etalon::PollType::wait_indefinitely())
        .map_err(|error| format!("RGBA8 device poll: {error}"))?;
    if pending.try_complete() != Some(true) {
        return Err("RGBA8 map failed or completion was cancelled".into());
    }
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .map_err(|error| format!("RGBA8 mapped range: {error}"))?;
    let tight_row = width as usize * 4;
    let padded_row = bytes_per_row as usize;
    for row in 0..height as usize {
        let src = &mapped[row * padded_row..row * padded_row + tight_row];
        let dst = &mut out[row * tight_row..(row + 1) * tight_row];
        dst.copy_from_slice(src);
    }
    drop(mapped);
    drop(pending);
    Ok(need)
}
