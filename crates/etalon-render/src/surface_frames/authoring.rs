//! Deterministic tileable surface authoring. Callers own all public output.
//! This cold builder creates colour, tangent normal and packed ORM maps;
//! mip generation, transfer functions and texture residency are separate.

#[derive(Debug, Clone, Copy)]
pub struct SurfaceRecipe {
    pub seed: u32,
    /// Multiplicative sRGB tint; normally white when vertex colours supply paint.
    pub tint: [f32; 3],
    pub contrast: f32,
    pub relief: f32,
    /// Integer periods make the banding seamless across repeated UVs.
    pub bands: [u32; 2],
    pub band_strength: f32,
    pub roughness: f32,
    pub metallic: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceAuthoringError {
    InvalidRecipe,
    InvalidSize,
    OutputTooSmall,
    InvalidImage,
    Encode,
}

fn lattice(x: u32, y: u32, period: u32, seed: u32) -> f32 {
    let mut v =
        (x % period).wrapping_mul(0x9e37_79b9) ^ (y % period).wrapping_mul(0x85eb_ca6b) ^ seed;
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    (v & 0xffff) as f32 / 65535.0
}

fn noise(u: f32, v: f32, period: u32, seed: u32) -> f32 {
    let x = u.rem_euclid(1.0) * period as f32;
    let y = v.rem_euclid(1.0) * period as f32;
    let ix = x.floor() as u32;
    let iy = y.floor() as u32;
    let sx = x - x.floor();
    let sy = y - y.floor();
    let sx = sx * sx * (3.0 - 2.0 * sx);
    let sy = sy * sy * (3.0 - 2.0 * sy);
    let a = lattice(ix, iy, period, seed);
    let b = lattice(ix + 1, iy, period, seed);
    let c = lattice(ix, iy + 1, period, seed);
    let d = lattice(ix + 1, iy + 1, period, seed);
    (a + (b - a) * sx) * (1.0 - sy) + (c + (d - c) * sx) * sy
}

fn height(r: SurfaceRecipe, u: f32, v: f32) -> f32 {
    let coarse = noise(u, v, 8, r.seed);
    let fine = noise(u, v, 32, r.seed ^ 0x51ed_270b);
    let wave = (std::f32::consts::TAU * (u * r.bands[0] as f32 + v * r.bands[1] as f32)
        + (coarse - 0.5) * 0.7)
        .sin();
    (coarse * 0.55 + fine * 0.45 - 0.5) * (1.0 - r.band_strength) + wave * 0.5 * r.band_strength
}

fn byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// At most 12 MiB of map output (three 1024² RGBA8 images). Validate every
/// recipe and buffer before writing, so admission failures leave output intact.
pub fn generate_surface_maps_into(
    r: SurfaceRecipe,
    side: u32,
    colour: &mut [u8],
    normal: &mut [u8],
    orm: &mut [u8],
) -> Result<usize, SurfaceAuthoringError> {
    if !(8..=1024).contains(&side) {
        return Err(SurfaceAuthoringError::InvalidSize);
    }
    if r.tint
        .iter()
        .chain(
            [
                r.contrast,
                r.relief,
                r.band_strength,
                r.roughness,
                r.metallic,
            ]
            .iter(),
        )
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || r.bands.iter().any(|v| *v > 128)
    {
        return Err(SurfaceAuthoringError::InvalidRecipe);
    }
    let bytes = side as usize * side as usize * 4;
    if colour.len() < bytes || normal.len() < bytes || orm.len() < bytes {
        return Err(SurfaceAuthoringError::OutputTooSmall);
    }
    let step = 1.0 / side as f32;
    for y in 0..side {
        for x in 0..side {
            let u = x as f32 * step;
            let v = y as f32 * step;
            let h = height(r, u, v);
            let dx = (height(r, u + step, v) - height(r, u - step, v)) * r.relief / (2.0 * step);
            let dy = (height(r, u, v + step) - height(r, u, v - step)) * r.relief / (2.0 * step);
            let inv = (dx * dx + dy * dy + 1.0).sqrt().recip();
            let i = ((y * side + x) * 4) as usize;
            for c in 0..3 {
                colour[i + c] = byte(r.tint[c] * (1.0 - r.contrast * 0.5 + h * r.contrast));
            }
            colour[i + 3] = 255;
            normal[i..i + 4].copy_from_slice(&[
                byte(0.5 - dx * inv * 0.5),
                byte(0.5 - dy * inv * 0.5),
                byte(0.5 + inv * 0.5),
                255,
            ]);
            orm[i..i + 4].copy_from_slice(&[
                255,
                byte((r.roughness + h * 0.06).clamp(0.0, 1.0)),
                byte(r.metallic),
                255,
            ]);
        }
    }
    Ok(bytes)
}

/// Ordinary UV01 source coordinates in metres before scene normalisation.
pub fn planar_uvs_into(
    positions: &[[f32; 3]],
    axes: [usize; 2],
    metres_per_repeat: f32,
    out: &mut [[f32; 2]],
) -> Result<usize, SurfaceAuthoringError> {
    if axes[0] > 2
        || axes[1] > 2
        || axes[0] == axes[1]
        || !metres_per_repeat.is_finite()
        || metres_per_repeat <= 0.0
        || positions.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(SurfaceAuthoringError::InvalidRecipe);
    }
    if out.len() < positions.len() {
        return Err(SurfaceAuthoringError::OutputTooSmall);
    }
    for (p, uv) in positions.iter().zip(out.iter_mut()) {
        *uv = [
            p[axes[0]] / metres_per_repeat,
            p[axes[1]] / metres_per_repeat,
        ];
    }
    Ok(positions.len())
}

fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xedb8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn write_png_chunk(
    out: &mut [u8],
    cursor: &mut usize,
    chunk_type: [u8; 4],
    payload: &[u8],
) -> Result<(), SurfaceAuthoringError> {
    let needed = 12usize
        .checked_add(payload.len())
        .ok_or(SurfaceAuthoringError::Encode)?;
    if *cursor + needed > out.len() {
        return Err(SurfaceAuthoringError::Encode);
    }
    let len = u32::try_from(payload.len()).map_err(|_| SurfaceAuthoringError::Encode)?;
    out[*cursor..*cursor + 4].copy_from_slice(&len.to_be_bytes());
    out[*cursor + 4..*cursor + 8].copy_from_slice(&chunk_type);
    out[*cursor + 8..*cursor + 8 + payload.len()].copy_from_slice(payload);
    let crc = png_crc32(&out[*cursor + 4..*cursor + 8 + payload.len()]);
    out[*cursor + 8 + payload.len()..*cursor + 12 + payload.len()]
        .copy_from_slice(&crc.to_be_bytes());
    *cursor += needed;
    Ok(())
}

/// Encode original RGBA8 artwork as standard PNG into a caller-owned buffer.
pub fn encode_surface_png_into(
    side: u32,
    rgba: &[u8],
    out: &mut [u8],
) -> Result<usize, SurfaceAuthoringError> {
    if !(8..=1024).contains(&side) || rgba.len() != side as usize * side as usize * 4 {
        return Err(SurfaceAuthoringError::InvalidImage);
    }
    // Worst-case deflate plus row filters/container chunks; fail before writing.
    if out.len() < rgba.len() + side as usize + 4096 {
        return Err(SurfaceAuthoringError::OutputTooSmall);
    }

    let mut pos = 0usize;
    let sig = [0x89u8, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    out[pos..pos + 8].copy_from_slice(&sig);
    pos += 8;

    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&side.to_be_bytes());
    ihdr[4..8].copy_from_slice(&side.to_be_bytes());
    ihdr[8] = 8; // bit depth 8
    ihdr[9] = 6; // color type RGBA
    ihdr[10] = 0; // compression method 0
    ihdr[11] = 0; // filter method 0
    ihdr[12] = 0; // interlace method 0
    write_png_chunk(out, &mut pos, *b"IHDR", &ihdr)?;

    // Build IDAT chunk using zlib stored (uncompressed) blocks in-place.
    let idat_start = pos;
    let payload_start = idat_start + 8;
    let mut zpos = payload_start;
    // Zlib header: CMF = 0x78 (Deflate, 32K window), FLG = 0x01 (check bits so 0x7801 % 31 == 0)
    out[zpos] = 0x78;
    out[zpos + 1] = 0x01;
    zpos += 2;

    let row_bytes = side as usize * 4;
    let total_raw = side as usize * (row_bytes + 1);
    let mut adler_a = 1u32;
    let mut adler_b = 0u32;

    let mut block_header_pos = zpos;
    let mut block_len = 0usize;
    let mut remaining_raw = total_raw;

    let start_new_block = |buf: &mut [u8], zcursor: &mut usize, rem: usize| -> (usize, usize) {
        let chunk_cap = rem.min(65535);
        let bfinal = u8::from(rem <= 65535);
        let hdr_pos = *zcursor;
        buf[hdr_pos] = bfinal;
        let len16 = chunk_cap as u16;
        let nlen16 = !len16;
        buf[hdr_pos + 1..hdr_pos + 3].copy_from_slice(&len16.to_le_bytes());
        buf[hdr_pos + 3..hdr_pos + 5].copy_from_slice(&nlen16.to_le_bytes());
        *zcursor += 5;
        (hdr_pos, chunk_cap)
    };

    let (first_hdr, mut block_cap) = start_new_block(out, &mut zpos, remaining_raw);
    let _ = first_hdr;

    let mut push_byte = |b: u8,
                         buf: &mut [u8],
                         zcursor: &mut usize,
                         blk_len: &mut usize,
                         blk_cap: &mut usize,
                         rem: &mut usize| {
        if *blk_len == *blk_cap {
            let (_, new_cap) = start_new_block(buf, zcursor, *rem);
            *blk_len = 0;
            *blk_cap = new_cap;
        }
        buf[*zcursor] = b;
        *zcursor += 1;
        *blk_len += 1;
        *rem -= 1;
        adler_a = (adler_a + b as u32) % 65_521;
        adler_b = (adler_b + adler_a) % 65_521;
    };
    let _ = &mut block_header_pos;

    for y in 0..side as usize {
        // Filter type 0 (None)
        push_byte(
            0,
            out,
            &mut zpos,
            &mut block_len,
            &mut block_cap,
            &mut remaining_raw,
        );
        let row = &rgba[y * row_bytes..(y + 1) * row_bytes];
        for &px in row {
            push_byte(
                px,
                out,
                &mut zpos,
                &mut block_len,
                &mut block_cap,
                &mut remaining_raw,
            );
        }
    }

    let adler = (adler_b << 16) | adler_a;
    out[zpos..zpos + 4].copy_from_slice(&adler.to_be_bytes());
    zpos += 4;

    let idat_payload_len = zpos - payload_start;
    let idat_len_u32 =
        u32::try_from(idat_payload_len).map_err(|_| SurfaceAuthoringError::Encode)?;
    out[idat_start..idat_start + 4].copy_from_slice(&idat_len_u32.to_be_bytes());
    out[idat_start + 4..idat_start + 8].copy_from_slice(b"IDAT");
    let idat_crc = png_crc32(&out[idat_start + 4..zpos]);
    out[zpos..zpos + 4].copy_from_slice(&idat_crc.to_be_bytes());
    pos = zpos + 4;

    write_png_chunk(out, &mut pos, *b"IEND", &[])?;
    Ok(pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn recipe() -> SurfaceRecipe {
        SurfaceRecipe {
            seed: 7,
            tint: [1.0; 3],
            contrast: 0.2,
            relief: 0.002,
            bands: [12, 1],
            band_strength: 0.7,
            roughness: 0.8,
            metallic: 0.0,
        }
    }
    #[test]
    fn repeating_surface_is_continuous_and_buffer_rejection_is_atomic() {
        for v in [0.0, 0.2, 0.63, 1.0] {
            assert!((height(recipe(), 0.0, v) - height(recipe(), 1.0, v)).abs() < 1e-5);
        }
        let mut a = [9; 256];
        let mut b = [8; 256];
        let mut c = [7; 255];
        assert_eq!(
            generate_surface_maps_into(recipe(), 8, &mut a, &mut b, &mut c),
            Err(SurfaceAuthoringError::OutputTooSmall)
        );
        assert_eq!(a, [9; 256]);
        assert_eq!(b, [8; 256]);
        assert_eq!(c, [7; 255]);
    }
    #[test]
    fn png_encoding_produces_valid_png_chunks_and_stored_scanlines() {
        let mut a = [0; 256];
        let mut b = [0; 256];
        let mut c = [0; 256];
        generate_surface_maps_into(recipe(), 8, &mut a, &mut b, &mut c).unwrap();
        let mut encoded = [0; 8192];
        let len = encode_surface_png_into(8, &b, &mut encoded).unwrap();
        assert_eq!(&encoded[0..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&encoded[12..16], b"IHDR");
        assert_eq!(&encoded[37..41], b"IDAT");
        assert_eq!(&encoded[len - 8..len - 4], b"IEND");
        // Verify first stored scanline after 2-byte zlib header + 5-byte deflate stored header + 1-byte filter(0)
        let first_row_start = 41 + 2 + 5 + 1;
        assert_eq!(&encoded[first_row_start..first_row_start + 32], &b[0..32]);
    }
}
