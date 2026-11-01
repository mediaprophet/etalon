use super::{lifecycle, response, MAX_HEIGHT, MAX_WIDTH, SPECTRAL_STATUS};
use js_sys::Uint8Array;
use serde_json::json;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn camera_ingest_rgba(width: u32, height: u32, pixels: Uint8Array) -> String {
    response((|| {
        if width == 0 || height == 0 || width > MAX_WIDTH || height > MAX_HEIGHT {
            return Err("Camera frame dimensions must be within 1..640 by 1..480".into());
        }
        let bytes = width * height * 4;
        if pixels.length() != bytes {
            return Err("Camera RGBA byte length does not match dimensions".into());
        }
        lifecycle::live_stream()?;
        let mut scratch = [0u8; 4096];
        let mut sum = [0u64; 4];
        let mut min = [255u8; 4];
        let mut max = [0u8; 4];
        let mut hash = 0xcbf29ce484222325u64;
        let mut offset = 0;
        while offset < bytes {
            let length = (bytes - offset).min(scratch.len() as u32);
            pixels
                .subarray(offset, offset + length)
                .copy_to(&mut scratch[..length as usize]);
            for (index, value) in scratch[..length as usize].iter().copied().enumerate() {
                let channel = (offset as usize + index) % 4;
                sum[channel] += u64::from(value);
                min[channel] = min[channel].min(value);
                max[channel] = max[channel].max(value);
                hash = (hash ^ u64::from(value)).wrapping_mul(0x100000001b3);
            }
            offset += length;
        }
        let count = f64::from(width * height);
        Ok(
            json!({"ok":true,"schema":"etalon.camera-frame.v1","width":width,"height":height,"bytes_ingested":bytes,
            "format":"RGBA8","checksum_fnv1a64":format!("{hash:016x}"),"channel_min":min,"channel_max":max,
            "channel_mean":sum.map(|v|v as f64/count),"processing":"Rust/WASM CPU; fixed 4096-byte scratch",
            "pixel_source":"caller-supplied canvas RGBA; stream live at ingestion; pixel provenance is not attested",
            "spectral_classification":SPECTRAL_STATUS,"thermal_radiometry_verified":false,"gpu_qualification":false,
            "automatic_upload":false,"automatic_save":false}),
        )
    })())
}
