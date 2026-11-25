//! Observable GLES unpack and storage-emulation counters.
//!
//! Record functions are atomic stores for the queue submit hot path.
//! Snapshot/JSON lives on the portal diagnostic cadence, not per draw.

use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use etalon_sync::atomic::AtomicU64;

const RELAXED: Ordering = Ordering::Relaxed;

static COPY_PBO: AtomicU64 = AtomicU64::new(0);
static COPY_SLICE: AtomicU64 = AtomicU64::new(0);
static COPY_COMPRESSED_PBO: AtomicU64 = AtomicU64::new(0);
static COPY_COMPRESSED_SLICE: AtomicU64 = AtomicU64::new(0);
static EMU_UPLOADS: AtomicU64 = AtomicU64::new(0);
static EMU_SKIPS: AtomicU64 = AtomicU64::new(0);
static EMU_TEXELS: AtomicU64 = AtomicU64::new(0);
static EMU_BYTES: AtomicU64 = AtomicU64::new(0);
static EMU_REMAINDER_ROWS: AtomicU64 = AtomicU64::new(0);

static LAST_COPY_KIND: AtomicU32 = AtomicU32::new(0);
static LAST_COPY_W: AtomicU32 = AtomicU32::new(0);
static LAST_COPY_H: AtomicU32 = AtomicU32::new(0);
static LAST_COPY_ROW: AtomicI32 = AtomicI32::new(0);
static LAST_COPY_IMAGE: AtomicI32 = AtomicI32::new(0);
static LAST_COPY_NEEDED: AtomicU64 = AtomicU64::new(0);
static LAST_COPY_TAKE: AtomicU64 = AtomicU64::new(0);

static LAST_EMU_SLOT: AtomicU32 = AtomicU32::new(0);
static LAST_EMU_OFFSET: AtomicI32 = AtomicI32::new(0);
static LAST_EMU_SIZE: AtomicI32 = AtomicI32::new(0);
static LAST_EMU_TEXELS: AtomicU64 = AtomicU64::new(0);
static LAST_EMU_W: AtomicI32 = AtomicI32::new(0);
static LAST_EMU_H: AtomicI32 = AtomicI32::new(0);
static LAST_EMU_FULL_ROWS: AtomicI32 = AtomicI32::new(0);
static LAST_EMU_REM: AtomicI32 = AtomicI32::new(0);

/// Last CopyBufferToTexture / BindEmulatedStorage parameters observed by the GLES queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlesUnpackDebug {
    pub copy_buffer_to_texture_pbo: u64,
    pub copy_buffer_to_texture_slice: u64,
    pub copy_compressed_pbo: u64,
    pub copy_compressed_slice: u64,
    pub bind_emulated_storage_uploads: u64,
    pub bind_emulated_storage_skips: u64,
    pub bind_emulated_storage_texels: u64,
    pub bind_emulated_storage_bytes: u64,
    pub bind_emulated_storage_remainder_rows: u64,
    pub last_copy_kind: &'static str,
    pub last_copy_width: u32,
    pub last_copy_height: u32,
    pub last_copy_unpack_row_length: i32,
    pub last_copy_unpack_image_height: i32,
    pub last_copy_needed_bytes: u64,
    pub last_copy_take_bytes: u64,
    pub last_emu_slot: u32,
    pub last_emu_offset: i32,
    pub last_emu_size: i32,
    pub last_emu_texels: u64,
    pub last_emu_width: i32,
    pub last_emu_height: i32,
    pub last_emu_full_rows: i32,
    pub last_emu_remainder_texels: i32,
}

fn copy_kind_name(kind: u32) -> &'static str {
    match kind {
        1 => "pbo",
        2 => "slice",
        3 => "compressed_pbo",
        4 => "compressed_slice",
        _ => "none",
    }
}

/// Current GLES unpack/emulation counters.
pub fn snapshot() -> GlesUnpackDebug {
    GlesUnpackDebug {
        copy_buffer_to_texture_pbo: COPY_PBO.load(RELAXED),
        copy_buffer_to_texture_slice: COPY_SLICE.load(RELAXED),
        copy_compressed_pbo: COPY_COMPRESSED_PBO.load(RELAXED),
        copy_compressed_slice: COPY_COMPRESSED_SLICE.load(RELAXED),
        bind_emulated_storage_uploads: EMU_UPLOADS.load(RELAXED),
        bind_emulated_storage_skips: EMU_SKIPS.load(RELAXED),
        bind_emulated_storage_texels: EMU_TEXELS.load(RELAXED),
        bind_emulated_storage_bytes: EMU_BYTES.load(RELAXED),
        bind_emulated_storage_remainder_rows: EMU_REMAINDER_ROWS.load(RELAXED),
        last_copy_kind: copy_kind_name(LAST_COPY_KIND.load(RELAXED)),
        last_copy_width: LAST_COPY_W.load(RELAXED),
        last_copy_height: LAST_COPY_H.load(RELAXED),
        last_copy_unpack_row_length: LAST_COPY_ROW.load(RELAXED),
        last_copy_unpack_image_height: LAST_COPY_IMAGE.load(RELAXED),
        last_copy_needed_bytes: LAST_COPY_NEEDED.load(RELAXED),
        last_copy_take_bytes: LAST_COPY_TAKE.load(RELAXED),
        last_emu_slot: LAST_EMU_SLOT.load(RELAXED),
        last_emu_offset: LAST_EMU_OFFSET.load(RELAXED),
        last_emu_size: LAST_EMU_SIZE.load(RELAXED),
        last_emu_texels: LAST_EMU_TEXELS.load(RELAXED),
        last_emu_width: LAST_EMU_W.load(RELAXED),
        last_emu_height: LAST_EMU_H.load(RELAXED),
        last_emu_full_rows: LAST_EMU_FULL_ROWS.load(RELAXED),
        last_emu_remainder_texels: LAST_EMU_REM.load(RELAXED),
    }
}

/// RGBA32UI texels covering `size` bytes.
///
/// `div_ceil(16)` is enough. `(size + 15).div_ceil(16)` double-ceils when
/// `size` is already 16-aligned, so WebGL sees a PBO overflow
/// (65536-byte instance buffer → 4097×1×16 = 65552).
pub fn emulated_storage_texels(size: i32) -> usize {
    (size.max(0) as usize).div_ceil(16).max(1)
}

/// Zero counters. Tests and a fresh portal boot call this.
pub fn reset() {
    COPY_PBO.store(0, RELAXED);
    COPY_SLICE.store(0, RELAXED);
    COPY_COMPRESSED_PBO.store(0, RELAXED);
    COPY_COMPRESSED_SLICE.store(0, RELAXED);
    EMU_UPLOADS.store(0, RELAXED);
    EMU_SKIPS.store(0, RELAXED);
    EMU_TEXELS.store(0, RELAXED);
    EMU_BYTES.store(0, RELAXED);
    EMU_REMAINDER_ROWS.store(0, RELAXED);
    LAST_COPY_KIND.store(0, RELAXED);
    LAST_COPY_W.store(0, RELAXED);
    LAST_COPY_H.store(0, RELAXED);
    LAST_COPY_ROW.store(0, RELAXED);
    LAST_COPY_IMAGE.store(0, RELAXED);
    LAST_COPY_NEEDED.store(0, RELAXED);
    LAST_COPY_TAKE.store(0, RELAXED);
    LAST_EMU_SLOT.store(0, RELAXED);
    LAST_EMU_OFFSET.store(0, RELAXED);
    LAST_EMU_SIZE.store(0, RELAXED);
    LAST_EMU_TEXELS.store(0, RELAXED);
    LAST_EMU_W.store(0, RELAXED);
    LAST_EMU_H.store(0, RELAXED);
    LAST_EMU_FULL_ROWS.store(0, RELAXED);
    LAST_EMU_REM.store(0, RELAXED);
}

/// Record one CopyBufferToTexture unpack.
pub fn record_copy_buffer_to_texture(
    pbo: bool,
    compressed: bool,
    width: u32,
    height: u32,
    unpack_row_length: i32,
    unpack_image_height: i32,
    needed_bytes: u64,
    take_bytes: u64,
) {
    let kind = match (compressed, pbo) {
        (false, true) => {
            COPY_PBO.fetch_add(1, RELAXED);
            1
        }
        (false, false) => {
            COPY_SLICE.fetch_add(1, RELAXED);
            2
        }
        (true, true) => {
            COPY_COMPRESSED_PBO.fetch_add(1, RELAXED);
            3
        }
        (true, false) => {
            COPY_COMPRESSED_SLICE.fetch_add(1, RELAXED);
            4
        }
    };
    LAST_COPY_KIND.store(kind, RELAXED);
    LAST_COPY_W.store(width, RELAXED);
    LAST_COPY_H.store(height, RELAXED);
    LAST_COPY_ROW.store(unpack_row_length, RELAXED);
    LAST_COPY_IMAGE.store(unpack_image_height, RELAXED);
    LAST_COPY_NEEDED.store(needed_bytes, RELAXED);
    LAST_COPY_TAKE.store(take_bytes, RELAXED);
}

/// Record one BindEmulatedStorage texture upload.
pub fn record_bind_emulated_storage(
    uploaded: bool,
    slot: u32,
    offset: i32,
    size: i32,
    num_texels: u64,
    width: i32,
    height: i32,
    full_rows: i32,
    remainder_texels: i32,
) {
    if uploaded {
        EMU_UPLOADS.fetch_add(1, RELAXED);
        EMU_TEXELS.fetch_add(num_texels, RELAXED);
        EMU_BYTES.fetch_add(size.max(0) as u64, RELAXED);
        if remainder_texels > 0 {
            EMU_REMAINDER_ROWS.fetch_add(1, RELAXED);
        }
        LAST_EMU_SLOT.store(slot, RELAXED);
        LAST_EMU_OFFSET.store(offset, RELAXED);
        LAST_EMU_SIZE.store(size, RELAXED);
        LAST_EMU_TEXELS.store(num_texels, RELAXED);
        LAST_EMU_W.store(width, RELAXED);
        LAST_EMU_H.store(height, RELAXED);
        LAST_EMU_FULL_ROWS.store(full_rows, RELAXED);
        LAST_EMU_REM.store(remainder_texels, RELAXED);
    } else {
        EMU_SKIPS.fetch_add(1, RELAXED);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_copy_and_emulation_separately() {
        reset();
        record_copy_buffer_to_texture(false, false, 32, 16, 0, 0, 2048, 2048);
        record_bind_emulated_storage(true, 3, 0, 65536, 4096, 4096, 1, 1, 0);
        let snap = snapshot();
        assert_eq!(snap.copy_buffer_to_texture_slice, 1);
        assert_eq!(snap.copy_buffer_to_texture_pbo, 0);
        assert_eq!(snap.last_copy_kind, "slice");
        assert_eq!(snap.last_copy_width, 32);
        assert_eq!(snap.bind_emulated_storage_uploads, 1);
        assert_eq!(snap.last_emu_size, 65536);
        assert_eq!(snap.last_emu_texels, 4096);
        reset();
        assert_eq!(snapshot().bind_emulated_storage_uploads, 0);
    }

    #[test]
    fn emulated_storage_texels_fit_aligned_pbo() {
        assert_eq!(emulated_storage_texels(65536), 4096);
        assert_eq!(emulated_storage_texels(65536) * 16, 65536);
        assert_eq!(emulated_storage_texels(38144), 2384);
        assert_eq!(emulated_storage_texels(1), 1);
        assert_eq!(emulated_storage_texels(0), 1);
        assert_eq!((65536usize + 15).div_ceil(16), 4097);
    }
}
