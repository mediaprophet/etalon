//! COOP/COEP SharedArrayBuffer SPSC ring buffer with Atomic fences.
//!
//! Provides a lock-free, zero-allocation single-producer single-consumer (SPSC)
//! ring buffer over WebAssembly `SharedArrayBuffer` memory.

use bytemuck::{Pod, Zeroable};
use core::sync::atomic::{AtomicU32, Ordering};

pub const SAB_RING_MAGIC: u32 = 0x4554_5352; // "ETSR"
pub const SAB_RING_VERSION: u16 = 1;

/// Fixed 32-byte header located at the base of the SharedArrayBuffer.
#[repr(C, align(16))]
#[derive(Debug)]
pub struct SabRingHeader {
    pub magic: u32,
    pub version: u16,
    pub slot_size: u16,
    pub capacity: u32,
    pub head: AtomicU32,
    pub tail: AtomicU32,
    pub flags: AtomicU32,
    pub _pad: u32,
}

impl SabRingHeader {
    pub fn new(capacity: u32, slot_size: u16) -> Self {
        Self {
            magic: SAB_RING_MAGIC,
            version: SAB_RING_VERSION,
            slot_size,
            capacity,
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            flags: AtomicU32::new(0),
            _pad: 0,
        }
    }
}

/// SPSC ring buffer controller operating over shared memory slots.
pub struct SharedArrayBufferRing<'a> {
    pub header: &'a SabRingHeader,
    pub payload_buffer: &'a mut [u8],
}

impl<'a> SharedArrayBufferRing<'a> {
    pub fn new(header: &'a SabRingHeader, payload_buffer: &'a mut [u8]) -> Result<Self, &'static str> {
        let expected_bytes = (header.capacity as usize) * (header.slot_size as usize);
        if payload_buffer.len() < expected_bytes {
            return Err("Payload buffer is smaller than capacity * slot_size");
        }
        Ok(Self {
            header,
            payload_buffer,
        })
    }

    #[inline]
    pub fn available_read(&self) -> u32 {
        let head = self.header.head.load(Ordering::Acquire);
        let tail = self.header.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    #[inline]
    pub fn available_write(&self) -> u32 {
        self.header.capacity.saturating_sub(self.available_read())
    }

    /// Try to push a slot into the ring buffer.
    /// Returns `true` if pushed successfully, `false` if the buffer is full.
    pub fn try_push(&mut self, slot_data: &[u8]) -> bool {
        let slot_sz = self.header.slot_size as usize;
        if slot_data.len() > slot_sz || self.available_write() == 0 {
            return false;
        }

        let head = self.header.head.load(Ordering::Relaxed);
        let slot_idx = (head % self.header.capacity) as usize;
        let offset = slot_idx * slot_sz;

        self.payload_buffer[offset..offset + slot_data.len()].copy_from_slice(slot_data);
        if slot_data.len() < slot_sz {
            self.payload_buffer[offset + slot_data.len()..offset + slot_sz].fill(0);
        }

        self.header.head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Try to pop a slot from the ring buffer into the caller-provided buffer.
    /// Returns `true` if popped, `false` if empty.
    pub fn try_pop(&mut self, out_slot: &mut [u8]) -> bool {
        let slot_sz = self.header.slot_size as usize;
        if out_slot.len() < slot_sz || self.available_read() == 0 {
            return false;
        }

        let tail = self.header.tail.load(Ordering::Relaxed);
        let slot_idx = (tail % self.header.capacity) as usize;
        let offset = slot_idx * slot_sz;

        out_slot[..slot_sz].copy_from_slice(&self.payload_buffer[offset..offset + slot_sz]);

        self.header.tail.store(tail.wrapping_add(1), Ordering::Release);
        true
    }
}
