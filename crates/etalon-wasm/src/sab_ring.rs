//! COOP/COEP SharedArrayBuffer SPSC ring buffer with Atomic fences.

use core::sync::atomic::{AtomicU32, Ordering};

pub struct SharedArrayBufferRing {
    pub head: AtomicU32,
    pub tail: AtomicU32,
    pub capacity: u32,
}

impl SharedArrayBufferRing {
    pub const fn new(capacity: u32) -> Self {
        Self {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            capacity,
        }
    }

    pub fn can_push(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        (head.wrapping_sub(tail)) < self.capacity
    }
}
