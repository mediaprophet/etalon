//! Worker event loop steward and OffscreenCanvas ownership manager.

use core::sync::atomic::{AtomicBool, Ordering};

pub struct WorkerSteward {
    pub is_isolated: bool,
    pub running: AtomicBool,
}

impl WorkerSteward {
    pub fn new(is_isolated: bool) -> Self {
        Self {
            is_isolated,
            running: AtomicBool::new(true),
        }
    }

    pub fn is_active(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}
