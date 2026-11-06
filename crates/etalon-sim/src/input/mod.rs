//! Zero-Heap Bitset Input Engine & Mobile Touch Gestures (`input`).
//!
//! Provides:
//! - 256-channel zero-allocation button tracking (`ZeroHeapButtonInput`),
//! - Mobile virtual touch sticks with deadzones and radial clamping (`VirtualTouchStick`),
//! - 10-slot multi-touch finger gesture tracker (`MultiTouchTracker`).

pub mod buttons;
pub mod touch;

#[cfg(test)]
mod tests;

pub use buttons::*;
pub use touch::*;
