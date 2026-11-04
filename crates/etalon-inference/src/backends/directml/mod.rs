//! DirectML 1.15 inference bridge + DirectStorage integration.
//!
//! Provides a D3D12 + DirectML device pair and Q4_K dequantization used by
//! `gguf_bridge::QTensorEngine` on Windows x64.
//!
//! Inference path (Windows):
//!   GGUF tensor bytes → dequantize_q4_k_tensor() → DmlGemmOp::execute()
//!     → D3D12 upload buffers → DirectML GEMM dispatch → readback → f32 logits
//!
//! Calculus modality path (Windows):
//!   NVMe file → DirectStorage → GPU VRAM (DMA bypass) → compute shader → result
//!
//! FFI Firewall:
//!   To resolve DirectX API version conflicts between host.rs and directml_bridge.rs,
//!   this module exposes type-erased FFI functions using *mut c_void. Callers
//!   pass raw pointers, and this module casts them back to specific Windows types
//!   internally. This isolates dependency trees and prevents COM interface conflicts.
//!
//! On non-Windows targets this entire module is compiled out via the #[cfg] in lib.rs.

#![cfg(target_os = "windows")]
#![allow(non_snake_case)]

use std::fs::File;
use std::mem::ManuallyDrop;
use windows::{
    core::Interface,
    Win32::{
        Foundation::HANDLE,
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_0,
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
        AI::MachineLearning::DirectML::*,
    },
};

mod buffers;
mod device;
mod ffi;
mod gemm;
mod lease;
mod quant;
mod storage;
use buffers::*;
pub use device::*;
pub use ffi::*;
pub use gemm::*;
pub use lease::DmlProviderLease;
pub use quant::*;
pub use storage::*;
// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_q4k_midpoint() {
        let mut block = [0u8; Q4_K_BLOCK_BYTES];
        // scale = f16(1.0) = 0x3C00, min = f16(0.0) = 0x0000, all nibbles = 7
        block[0] = 0x00;
        block[1] = 0x3C;
        block[2] = 0x00;
        block[3] = 0x00;
        for i in 4..Q4_K_BLOCK_BYTES {
            block[i] = 0x77;
        }
        let mut out = [0f32; Q4_K_BLOCK_SIZE];
        dequantize_q4_k_block(&block, &mut out);
        for v in &out {
            assert!((v - 7.0).abs() < 1e-3, "got {v}");
        }
    }

    #[test]
    fn dml_device_init_smoke() {
        match DmlDevice::new() {
            Ok(dev) => println!("DirectML OK — {}", dev.adapter_desc),
            Err(e) => println!("DirectML unavailable (expected in headless CI): {e}"),
        }
    }
}
