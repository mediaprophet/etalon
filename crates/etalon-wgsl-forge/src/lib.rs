//! Pure IR, multi-target shader emission, and hardware certification for Etalon.

pub mod ir;
pub mod schedule;
pub mod emit;
pub mod certifier;
pub mod autotune;

pub use ir::{DType, ForgeError, Op, P64GpuWords64, TensorViewDesc};
pub use schedule::{Schedule, TileConfig, WorkgroupSize};
pub use emit::{TargetBackend, emit_kernel};
pub use certifier::certify_kernel;

pub const FORGE_SCHEMA_VERSION: &str = "0.0.1-etalon";
pub const FORGE_CACHE_MAGIC: [u8; 8] = *b"ETALFORG";
