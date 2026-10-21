//! DXC SPIR-V binary cache — avoids re-running DXC on identical HLSL source.
//!
//! Includes Runbook 01 Bug 1.4 fix: uses `DXC_CACHE_SCHEMA_V2` with length-delimited
//! `blake3` hashing so `("ab", "c")` and `("a", "bc")` never collide.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::ir::ForgeError;

pub const DXC_CACHE_SCHEMA_V2: &[u8] = b"etalon-dxc-cache-v2";

/// Cache key: length-delimited blake3 hash of `(schema, hlsl_source, entry_point)`.
pub fn cache_key(hlsl_source: &str, entry_point: &str) -> String {
    cache_key_with_flags(hlsl_source, entry_point, "cs_6_0", &[])
}

/// Cache key with explicit target profile and compiler flags (all length-delimited).
pub fn cache_key_with_flags(
    hlsl_source: &str,
    entry_point: &str,
    target_profile: &str,
    flags: &[&str],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DXC_CACHE_SCHEMA_V2);
    hasher.update(&(hlsl_source.len() as u64).to_le_bytes());
    hasher.update(hlsl_source.as_bytes());
    hasher.update(&(entry_point.len() as u64).to_le_bytes());
    hasher.update(entry_point.as_bytes());
    hasher.update(&(target_profile.len() as u64).to_le_bytes());
    hasher.update(target_profile.as_bytes());
    hasher.update(&(flags.len() as u64).to_le_bytes());
    for flag in flags {
        hasher.update(&(flag.len() as u64).to_le_bytes());
        hasher.update(flag.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

struct DxcCache {
    entries: HashMap<String, Vec<u8>>,
}

impl DxcCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

fn dxc_cache() -> &'static Mutex<DxcCache> {
    static C: OnceLock<Mutex<DxcCache>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(DxcCache::new()))
}

/// Compile HLSL to SPIR-V with caching. On a cache hit, returns the stored
/// binary without invoking DXC. On a miss, calls the inner compiler and stores
/// the result.
pub fn compile_hlsl_to_spirv_cached(
    hlsl_source: &str,
    entry_point: &str,
    inner: impl Fn(&str, &str) -> Result<Vec<u8>, ForgeError>,
) -> Result<Vec<u8>, ForgeError> {
    let key = cache_key(hlsl_source, entry_point);

    if let Ok(guard) = dxc_cache().lock() {
        if let Some(blob) = guard.entries.get(&key) {
            return Ok(blob.clone());
        }
    }

    let spirv = inner(hlsl_source, entry_point)?;

    if let Ok(mut guard) = dxc_cache().lock() {
        guard.entries.insert(key, spirv.clone());
    }

    Ok(spirv)
}

/// Clear the DXC SPIR-V cache (useful for testing or forced recompilation).
pub fn clear_dxc_cache() {
    if let Ok(mut guard) = dxc_cache().lock() {
        guard.entries.clear();
    }
}

/// Number of cached SPIR-V binaries.
pub fn dxc_cache_len() -> usize {
    dxc_cache().lock().map(|g| g.entries.len()).unwrap_or(0)
}
