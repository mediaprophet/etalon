//! Length-delimited Blake3 artifact caching.

use blake3::Hasher;

pub const CACHE_SCHEMA_V2: &[u8] = b"etalon-forge-cache-v2";

pub fn compute_cache_key(source: &str, entry_point: &str, device_id: u64) -> String {
    let mut hasher = Hasher::new();
    hasher.update(CACHE_SCHEMA_V2);
    hasher.update(&device_id.to_le_bytes());
    hasher.update(&(source.len() as u64).to_le_bytes());
    hasher.update(source.as_bytes());
    hasher.update(&(entry_point.len() as u64).to_le_bytes());
    hasher.update(entry_point.as_bytes());
    hasher.finalize().to_hex().to_string()
}
