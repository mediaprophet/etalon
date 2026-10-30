//! Atomic, adapter-keyed storage and manifests for certification and tuning evidence.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::autotune::TuningResult;
use crate::certify::{ComparisonReport, ValidationReport};
use crate::emit::GeneratedShader;
use crate::ir::ForgeError;
use crate::schedule::{AdapterConstraints, Schedule};
use crate::{CUDARC_API_VERSION, FORGE_SCHEMA_VERSION_NUM, NAGA_API_VERSION, WGPU_API_VERSION};

pub const CACHE_SCHEMA_V2: &[u8] = b"etalon-forge-cache-v2";

/// Computes a length-delimited Blake3 cache key over source, entry point, and device ID.
pub fn compute_cache_key(source: &str, entry_point: &str, device_id: u64) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CACHE_SCHEMA_V2);
    hasher.update(&device_id.to_le_bytes());
    hasher.update(&(source.len() as u64).to_le_bytes());
    hasher.update(source.as_bytes());
    hasher.update(&(entry_point.len() as u64).to_le_bytes());
    hasher.update(entry_point.as_bytes());
    hasher.finalize().to_hex().to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub adapter: AdapterIdentity,
    pub constraints: AdapterConstraints,
    pub memory_class: String,
    pub supports_timestamp_query: bool,
    pub max_compute_workgroup_storage_size: u32,
    pub max_storage_buffer_binding_size: u64,
    pub min_storage_buffer_offset_alignment: u32,
    pub min_uniform_buffer_offset_alignment: u32,
}

impl HardwareProfile {
    pub fn topology_hash(&self) -> Result<String, ForgeError> {
        let bytes = serde_json::to_vec(&(
            FORGE_SCHEMA_VERSION_NUM,
            &self.adapter,
            &self.constraints,
            &self.memory_class,
            self.max_compute_workgroup_storage_size,
            self.min_storage_buffer_offset_alignment,
            self.min_uniform_buffer_offset_alignment,
            WGPU_API_VERSION,
        ))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationLevel {
    Generated,
    NagaValidated,
    PipelineCreated,
    OracleVerified,
    Profiled,
    Certified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingSource {
    GpuTimestamp,
    CompletionClock,
    Synthetic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingSummary {
    pub source: TimingSource,
    pub sample_count: usize,
    pub minimum_ns: u64,
    pub median_ns: u64,
    pub p95_ns: u64,
}

impl TimingSummary {
    pub fn from_samples(source: TimingSource, samples: &[u64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let median_ns = sorted[(sorted.len() - 1) / 2];
        let p95_index = ((sorted.len() - 1) * 95).div_ceil(100);
        Some(Self {
            source,
            sample_count: sorted.len(),
            minimum_ns: sorted[0],
            median_ns,
            p95_ns: sorted[p95_index],
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterIdentity {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
}

impl AdapterIdentity {
    pub fn cache_key(
        &self,
        semantic_hash: &str,
        source_hash: &str,
        schedule: Schedule,
        tolerance: (f32, f32),
    ) -> Result<String, ForgeError> {
        let bytes = serde_json::to_vec(&(
            FORGE_SCHEMA_VERSION_NUM,
            self,
            semantic_hash,
            source_hash,
            schedule,
            env!("CARGO_PKG_VERSION"),
            WGPU_API_VERSION,
            NAGA_API_VERSION,
            CUDARC_API_VERSION,
            tolerance.0.to_bits(),
            tolerance.1.to_bits(),
        ))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CertificationManifest {
    pub forge_schema_version: u32,
    pub crate_version: String,
    pub wgpu_api_version: String,
    pub naga_api_version: String,
    pub kernel_id: String,
    pub semantic_hash: String,
    pub source_hash: String,
    pub schedule: Schedule,
    pub validation_level: ValidationLevel,
    pub validation: ValidationReport,
    pub adapter: Option<AdapterIdentity>,
    pub oracle: Option<ComparisonReport>,
    pub timing: Option<TimingSummary>,
    pub cache_key: Option<String>,
}

impl CertificationManifest {
    /// Records Naga validation only when the report matches the actual shader source.
    /// Missing or inconsistent evidence retains the report at `ValidationLevel::Generated`.
    pub fn naga_only(generated: &GeneratedShader, validation: ValidationReport) -> Self {
        let validation_level = if validation.naga_validated
            && validation.source_hash == generated.source_hash
            && generated.source_hash == blake3::hash(generated.source.as_bytes()).to_hex().as_str()
        {
            ValidationLevel::NagaValidated
        } else {
            ValidationLevel::Generated
        };
        Self {
            forge_schema_version: FORGE_SCHEMA_VERSION_NUM,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            wgpu_api_version: WGPU_API_VERSION.to_string(),
            naga_api_version: NAGA_API_VERSION.to_string(),
            kernel_id: generated.kernel_id.clone(),
            semantic_hash: generated.semantic_hash.clone(),
            source_hash: generated.source_hash.clone(),
            schedule: generated.schedule,
            validation_level,
            validation,
            adapter: None,
            oracle: None,
            timing: None,
            cache_key: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningManifest {
    pub forge_schema_version: u32,
    pub crate_version: String,
    pub wgpu_api_version: String,
    pub naga_api_version: String,
    pub kernel_id: String,
    pub semantic_hash: String,
    pub winning_source_hash: String,
    pub adapter: AdapterIdentity,
    pub cache_key: String,
    pub result: TuningResult,
}

impl TuningManifest {
    pub fn new(
        generated_winner: &GeneratedShader,
        adapter: AdapterIdentity,
        result: TuningResult,
    ) -> Result<Self, ForgeError> {
        let cache_key = adapter.cache_key(
            &generated_winner.semantic_hash,
            &generated_winner.source_hash,
            generated_winner.schedule,
            (1.0e-6, 1.0e-5),
        )?;
        Ok(Self {
            forge_schema_version: FORGE_SCHEMA_VERSION_NUM,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            wgpu_api_version: WGPU_API_VERSION.to_string(),
            naga_api_version: NAGA_API_VERSION.to_string(),
            kernel_id: generated_winner.kernel_id.clone(),
            semantic_hash: generated_winner.semantic_hash.clone(),
            winning_source_hash: generated_winner.source_hash.clone(),
            adapter,
            cache_key,
            result,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ManifestCache {
    root: PathBuf,
}

impl ManifestCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn store_certification(
        &self,
        manifest: &CertificationManifest,
    ) -> Result<PathBuf, ForgeError> {
        let key = manifest.cache_key.as_deref().ok_or_else(|| {
            ForgeError::Serialization("certification manifest has no adapter cache key".to_string())
        })?;
        self.store_json(key, "certification", manifest)
    }

    pub fn store_tuning(&self, manifest: &TuningManifest) -> Result<PathBuf, ForgeError> {
        self.store_json(&manifest.cache_key, "tuning", manifest)
    }

    pub fn load_tuning(&self, key: &str) -> Result<Option<TuningManifest>, ForgeError> {
        validate_cache_key(key)?;
        let path = self.path_for(key, "tuning");
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_slice(&std::fs::read(path)?)?))
    }

    pub fn topology_key(topology_hash: &str, kernel_id: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CACHE_SCHEMA_V2);
        hasher.update(&(topology_hash.len() as u64).to_le_bytes());
        hasher.update(topology_hash.as_bytes());
        hasher.update(&(kernel_id.len() as u64).to_le_bytes());
        hasher.update(kernel_id.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    fn store_json<T: serde::Serialize>(
        &self,
        key: &str,
        kind: &str,
        value: &T,
    ) -> Result<PathBuf, ForgeError> {
        validate_cache_key(key)?;
        std::fs::create_dir_all(&self.root)?;
        let final_path = self.path_for(key, kind);
        let bytes = serde_json::to_vec_pretty(value)?;
        if final_path.exists() {
            if std::fs::read(&final_path)? == bytes {
                return Ok(final_path);
            }
            return Err(ForgeError::Serialization(format!(
                "immutable cache collision at {}",
                final_path.display()
            )));
        }

        static WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let sequence = WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary_path = self.root.join(format!(
            ".{kind}-{key}-{}-{sequence}.tmp",
            std::process::id()
        ));
        std::fs::write(&temporary_path, &bytes)?;
        if let Err(error) = std::fs::rename(&temporary_path, &final_path) {
            if final_path.exists() && std::fs::read(&final_path)? == bytes {
                let _ = std::fs::remove_file(&temporary_path);
                return Ok(final_path);
            }
            let _ = std::fs::remove_file(&temporary_path);
            return Err(ForgeError::Io(error.to_string()));
        }
        Ok(final_path)
    }

    fn path_for(&self, key: &str, kind: &str) -> PathBuf {
        self.root.join(format!("{kind}-{key}.json"))
    }
}

fn validate_cache_key(key: &str) -> Result<(), ForgeError> {
    if key.len() != 64 || !key.bytes().all(|value| value.is_ascii_hexdigit()) {
        return Err(ForgeError::Serialization(
            "cache key must be exactly 64 hexadecimal characters".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
