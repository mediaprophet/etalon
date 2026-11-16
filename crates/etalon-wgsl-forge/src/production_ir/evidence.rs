//! Stage-specific receipts, canonical keys, and dependency-aware cache adapters.

use serde::{Deserialize, Serialize};

use crate::cache::{
    AdapterIdentity, CertificationManifest, TimingSource, TimingSummary, ValidationLevel,
    CACHE_SCHEMA_V2,
};
use crate::certify::{ComparisonReport, ValidationReport};
use crate::emit::GeneratedShader;
use crate::ir::ForgeError;
use crate::{
    CUDARC_API_VERSION, FORGE_SCHEMA_VERSION_NUM, NAGA_API_VERSION, WGPU_API_VERSION,
};

use super::contract::{ArithmeticModel, ProductionContract};
use super::rewrites::RewriteTrace;
use super::{delimited_blake3, hex64};

pub const EVIDENCE_SCHEMA_V1: &[u8] = b"etalon-production-evidence-v1";
pub const EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStage {
    Generated,
    NagaValidated,
    PipelineCreated,
    OracleVerified,
    Profiled,
    Certified,
}

impl EvidenceStage {
    pub fn validation_level(self) -> ValidationLevel {
        match self {
            Self::Generated => ValidationLevel::Generated,
            Self::NagaValidated => ValidationLevel::NagaValidated,
            Self::PipelineCreated => ValidationLevel::PipelineCreated,
            Self::OracleVerified => ValidationLevel::OracleVerified,
            Self::Profiled => ValidationLevel::Profiled,
            Self::Certified => ValidationLevel::Certified,
        }
    }

    pub fn successor(self) -> Option<Self> {
        match self {
            Self::Generated => Some(Self::NagaValidated),
            Self::NagaValidated => Some(Self::PipelineCreated),
            Self::PipelineCreated => Some(Self::OracleVerified),
            Self::OracleVerified => Some(Self::Profiled),
            Self::Profiled => Some(Self::Certified),
            Self::Certified => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigestAlgorithm {
    Blake3,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digest {
    pub algorithm: DigestAlgorithm,
    pub hex: String,
}

impl Digest {
    pub fn blake3(bytes: &[u8]) -> Self {
        Self {
            algorithm: DigestAlgorithm::Blake3,
            hex: blake3::hash(bytes).to_hex().to_string(),
        }
    }

    pub fn from_hex(hex: impl Into<String>) -> Result<Self, ForgeError> {
        let hex = hex.into();
        if !hex64(&hex) {
            return Err(ForgeError::Validation(
                "digest must be a 64-character blake3 hex string".to_string(),
            ));
        }
        Ok(Self {
            algorithm: DigestAlgorithm::Blake3,
            hex,
        })
    }

    /// CRC and other checksums are not signatures and are not accepted.
    pub fn parse(algorithm: &str, hex: &str) -> Result<Self, ForgeError> {
        let normalized = algorithm.trim().to_ascii_lowercase();
        if normalized == "crc" || normalized.starts_with("crc") {
            return Err(ForgeError::Validation(
                "CRC is a checksum, not a signature".to_string(),
            ));
        }
        if normalized != "blake3" && normalized != "blake-3" {
            return Err(ForgeError::Validation(format!(
                "unsupported digest algorithm {algorithm:?}; production evidence requires blake3"
            )));
        }
        Self::from_hex(hex)
    }

    pub fn validate(&self) -> Result<(), ForgeError> {
        if self.algorithm != DigestAlgorithm::Blake3 {
            return Err(ForgeError::Validation(
                "production evidence digests must be blake3".to_string(),
            ));
        }
        if self.hex.is_empty() {
            return Err(ForgeError::Validation(
                "missing evidence digest hash".to_string(),
            ));
        }
        if !hex64(&self.hex) {
            return Err(ForgeError::Validation(
                "evidence digest is not a 64-character blake3 hex string".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementMethod {
    None,
    GpuTimestamp,
    CompletionClock,
    Synthetic,
}

impl MeasurementMethod {
    pub fn from_timing_source(source: TimingSource) -> Self {
        match source {
            TimingSource::GpuTimestamp => Self::GpuTimestamp,
            TimingSource::CompletionClock => Self::CompletionClock,
            TimingSource::Synthetic => Self::Synthetic,
        }
    }

    pub fn as_timing_source(self) -> Option<TimingSource> {
        match self {
            Self::None => None,
            Self::GpuTimestamp => Some(TimingSource::GpuTimestamp),
            Self::CompletionClock => Some(TimingSource::CompletionClock),
            Self::Synthetic => Some(TimingSource::Synthetic),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageOptions {
    pub backend: String,
    pub precision: ArithmeticModel,
}

impl StageOptions {
    pub fn wgsl(precision: ArithmeticModel) -> Self {
        Self {
            backend: "wgsl".to_string(),
            precision,
        }
    }

    pub fn canonical_hash(&self) -> Result<String, ForgeError> {
        let bytes = serde_json::to_vec(self)?;
        Ok(delimited_blake3(EVIDENCE_SCHEMA_V1, &[bytes.as_slice()]))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PromotionEvidence<'a> {
    pub shader: &'a GeneratedShader,
    pub validation: Option<&'a ValidationReport>,
    pub adapter: Option<&'a AdapterIdentity>,
    pub oracle: Option<&'a ComparisonReport>,
    pub timing: Option<&'a TimingSummary>,
    pub measurement: MeasurementMethod,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageReceipt {
    pub schema_version: u32,
    pub stage: EvidenceStage,
    pub kernel_id: String,
    pub input_digest: Digest,
    pub source_digest: Digest,
    pub options_digest: Digest,
    pub abi_digest: Digest,
    pub precision: ArithmeticModel,
    pub adapter: Option<AdapterIdentity>,
    pub measurement: MeasurementMethod,
    pub parent_cache_key: Option<String>,
    pub cache_key: String,
    pub validation_level: ValidationLevel,
    pub timing: Option<TimingSummary>,
    pub rewrite_trace_digest: Option<Digest>,
    pub certification: Option<CertificationManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub cache_key: String,
    pub stage: EvidenceStage,
}

impl StageReceipt {
    pub fn issue_generated(
        contract: &ProductionContract,
        shader: &GeneratedShader,
        options: &StageOptions,
        rewrite_trace: Option<&RewriteTrace>,
    ) -> Result<Self, ForgeError> {
        contract.validate()?;
        if options.precision != contract.arithmetic {
            return Err(ForgeError::Validation(
                "stage options precision does not match contract arithmetic".to_string(),
            ));
        }
        if shader.kernel_id != contract.kernel.id {
            return Err(ForgeError::Validation(
                "generated shader kernel id does not match contract".to_string(),
            ));
        }
        if shader.schedule != contract.schedule {
            return Err(ForgeError::Validation(
                "generated shader schedule does not match contract".to_string(),
            ));
        }
        let source_digest = Digest::blake3(shader.source.as_bytes());
        if shader.source_hash != source_digest.hex {
            return Err(ForgeError::Validation(
                "generated source_hash does not match blake3(source)".to_string(),
            ));
        }
        let mut receipt = Self {
            schema_version: EVIDENCE_SCHEMA_VERSION,
            stage: EvidenceStage::Generated,
            kernel_id: contract.kernel.id.clone(),
            input_digest: Digest::from_hex(contract.canonical_hash()?)?,
            source_digest,
            options_digest: combine_options(contract, options)?,
            abi_digest: Digest::from_hex(contract.abi_hash()?)?,
            precision: contract.arithmetic,
            adapter: None,
            measurement: MeasurementMethod::None,
            parent_cache_key: None,
            cache_key: String::new(),
            validation_level: ValidationLevel::Generated,
            timing: None,
            rewrite_trace_digest: match rewrite_trace {
                Some(trace) => Some(Digest::from_hex(trace.canonical_hash()?)?),
                None => None,
            },
            certification: None,
        };
        receipt.cache_key = receipt.canonical_cache_key()?;
        Ok(receipt)
    }

    pub fn promote(
        &self,
        next: EvidenceStage,
        evidence: PromotionEvidence<'_>,
    ) -> Result<Self, ForgeError> {
        self.validate_shape()?;
        if self.stage.successor() != Some(next) {
            return Err(ForgeError::Validation(format!(
                "cannot promote {:?} to {:?}; stages are not skipped",
                self.stage, next
            )));
        }
        if self.source_digest.hex != evidence.shader.source_hash
            || evidence.shader.source_hash
                != blake3::hash(evidence.shader.source.as_bytes())
                    .to_hex()
                    .as_str()
        {
            return Err(ForgeError::Validation(
                "refusing to promote stale source evidence".to_string(),
            ));
        }
        let mut promoted = self.clone();
        promoted.schema_version = EVIDENCE_SCHEMA_VERSION;
        promoted.stage = next;
        promoted.validation_level = next.validation_level();
        promoted.parent_cache_key = Some(self.cache_key.clone());
        promoted.measurement = evidence.measurement;
        promoted.timing = evidence.timing.cloned();
        match next {
            EvidenceStage::NagaValidated => {
                let report = evidence.validation.ok_or_else(|| {
                    ForgeError::Validation("Naga promotion requires a validation report".to_string())
                })?;
                let manifest = CertificationManifest::naga_only(evidence.shader, report.clone());
                if manifest.validation_level != ValidationLevel::NagaValidated {
                    return Err(ForgeError::Validation(
                        "refusing to promote stale or mismatched Naga evidence".to_string(),
                    ));
                }
                promoted.certification = Some(manifest);
                promoted.adapter = None;
                promoted.measurement = MeasurementMethod::None;
                promoted.timing = None;
            }
            EvidenceStage::PipelineCreated => {
                let adapter = evidence.adapter.ok_or_else(|| {
                    ForgeError::Validation("pipeline evidence requires AdapterIdentity".to_string())
                })?;
                promoted.adapter = Some(adapter.clone());
                bind_adapter_key(&mut promoted, evidence.shader, adapter)?;
            }
            EvidenceStage::OracleVerified => {
                let adapter = require_same_adapter(self, evidence.adapter)?;
                let report = evidence.oracle.ok_or_else(|| {
                    ForgeError::Validation(
                        "oracle promotion requires a comparison report".to_string(),
                    )
                })?;
                if !report.passed() {
                    return Err(ForgeError::OracleMismatch(
                        "oracle comparison did not pass".to_string(),
                    ));
                }
                promoted.adapter = Some(adapter.clone());
                if let Some(manifest) = promoted.certification.as_mut() {
                    manifest.adapter = Some(adapter.clone());
                    manifest.oracle = Some(report.clone());
                    manifest.validation_level = ValidationLevel::OracleVerified;
                }
                bind_adapter_key(&mut promoted, evidence.shader, adapter)?;
            }
            EvidenceStage::Profiled => {
                let adapter = require_same_adapter(self, evidence.adapter)?;
                let summary = evidence.timing.ok_or_else(|| {
                    ForgeError::Validation("profiled evidence requires a timing summary".to_string())
                })?;
                reject_synthetic_as_gpu(evidence.measurement, summary)?;
                promoted.adapter = Some(adapter.clone());
                promoted.timing = Some(summary.clone());
                promoted.measurement = evidence.measurement;
                if let Some(manifest) = promoted.certification.as_mut() {
                    manifest.timing = Some(summary.clone());
                    manifest.validation_level = ValidationLevel::Profiled;
                }
                bind_adapter_key(&mut promoted, evidence.shader, adapter)?;
            }
            EvidenceStage::Certified => {
                let adapter = require_same_adapter(self, evidence.adapter)?;
                if self.certification.is_none() || self.adapter.is_none() || self.timing.is_none()
                {
                    return Err(ForgeError::Validation(
                        "certified evidence requires Naga, adapter, oracle, and timing parents"
                            .to_string(),
                    ));
                }
                if let Some(summary) = evidence.timing.or(self.timing.as_ref()) {
                    promoted.timing = Some(summary.clone());
                }
                promoted.adapter = Some(adapter.clone());
                promoted.measurement = if evidence.measurement == MeasurementMethod::None {
                    self.measurement
                } else {
                    evidence.measurement
                };
                if let Some(summary) = promoted.timing.as_ref() {
                    reject_synthetic_as_gpu(promoted.measurement, summary)?;
                }
                bind_adapter_key(&mut promoted, evidence.shader, adapter)?;
                if let Some(manifest) = promoted.certification.as_mut() {
                    manifest.timing = promoted.timing.clone();
                    manifest.validation_level = ValidationLevel::Certified;
                }
            }
            EvidenceStage::Generated => {
                return Err(ForgeError::Validation(
                    "Generated is issued, not promoted".to_string(),
                ));
            }
        }
        promoted.cache_key = promoted.canonical_cache_key()?;
        promoted.validate_shape()?;
        Ok(promoted)
    }

    pub fn canonical_cache_key(&self) -> Result<String, ForgeError> {
        let stage = stage_bytes(self.stage);
        let precision = precision_bytes(self.precision);
        let measurement = measurement_bytes(self.measurement);
        let adapter = match &self.adapter {
            Some(adapter) => serde_json::to_vec(adapter)?,
            None => Vec::new(),
        };
        let parent = self.parent_cache_key.as_deref().unwrap_or("");
        let rewrite = self
            .rewrite_trace_digest
            .as_ref()
            .map(|digest| digest.hex.as_str())
            .unwrap_or("");
        let crate_version = env!("CARGO_PKG_VERSION");
        Ok(delimited_blake3(
            EVIDENCE_SCHEMA_V1,
            &[
                CACHE_SCHEMA_V2,
                &EVIDENCE_SCHEMA_VERSION.to_le_bytes(),
                &FORGE_SCHEMA_VERSION_NUM.to_le_bytes(),
                stage,
                self.kernel_id.as_bytes(),
                self.input_digest.hex.as_bytes(),
                self.source_digest.hex.as_bytes(),
                self.options_digest.hex.as_bytes(),
                self.abi_digest.hex.as_bytes(),
                precision,
                adapter.as_slice(),
                measurement,
                parent.as_bytes(),
                rewrite.as_bytes(),
                crate_version.as_bytes(),
                WGPU_API_VERSION.as_bytes(),
                NAGA_API_VERSION.as_bytes(),
                CUDARC_API_VERSION.as_bytes(),
            ],
        ))
    }

    pub fn depends_on(&self, parent: &Self) -> bool {
        self.parent_cache_key.as_deref() == Some(parent.cache_key.as_str())
    }

    pub fn identity_matches(&self, other: &Self) -> bool {
        self.input_digest == other.input_digest
            && self.source_digest == other.source_digest
            && self.options_digest == other.options_digest
            && self.abi_digest == other.abi_digest
            && self.precision == other.precision
            && self.rewrite_trace_digest == other.rewrite_trace_digest
    }

    pub fn is_stale_child_of(&self, parent: &Self) -> bool {
        if self.kernel_id != parent.kernel_id {
            return false;
        }
        if self.depends_on(parent) {
            return !self.identity_matches(parent);
        }
        self.stage > EvidenceStage::Generated && !self.identity_matches(parent)
    }

    pub fn replay(&self) -> Result<ReplayReport, ForgeError> {
        if self.schema_version != EVIDENCE_SCHEMA_VERSION {
            return Err(ForgeError::Validation(format!(
                "unsupported evidence schema {}",
                self.schema_version
            )));
        }
        self.validate_shape()?;
        let recomputed = self.canonical_cache_key()?;
        if recomputed != self.cache_key {
            return Err(ForgeError::Validation(
                "replay cache key does not match canonical bytes".to_string(),
            ));
        }
        if let Some(manifest) = &self.certification {
            if manifest.source_hash != self.source_digest.hex {
                return Err(ForgeError::Validation(
                    "replay rejected stale certification source hash".to_string(),
                ));
            }
            if self.stage >= EvidenceStage::NagaValidated
                && manifest.validation_level < ValidationLevel::NagaValidated
            {
                return Err(ForgeError::Validation(
                    "replay rejected certification that was never Naga-validated".to_string(),
                ));
            }
        }
        Ok(ReplayReport {
            cache_key: self.cache_key.clone(),
            stage: self.stage,
        })
    }

    fn validate_shape(&self) -> Result<(), ForgeError> {
        if self.schema_version != EVIDENCE_SCHEMA_VERSION {
            return Err(ForgeError::Validation(format!(
                "unsupported evidence schema {}",
                self.schema_version
            )));
        }
        if self.kernel_id.is_empty() {
            return Err(ForgeError::Validation(
                "evidence receipt missing kernel id".to_string(),
            ));
        }
        self.input_digest.validate()?;
        self.source_digest.validate()?;
        self.options_digest.validate()?;
        self.abi_digest.validate()?;
        if let Some(digest) = &self.rewrite_trace_digest {
            digest.validate()?;
        }
        if self.validation_level != self.stage.validation_level() {
            return Err(ForgeError::Validation(
                "receipt validation_level does not match evidence stage".to_string(),
            ));
        }
        if let Some(timing) = &self.timing {
            reject_synthetic_as_gpu(self.measurement, timing)?;
        } else if matches!(
            self.measurement,
            MeasurementMethod::GpuTimestamp | MeasurementMethod::CompletionClock
        ) {
            return Err(ForgeError::Validation(
                "GPU measurement method requires a timing summary".to_string(),
            ));
        }
        if self.stage >= EvidenceStage::PipelineCreated && self.adapter.is_none() {
            return Err(ForgeError::Validation(
                "adapter-backed stages require AdapterIdentity".to_string(),
            ));
        }
        if let Some(parent) = &self.parent_cache_key {
            if !hex64(parent) {
                return Err(ForgeError::Validation(
                    "parent cache key must be a 64-character hex digest".to_string(),
                ));
            }
        } else if self.stage != EvidenceStage::Generated {
            return Err(ForgeError::Validation(
                "non-generated receipts require a parent cache key".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn invalidate_descendants(parent: &StageReceipt, receipts: &mut [StageReceipt]) -> usize {
    let mut invalidated_keys = vec![parent.cache_key.clone()];
    let mut stale = 0usize;
    let mut changed = true;
    while changed {
        changed = false;
        for receipt in receipts.iter_mut() {
            if receipt.stage == EvidenceStage::Generated && receipt.parent_cache_key.is_none() {
                continue;
            }
            let depends = receipt
                .parent_cache_key
                .as_ref()
                .is_some_and(|key| invalidated_keys.iter().any(|inv| inv == key));
            if depends {
                invalidated_keys.push(receipt.cache_key.clone());
                receipt.validation_level = ValidationLevel::Generated;
                receipt.stage = EvidenceStage::Generated;
                receipt.certification = None;
                receipt.timing = None;
                receipt.adapter = None;
                receipt.measurement = MeasurementMethod::None;
                receipt.parent_cache_key = None;
                stale += 1;
                changed = true;
            }
        }
    }
    stale
}

fn combine_options(
    contract: &ProductionContract,
    options: &StageOptions,
) -> Result<Digest, ForgeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(EVIDENCE_SCHEMA_V1);
    let contract_options = contract.options_hash()?;
    hasher.update(&(contract_options.len() as u64).to_le_bytes());
    hasher.update(contract_options.as_bytes());
    let stage_options = options.canonical_hash()?;
    hasher.update(&(stage_options.len() as u64).to_le_bytes());
    hasher.update(stage_options.as_bytes());
    Digest::from_hex(hasher.finalize().to_hex().to_string())
}

fn bind_adapter_key(
    receipt: &mut StageReceipt,
    shader: &GeneratedShader,
    adapter: &AdapterIdentity,
) -> Result<(), ForgeError> {
    if let Some(manifest) = receipt.certification.as_mut() {
        manifest.adapter = Some(adapter.clone());
        manifest.cache_key = Some(adapter.cache_key(
            &shader.semantic_hash,
            &shader.source_hash,
            shader.schedule,
            (1.0e-6, 1.0e-5),
        )?);
    }
    Ok(())
}

fn require_same_adapter<'a>(
    parent: &StageReceipt,
    adapter: Option<&'a AdapterIdentity>,
) -> Result<&'a AdapterIdentity, ForgeError> {
    let adapter = adapter.ok_or_else(|| {
        ForgeError::Validation("adapter-backed promotion requires AdapterIdentity".to_string())
    })?;
    match &parent.adapter {
        Some(existing) if existing != adapter => {
            Err(ForgeError::Validation(
                "adapter identity changed; downstream evidence is invalid".to_string(),
            ))
        }
        _ => Ok(adapter),
    }
}

fn reject_synthetic_as_gpu(
    measurement: MeasurementMethod,
    timing: &TimingSummary,
) -> Result<(), ForgeError> {
    if measurement == MeasurementMethod::GpuTimestamp
        && timing.source != TimingSource::GpuTimestamp
    {
        return Err(ForgeError::Validation(
            "synthetic or completion-clock timing cannot be presented as GPU timing".to_string(),
        ));
    }
    if timing.source == TimingSource::Synthetic && measurement == MeasurementMethod::GpuTimestamp {
        return Err(ForgeError::Validation(
            "synthetic timing cannot be presented as GPU timing".to_string(),
        ));
    }
    if let Some(expected) = measurement.as_timing_source() {
        if expected != timing.source {
            return Err(ForgeError::Validation(
                "measurement method does not match timing source".to_string(),
            ));
        }
    }
    Ok(())
}

fn stage_bytes(stage: EvidenceStage) -> &'static [u8] {
    match stage {
        EvidenceStage::Generated => b"generated",
        EvidenceStage::NagaValidated => b"naga_validated",
        EvidenceStage::PipelineCreated => b"pipeline_created",
        EvidenceStage::OracleVerified => b"oracle_verified",
        EvidenceStage::Profiled => b"profiled",
        EvidenceStage::Certified => b"certified",
    }
}

fn precision_bytes(model: ArithmeticModel) -> &'static [u8] {
    match model {
        ArithmeticModel::IeeeF32 => b"ieee_f32",
        ArithmeticModel::Df64PairedF32 => b"df64_paired_f32",
        ArithmeticModel::IntegerI32 => b"integer_i32",
        ArithmeticModel::IntegerU32 => b"integer_u32",
        ArithmeticModel::FilteredPredicate => b"filtered_predicate",
    }
}

fn measurement_bytes(method: MeasurementMethod) -> &'static [u8] {
    match method {
        MeasurementMethod::None => b"none",
        MeasurementMethod::GpuTimestamp => b"gpu_timestamp",
        MeasurementMethod::CompletionClock => b"completion_clock",
        MeasurementMethod::Synthetic => b"synthetic",
    }
}
