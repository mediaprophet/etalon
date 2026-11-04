mod run;
mod types;
use crate::config_space::{ConfigurationSpace, ParameterDef};
pub use run::*;
use std::path::Path;
pub use types::*;
/// Build a default configuration space for inference toggle optimization.
/// This covers the main toggle parameters that `ExperimentConfig::apply_toggles` supports.
pub fn default_toggle_space() -> ConfigurationSpace {
    ConfigurationSpace::new("inference_toggles")
        .with("coop_gemv", ParameterDef::Bool)
        .with("ffn_fusion", ParameterDef::Bool)
        .with("kv_int8", ParameterDef::Bool)
        .with("kv_dict", ParameterDef::Bool)
        .with("resident_decode", ParameterDef::Bool)
        .with("resident_prefill", ParameterDef::Bool)
        .with("resident_weights", ParameterDef::Bool)
        .with("spec_decode", ParameterDef::Bool)
        .with("ternary_ffn", ParameterDef::Bool)
        .with("attention_preproject", ParameterDef::Bool)
        .with("attention_o_fuse", ParameterDef::Bool)
        .with("gpu_topk", ParameterDef::Bool)
        .with("prefix_cache_enabled", ParameterDef::Bool)
        .with("kv_pool_budget_mb", ParameterDef::Int { lo: 64, hi: 2048 })
        .with("prefill_chunk_size", ParameterDef::Int { lo: 64, hi: 512 })
        .with(
            "prompt_precision_mode",
            ParameterDef::Categorical {
                choices: vec![
                    "standard".into(),
                    "compact".into(),
                    "strict_evidence".into(),
                ],
            },
        )
}

/// Save a campaign report to a JSON file.
pub fn save_campaign_report(report: &CampaignReport, path: &Path) -> Result<(), String> {
    let bytes =
        crate::evidence::serialize_bounded(report, crate::evidence::DEFAULT_ARTIFACT_BUDGET)?;
    crate::evidence::atomic_write(path, &bytes, crate::evidence::DEFAULT_ARTIFACT_BUDGET)
}

#[cfg(test)]
mod tests;
