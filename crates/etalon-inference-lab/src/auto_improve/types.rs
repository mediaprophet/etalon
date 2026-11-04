//! Autonomous multi-hour lab loop: measure → search configs → re-measure → lock-in.
//!
//! This is **not** an LLM rewriting kernels. It is a disciplined experimental
//! program that:
//! 1. Explores a discrete configuration space (resident, coop, kv, mode, backend)
//! 2. Optionally re-samples the best configs (self-improvement via evidence)
//! 3. Tracks plateau / wall-clock budget
//! 4. Emits a **lock-in package**: best config, full CSV, methodology text, apply script
//!
//! Plan: `docs/plans/inference-superiority-lab-and-toolset-plan.md` L4.

use super::LabMode;
use std::path::PathBuf;
use std::time::Duration;
/// One point in the searchable config space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabConfig<M: LabMode> {
    pub label: String,
    pub resident: bool,
    pub coop: bool,
    pub ffn_fusion: bool,
    pub kv_int8: bool,
    pub mode: M,
    pub backend: Option<&'static str>, // None = leave env alone
}

impl<M: LabMode> LabConfig<M> {
    /// Serializes toggles to JSON for the experiment log.
    pub fn toggles_json(&self) -> String {
        format!(
            "{{\"resident\":{},\"coop\":{},\"ffn_fusion\":{},\"kv_int8\":{},\"mode\":\"{}\",\"backend\":\"{}\"}}",
            self.resident,
            self.coop,
            self.ffn_fusion,
            self.kv_int8,
            self.mode.as_str(),
            self.backend.unwrap_or("auto")
        )
    }
}

/// Search space for recursive improvement (expandable).
pub fn default_search_space<M: LabMode>() -> Vec<LabConfig<M>> {
    let mut v = Vec::new();
    let modes = [M::PORTABLE, M::FAST_VERIFY, M::CUDA_TC];
    let backends: [Option<&'static str>; 3] = [None, Some("vulkan"), Some("dx12")];
    for &resident in &[true, false] {
        for &coop in &[true, false] {
            for &kv in &[true, false] {
                for &mode in &modes {
                    // Skip nonsense combos early
                    if !resident && mode == M::CUDA_TC {
                        continue;
                    }
                    for &backend in &backends {
                        // Only vary backend for portable/fast-verify resident path
                        if backend.is_some() && (!resident || mode == M::CUDA_TC) {
                            continue;
                        }
                        let label = format!(
                            "r{}_c{}_k{}_{}_{}",
                            resident as u8,
                            coop as u8,
                            kv as u8,
                            mode.as_str(),
                            backend.unwrap_or("auto")
                        );
                        v.push(LabConfig {
                            label,
                            resident,
                            coop,
                            ffn_fusion: true, // flag only; audit says not in resident yet
                            kv_int8: kv,
                            mode,
                            backend,
                        });
                    }
                }
            }
        }
    }
    v
}

#[derive(Debug, Clone)]
pub struct TrialResult<M: LabMode> {
    pub config: LabConfig<M>,
    pub tok_s: Option<f64>,
    pub wall_ms: f64,
    pub generation: u32,
}

#[derive(Debug, Clone)]
pub struct AutoImproveConfig {
    pub model: PathBuf,
    pub tokens: u32,
    pub max_duration: Duration,
    pub out_dir: PathBuf,
    pub ollama_model: Option<String>,
    pub ollama_url: String,
    /// Re-sample top-k each generation (self-improve via variance reduction).
    pub elite_resample: usize,
    /// Stop if best has not improved by this relative fraction for `plateau_gens` gens.
    pub plateau_rel: f64,
    pub plateau_gens: u32,
    pub max_generations: u32,
}

impl Default for AutoImproveConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::from("C:\\LLM_Models\\P64\\smollm2-360m-instruct-q8_0.p64"),
            tokens: 16,
            max_duration: Duration::from_secs(2 * 3600),
            out_dir: PathBuf::from("experiments/inference-lab/lockin"),
            ollama_model: Some("qualia-smol-q8:latest".into()),
            ollama_url: "http://127.0.0.1:11434".into(),
            elite_resample: 3,
            plateau_rel: 0.02,
            plateau_gens: 2,
            max_generations: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LockInPackage<M: LabMode> {
    pub best: Option<TrialResult<M>>,
    pub ollama_tok_s: Option<f64>,
    pub a_gap: Option<f64>,
    pub trials: usize,
    pub generations: u32,
    pub elapsed_secs: f64,
    pub out_dir: PathBuf,
    pub methodology: String,
}
