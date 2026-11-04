//! Ablation matrix: toggle axes × decode-proxy tok/s → experiment CSV.

use std::path::Path;

use crate::auto_improve::LabMode;
use crate::experiment_log::{append_run_csv, ExperimentRun};
pub trait AblationExecutor<M: LabMode> {
    fn execute(
        &mut self,
        model: &Path,
        tokens: u32,
        resident: bool,
        coop: bool,
        ffn: bool,
        kv: bool,
        mode: M,
        forge: Option<&str>,
    ) -> Option<f64>;
    fn passport_key(&self) -> String {
        String::new()
    }
    fn backend_label(&self) -> String {
        String::from("unknown")
    }
}
#[derive(Debug, Clone)]
pub struct AblationRow {
    pub label: String,
    pub tok_s: Option<f64>,
    pub toggles: String,
}

/// Default small ablation matrix (fast enough for lab sessions).
pub fn run_ablation_matrix<M: LabMode>(
    model: &Path,
    tokens: u32,
    csv_out: Option<&Path>,
    host: &mut dyn AblationExecutor<M>,
) -> Vec<AblationRow> {
    let tokens = tokens.max(4).min(32);
    let model_id = model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    // (label, resident, coop, ffn_flag, kv_int8, mode)
    let cases: [(&str, bool, bool, bool, bool, M); 6] = [
        ("baseline_resident", true, true, true, true, M::PORTABLE),
        ("no_coop", true, false, true, true, M::PORTABLE),
        ("no_resident", false, true, true, true, M::PORTABLE),
        ("kv_f32", true, true, true, false, M::PORTABLE),
        ("fast_verify", true, true, true, true, M::FAST_VERIFY),
        ("cuda_mode", true, true, true, true, M::CUDA_TC),
    ];

    let mut rows = Vec::new();
    let passport_key = host.passport_key();

    for (label, res, coop, ffn, kv, mode) in cases {
        let toggles = format!(
            "{{\"resident\":{res},\"coop\":{coop},\"ffn_fusion\":{ffn},\"kv_int8\":{kv},\"mode\":\"{}\"}}",
            mode.as_str()
        );
        let tok_s = host.execute(model, tokens, res, coop, ffn, kv, mode, None);
        rows.push(AblationRow {
            label: label.into(),
            tok_s,
            toggles: toggles.clone(),
        });

        if let Some(path) = csv_out {
            let mut run = ExperimentRun {
                run_id: ExperimentRun::new_id(),
                utc: ExperimentRun::utc_now(),
                git_sha: String::new(),
                host_passport_key: passport_key.clone(),
                adapter: String::new(),
                backend: host.backend_label(),
                model_id: model_id.clone(),
                model_hash: String::new(),
                layout: guess_layout(&model_id),
                mode: mode.as_str().into(),
                profile: "lab-ablate".into(),
                toggles_json: toggles,
                qualia_decode_tok_s: tok_s,
                ollama_decode_tok_s: None,
                a_gap: None,
                prefill_tok_s: None,
                phase_ns_json: "{}".into(),
                n_ulp_max: None,
                c_score: None,
                notes: label.into(),
            };
            run.compute_a_gap();
            let _ = append_run_csv(path, &run);
        }
    }

    let forge_backends: &[(&str, &str)] = &[
        ("forge_wgsl", "wgsl"),
        ("forge_hlsl", "hlsl"),
        ("forge_spirv", "spirv"),
        ("forge_ptx", "ptx"),
    ];

    for &(label, backend) in forge_backends {
        let toggles = format!(
            "{{\"forge_backend\":\"{backend}\",\"resident\":true,\"coop\":true,\"ffn_fusion\":true,\"kv_int8\":true,\"mode\":\"portable\"}}"
        );
        let tok_s = host.execute(
            model,
            tokens,
            true,
            true,
            true,
            true,
            M::PORTABLE,
            Some(backend),
        );
        rows.push(AblationRow {
            label: label.into(),
            tok_s,
            toggles: toggles.clone(),
        });

        if let Some(path) = csv_out {
            let mut run = ExperimentRun {
                run_id: ExperimentRun::new_id(),
                utc: ExperimentRun::utc_now(),
                git_sha: String::new(),
                host_passport_key: passport_key.clone(),
                adapter: String::new(),
                backend: format!("forge-{backend}"),
                model_id: model_id.clone(),
                model_hash: String::new(),
                layout: guess_layout(&model_id),
                mode: "portable".into(),
                profile: "lab-ablate".into(),
                toggles_json: toggles,
                qualia_decode_tok_s: tok_s,
                ollama_decode_tok_s: None,
                a_gap: None,
                prefill_tok_s: None,
                phase_ns_json: "{}".into(),
                n_ulp_max: None,
                c_score: None,
                notes: label.into(),
            };
            run.compute_a_gap();
            let _ = append_run_csv(path, &run);
        }
    }

    rows
}

fn guess_layout(name: &str) -> String {
    let n = name.to_ascii_lowercase();
    if n.contains(".soa.") || n.contains("soa") {
        "soa".into()
    } else if n.contains("f16") {
        "f16".into()
    } else {
        "verbatim".into()
    }
}

/// Formats `ablation-report`.
pub fn format_ablation_report(rows: &[AblationRow]) -> String {
    let mut s = String::from("Ablation matrix (decode-proxy tok/s)\n");
    for r in rows {
        s.push_str(&format!(
            "  {:20}  {:>8}\n",
            r.label,
            r.tok_s
                .map(|t| format!("{t:.3}"))
                .unwrap_or_else(|| "fail".into())
        ));
    }
    s
}
