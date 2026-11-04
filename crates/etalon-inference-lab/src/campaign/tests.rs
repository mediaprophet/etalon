use super::*;
use crate::hypothesis::BeliefGraph;
use crate::pareto::{ApplicationProfileWeight, ParetoFrontier};

#[test]
fn default_toggle_space_has_dims() {
    let space = default_toggle_space();
    assert!(space.dims() >= 10);
}

#[test]
fn campaign_config_builder() {
    let space = default_toggle_space();
    let cfg = CampaignConfig::new(space, "model.gguf", 20)
        .with_prompt("Hello")
        .with_profile(ApplicationProfileWeight::LiveFast)
        .with_batch_size(4);

    assert_eq!(cfg.budget, 20);
    assert_eq!(cfg.batch_size, 4);
    assert_eq!(cfg.profile, ApplicationProfileWeight::LiveFast);
    assert_eq!(cfg.prompt, "Hello");
}

#[test]
fn campaign_report_json_serializes() {
    let report = CampaignReport {
        results: Vec::new(),
        frontier: ParetoFrontier::default(),
        beliefs: BeliefGraph::new(),
        best_index: None,
        trials_run: 0,
        elapsed_s: 0.0,
        baseline_tok_s: Some(50.0),
        best_tok_s: Some(75.0),
        summary: "test".into(),
    };
    let json = report.to_json();
    assert!(json.contains("trials_run"));
    assert!(json.contains("baseline_tok_s"));
}

#[test]
fn improvement_ratio_computes() {
    let report = CampaignReport {
        results: Vec::new(),
        frontier: ParetoFrontier::default(),
        beliefs: BeliefGraph::new(),
        best_index: None,
        trials_run: 0,
        elapsed_s: 0.0,
        baseline_tok_s: Some(50.0),
        best_tok_s: Some(75.0),
        summary: "test".into(),
    };
    assert_eq!(report.improvement_ratio(), Some(1.5));
}

#[test]
fn save_and_load_report() {
    let report = CampaignReport {
        results: Vec::new(),
        frontier: ParetoFrontier::default(),
        beliefs: BeliefGraph::new(),
        best_index: None,
        trials_run: 5,
        elapsed_s: 12.5,
        baseline_tok_s: Some(40.0),
        best_tok_s: Some(60.0),
        summary: "test save".into(),
    };
    let scratch = tempfile::tempdir().unwrap();
    let tmp = scratch.path().join("campaign.json");
    save_campaign_report(&report, &tmp).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert!(content.contains("trials_run"));
    assert!(content.contains("12.5"));
    let _ = std::fs::remove_file(&tmp);
}
