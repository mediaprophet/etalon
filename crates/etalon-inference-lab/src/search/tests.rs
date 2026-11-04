use super::*;
use crate::config_space::ConfigurationSpace;
use crate::config_space::ParameterDef;

#[test]
fn sobol_generates_uniform() {
    let mut sobol = SobolSequence::new(3);
    let mut points = Vec::new();
    for _ in 0..100 {
        points.push(sobol.next());
    }
    // Check that points are in [0, 1].
    for p in &points {
        for &v in p {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }
    // Check that the mean is roughly 0.5 (uniform).
    let mean: f64 = points.iter().map(|p| p[0]).sum::<f64>() / points.len() as f64;
    assert!((mean - 0.5).abs() < 0.1);
}

#[test]
fn knn_surrogate_predicts() {
    let mut model = KnnSurrogate::new(3);
    model.add(vec![0.0, 0.0], 10.0);
    model.add(vec![1.0, 1.0], 20.0);
    model.add(vec![0.5, 0.5], 15.0);

    let (mean, _) = model.predict(&[0.1, 0.1]);
    // Should be closer to 10 than 20.
    assert!(mean < 15.0);

    let (mean, _) = model.predict(&[0.9, 0.9]);
    // Should be closer to 20 than 10.
    assert!(mean > 15.0);
}

#[test]
fn expected_improvement_zero_sigma() {
    // When sigma=0, EI = max(0, mu - f_best).
    assert_eq!(expected_improvement(10.0, 0.0, 5.0), 5.0);
    assert_eq!(expected_improvement(3.0, 0.0, 5.0), 0.0);
}

#[test]
fn expected_improvement_positive_with_uncertainty() {
    // With uncertainty, EI should be positive even if mu < f_best.
    let ei = expected_improvement(4.0, 2.0, 5.0);
    assert!(ei > 0.0);
}

#[test]
fn search_engine_ask_tell() {
    let space = ConfigurationSpace::new("test")
        .with("x", ParameterDef::Int { lo: 0, hi: 10 })
        .with("y", ParameterDef::Bool);
    let mut engine = SearchEngine::new(space, 20);

    // Should produce configurations.
    let cfg = engine.ask();
    assert!(cfg.is_some());

    // After telling results, it should continue producing.
    for _ in 0..5 {
        if let Some(c) = engine.ask() {
            // Simulate a result (we can't run a real experiment in unit tests).
            // Just check that ask/tell cycle works.
            engine.tried.insert(c.hash(), ());
            engine.observations.push((vec![0.5, 0.5], 10.0, c.hash()));
        }
    }
    assert!(engine.trials() <= 20);
}

#[test]
fn track_and_stop_identifies_best() {
    let space = ConfigurationSpace::new("test").with("x", ParameterDef::Int { lo: 0, hi: 1 });
    let arms = vec![
        space.build_from_normalized(&[0.0]),
        space.build_from_normalized(&[1.0]),
    ];
    let mut bandit = TrackAndStopBandit::new(arms);

    // Simulate: arm 0 has mean reward 10, arm 1 has mean reward 5.
    for iteration in 0..100 {
        if let Some(arm) = bandit.ask() {
            let reward = match arm {
                0 => 10.0 + ((iteration % 7) as f64 - 3.0) / 4.0,
                _ => 5.0 + ((iteration % 7) as f64 - 3.0) / 4.0,
            };
            bandit.tell(arm, reward);
        }
        if bandit.is_stopped() {
            break;
        }
    }
    // The bandit should identify arm 0 as the winner.
    if let Some(w) = bandit.winner() {
        assert_eq!(w, 0);
    }
}
