use etalon_inference_kernel::operators::q4k_lookup_workspace_floats;

use super::*;

fn make_synthetic_q4k_blocks(n_blocks: usize, seed: u8) -> Vec<u8> {
    let mut raw = Vec::with_capacity(n_blocks * 144);
    for b in 0..n_blocks {
        let mut blk = [0u8; 144];
        blk[0] = 0x00;
        blk[1] = 0x14; // d
        blk[2] = 0x00;
        blk[3] = 0x0c; // dmin
        for i in 4..16 {
            blk[i] = ((seed as usize + b + i) & 0x3f) as u8;
        }
        for i in 16..144 {
            blk[i] = seed.wrapping_add((b as u8).wrapping_add(i as u8));
        }
        raw.extend_from_slice(&blk);
    }
    raw
}

#[test]
fn test_independent_expert_operator_swiglu() {
    let model_dim = 256;
    let hidden_dim = 256;
    let gate = make_synthetic_q4k_blocks((hidden_dim * model_dim) / 256, 11);
    let up = make_synthetic_q4k_blocks((hidden_dim * model_dim) / 256, 12);
    let down = make_synthetic_q4k_blocks((model_dim * hidden_dim) / 256, 13);

    let expert = IndependentExpertWeights {
        gate_bytes: &gate,
        up_bytes: &up,
        down_bytes: &down,
        model_dim,
        hidden_dim,
    };

    let mut input = vec![0.0f32; model_dim];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32 * 0.03).cos();
    }

    let mut gate_buf = vec![0.0f32; hidden_dim];
    let mut up_buf = vec![0.0f32; hidden_dim];
    let mut hidden_buf = vec![0.0f32; hidden_dim];
    let mut out = vec![0.0f32; model_dim];

    let ws_gate_len = q4k_lookup_workspace_floats(model_dim).unwrap();
    let ws_down_len = q4k_lookup_workspace_floats(hidden_dim).unwrap();
    let mut ws_gate = vec![0.0f32; ws_gate_len];
    let mut ws_down = vec![0.0f32; ws_down_len];

    expert
        .evaluate(
            &input,
            &mut gate_buf,
            &mut up_buf,
            &mut hidden_buf,
            &mut out,
            &mut ws_gate,
            &mut ws_down,
        )
        .expect("expert eval");

    assert!(
        out.iter().any(|&x| x.abs() > 1e-5),
        "expert produced non-zero output"
    );
}

#[test]
fn test_clustered_moe_mathematical_equivalence() {
    let model_dim = 256;
    let hidden_dim = 256;
    let base_gate = make_synthetic_q4k_blocks((hidden_dim * model_dim) / 256, 21);
    let base_up = make_synthetic_q4k_blocks((hidden_dim * model_dim) / 256, 22);
    let base_down = make_synthetic_q4k_blocks((model_dim * hidden_dim) / 256, 23);

    let cluster = ClusterAnchor {
        cluster_id: 0,
        model_dim,
        hidden_dim,
        base_gate: base_gate.clone(),
        base_up: base_up.clone(),
        base_down: base_down.clone(),
        expert_ids: vec![0, 1],
        expert_gate_deltas: vec![None, None],
        expert_up_deltas: vec![None, None],
        expert_down_deltas: vec![None, None],
    };

    let mut input = vec![0.0f32; model_dim];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i % 64) as f32 * 0.05).sin();
    }

    let routed = [
        RoutedExpert {
            expert_id: 0,
            weight: 0.6,
        },
        RoutedExpert {
            expert_id: 1,
            weight: 0.4,
        },
    ];

    // 1. Independent evaluation
    let expert = IndependentExpertWeights {
        gate_bytes: &base_gate,
        up_bytes: &base_up,
        down_bytes: &base_down,
        model_dim,
        hidden_dim,
    };
    let ws_gate_len = q4k_lookup_workspace_floats(model_dim).unwrap();
    let ws_down_len = q4k_lookup_workspace_floats(hidden_dim).unwrap();
    let mut ws_gate = vec![0.0f32; ws_gate_len];
    let mut ws_down = vec![0.0f32; ws_down_len];
    let mut gb = vec![0.0f32; hidden_dim];
    let mut ub = vec![0.0f32; hidden_dim];
    let mut hb = vec![0.0f32; hidden_dim];
    let mut single_out = vec![0.0f32; model_dim];
    expert
        .evaluate(
            &input,
            &mut gb,
            &mut ub,
            &mut hb,
            &mut single_out,
            &mut ws_gate,
            &mut ws_down,
        )
        .unwrap();

    let mut expected_out = vec![0.0f32; model_dim];
    for m in 0..model_dim {
        // Both experts share base weights, so sum of weights (0.6 + 0.4 = 1.0) times single_out
        expected_out[m] = (routed[0].weight + routed[1].weight) * single_out[m];
    }

    // 2. Clustered evaluation with down-projection pre-accumulation
    let mut cluster_out = vec![0.0f32; model_dim];
    let mut c_gb = vec![0.0f32; hidden_dim];
    let mut c_ub = vec![0.0f32; hidden_dim];
    let mut c_hb = vec![0.0f32; hidden_dim];
    let mut c_accum = vec![0.0f32; hidden_dim];
    let mut c_down = vec![0.0f32; model_dim];
    let mut rank_s = vec![0.0f32; 16];
    let mut c_wsg = vec![0.0f32; ws_gate_len];
    let mut c_wsd = vec![0.0f32; ws_down_len];

    let mut scratch = ClusteredMoeScratch {
        gate_buf: &mut c_gb,
        up_buf: &mut c_ub,
        hidden_buf: &mut c_hb,
        cluster_accum_hidden: &mut c_accum,
        cluster_down_out: &mut c_down,
        rank_scratch: &mut rank_s,
        ws_gate_up: &mut c_wsg,
        ws_down: &mut c_wsd,
    };

    dispatch_clustered_moe_step(&[cluster], &routed, &input, &mut cluster_out, &mut scratch)
        .unwrap();

    for m in 0..model_dim {
        let diff = (expected_out[m] - cluster_out[m]).abs();
        let scale = expected_out[m].abs().max(cluster_out[m].abs()).max(1.0);
        let rel = diff / scale;
        assert!(
            rel < 1e-4,
            "clustered output element {m} mismatch: expected={} clustered={} rel={rel}",
            expected_out[m],
            cluster_out[m]
        );
    }
}

#[test]
fn test_clustered_moe_with_deltas() {
    let model_dim = 256;
    let hidden_dim = 256;
    let base_gate = make_synthetic_q4k_blocks((hidden_dim * model_dim) / 256, 31);
    let base_up = make_synthetic_q4k_blocks((hidden_dim * model_dim) / 256, 32);
    let base_down = make_synthetic_q4k_blocks((model_dim * hidden_dim) / 256, 33);

    let delta_gate = LowRankDelta::new(
        model_dim,
        hidden_dim,
        2,
        vec![0.01f32; hidden_dim * 2],
        vec![0.02f32; 2 * model_dim],
    );
    let delta_down = LowRankDelta::new(
        hidden_dim,
        model_dim,
        2,
        vec![0.005f32; model_dim * 2],
        vec![0.005f32; 2 * hidden_dim],
    );

    let cluster = ClusterAnchor {
        cluster_id: 0,
        model_dim,
        hidden_dim,
        base_gate,
        base_up,
        base_down,
        expert_ids: vec![0],
        expert_gate_deltas: vec![Some(delta_gate)],
        expert_up_deltas: vec![None],
        expert_down_deltas: vec![Some(delta_down)],
    };

    let input = vec![0.1f32; model_dim];
    let routed = [RoutedExpert {
        expert_id: 0,
        weight: 1.0,
    }];

    let ws_gate_len = q4k_lookup_workspace_floats(model_dim).unwrap();
    let ws_down_len = q4k_lookup_workspace_floats(hidden_dim).unwrap();
    let mut out = vec![0.0f32; model_dim];
    let mut gb = vec![0.0f32; hidden_dim];
    let mut ub = vec![0.0f32; hidden_dim];
    let mut hb = vec![0.0f32; hidden_dim];
    let mut accum = vec![0.0f32; hidden_dim];
    let mut c_down = vec![0.0f32; model_dim];
    let mut rank_s = vec![0.0f32; 16];
    let mut wsg = vec![0.0f32; ws_gate_len];
    let mut wsd = vec![0.0f32; ws_down_len];

    let mut scratch = ClusteredMoeScratch {
        gate_buf: &mut gb,
        up_buf: &mut ub,
        hidden_buf: &mut hb,
        cluster_accum_hidden: &mut accum,
        cluster_down_out: &mut c_down,
        rank_scratch: &mut rank_s,
        ws_gate_up: &mut wsg,
        ws_down: &mut wsd,
    };

    dispatch_clustered_moe_step(&[cluster], &routed, &input, &mut out, &mut scratch).unwrap();
    assert!(
        out.iter().any(|&v| v.abs() > 1e-5),
        "delta-augmented clustered MoE produced non-zero output"
    );
}

#[test]
fn test_clustered_moe_operator_wrapper() {
    let model_dim = 256;
    let hidden_dim = 256;
    let base_gate = make_synthetic_q4k_blocks((hidden_dim * model_dim) / 256, 41);
    let base_up = make_synthetic_q4k_blocks((hidden_dim * model_dim) / 256, 42);
    let base_down = make_synthetic_q4k_blocks((model_dim * hidden_dim) / 256, 43);

    let cluster = ClusterAnchor {
        cluster_id: 0,
        model_dim,
        hidden_dim,
        base_gate,
        base_up,
        base_down,
        expert_ids: vec![0],
        expert_gate_deltas: vec![None],
        expert_up_deltas: vec![None],
        expert_down_deltas: vec![None],
    };

    let op = ClusteredMoEOperator::new(vec![cluster]);
    let input = vec![0.2f32; model_dim];
    let routed = [RoutedExpert {
        expert_id: 0,
        weight: 0.8,
    }];

    let ws_gate_len = q4k_lookup_workspace_floats(model_dim).unwrap();
    let ws_down_len = q4k_lookup_workspace_floats(hidden_dim).unwrap();
    let mut out = vec![0.0f32; model_dim];
    let mut gb = vec![0.0f32; hidden_dim];
    let mut ub = vec![0.0f32; hidden_dim];
    let mut hb = vec![0.0f32; hidden_dim];
    let mut accum = vec![0.0f32; hidden_dim];
    let mut c_down = vec![0.0f32; model_dim];
    let mut rank_s = vec![0.0f32; 16];
    let mut wsg = vec![0.0f32; ws_gate_len];
    let mut wsd = vec![0.0f32; ws_down_len];

    let mut scratch = ClusteredMoeScratch {
        gate_buf: &mut gb,
        up_buf: &mut ub,
        hidden_buf: &mut hb,
        cluster_accum_hidden: &mut accum,
        cluster_down_out: &mut c_down,
        rank_scratch: &mut rank_s,
        ws_gate_up: &mut wsg,
        ws_down: &mut wsd,
    };

    op.dispatch(&routed, &input, &mut out, &mut scratch)
        .unwrap();
    assert!(out.iter().any(|&v| v.abs() > 1e-5));
}
