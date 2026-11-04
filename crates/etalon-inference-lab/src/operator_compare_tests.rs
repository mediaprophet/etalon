
    use super::*;

    #[test]
    fn test_synthetic_tensor_comparison() {
        let in_features = 256;
        let out_features = 256;
        let raw_weights = make_synthetic_q4k_blocks((out_features * in_features) / 256, 7);

        let mut input = vec![0.0f32; in_features];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as f32 * 0.02).cos();
        }

        let receipt = compare_single_tensor(&raw_weights, in_features, out_features, &input);
        assert!(receipt.fidelity_satisfied, "tensor fidelity satisfied");
        assert!(receipt.numerical_max_error < 1e-3, "max error < 1e-3");
    }

    #[test]
    fn test_synthetic_ffn_block_comparison() {
        let model_dim = 256;
        let hidden_dim = 256;
        let ffn = FfnBlockWeights::new_synthetic(model_dim, hidden_dim);

        let mut input = vec![0.0f32; model_dim];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as f32 * 0.05).sin();
        }

        let mut existing_out = vec![0.0f32; model_dim];
        execute_ffn_existing(&ffn, &input, &mut existing_out);

        let ws_gate_len = q4k_lookup_workspace_floats(model_dim).unwrap();
        let ws_down_len = q4k_lookup_workspace_floats(hidden_dim).unwrap();
        let mut ws_gate = vec![0.0f32; ws_gate_len];
        let mut ws_down = vec![0.0f32; ws_down_len];

        let mut lookup_out = vec![0.0f32; model_dim];
        execute_ffn_lookup(&ffn, &input, &mut lookup_out, &mut ws_gate, &mut ws_down);

        for i in 0..model_dim {
            let diff = (existing_out[i] - lookup_out[i]).abs();
            let scale = existing_out[i].abs().max(lookup_out[i].abs()).max(1.0);
            let rel_diff = diff / scale;
            assert!(
                rel_diff < 1e-3,
                "FFN output element {i} mismatch: existing={} lookup={} rel_diff={rel_diff}",
                existing_out[i],
                lookup_out[i]
            );
        }
    }

    #[test]
    fn test_batch_scaling_1_2_4_8() {
        let model_dim = 256;
        let hidden_dim = 256;
        let ffn = FfnBlockWeights::new_synthetic(model_dim, hidden_dim);

        let batch_results = compare_batch_scaling(&ffn, &[1, 2, 4, 8]);
        assert_eq!(batch_results.len(), 4);

        for res in &batch_results {
            assert!(res.max_numerical_error < 1e-3);
            assert!(res.existing_time_ns > 0);
            assert!(res.lookup_time_ns > 0);
        }
    }

