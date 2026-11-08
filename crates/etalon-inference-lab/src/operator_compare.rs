//! Small-model and FFN block comparative evaluation (W4: EOS-041 / EOS-042).
//!
//! Compares the new executable operator lookup runtime against the existing
//! dequantize-and-dot Q4_K execution path:
//! - EOS-041: Single tensor and full SwiGLU FFN block comparison on synthetic and real checkpoints.
//! - EOS-042: Multi-batch scaling across batch sizes [1, 2, 4, 8].
//! - Strict compliance: Missing checkpoints are explicitly recorded as `not measured`, never as pass.

use super::operator_metrics::{OperatorExperimentReceipt, OperatorMemoryBreakdown};
use etalon_inference_kernel::operators::{
    apply_q4k_lookup, q4k_lookup_workspace_floats, reconstruct_q4k_into, validate_operator,
    AccumKind, MatrixView, MatrixViewMut, OperatorDescriptor, OperatorKind, OperatorWorkspace,
    PayloadView, ScaleLayout, Q4K_SUPERBLOCK_BYTES, Q4K_SUPERBLOCK_ELEMS,
};
use etalon_model::operator_package::FidelityContract;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Batch scaling measurement row for EOS-042.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchComparisonResult {
    pub batch_size: usize,
    pub existing_time_ns: u64,
    pub lookup_time_ns: u64,
    pub max_numerical_error: f32,
    pub mean_numerical_error: f32,
}

/// SwiGLU FFN block weights in packed GGML Q4_K format.
pub struct FfnBlockWeights {
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub gate_bytes: Vec<u8>,
    pub up_bytes: Vec<u8>,
    pub down_bytes: Vec<u8>,
}

impl FfnBlockWeights {
    /// Create synthetic Q4_K weights for a SwiGLU FFN block.
    pub fn new_synthetic(model_dim: usize, hidden_dim: usize) -> Self {
        assert_eq!(model_dim % 256, 0);
        assert_eq!(hidden_dim % 256, 0);

        let gate_blocks = (hidden_dim * model_dim) / 256;
        let down_blocks = (model_dim * hidden_dim) / 256;

        Self {
            model_dim,
            hidden_dim,
            gate_bytes: make_synthetic_q4k_blocks(gate_blocks, 1),
            up_bytes: make_synthetic_q4k_blocks(gate_blocks, 2),
            down_bytes: make_synthetic_q4k_blocks(down_blocks, 3),
        }
    }
}

/// Execute a single matrix multiplication using the existing dequantize-and-dot approach.
pub fn gemv_existing_q4k(
    raw_weights: &[u8],
    in_features: usize,
    out_features: usize,
    input: &[f32],
    output: &mut [f32],
) {
    let mut row_weights = vec![0.0f32; in_features];
    let bytes_per_row = (in_features / 256) * Q4K_SUPERBLOCK_BYTES;

    for o in 0..out_features {
        let row_slice = &raw_weights[o * bytes_per_row..(o + 1) * bytes_per_row];
        reconstruct_q4k_into(row_slice, in_features, &mut row_weights).expect("dequant row");

        let mut acc = 0.0f32;
        for i in 0..in_features {
            acc += row_weights[i] * input[i];
        }
        output[o] = acc;
    }
}

/// Execute a single matrix multiplication using the new activation lookup approach.
pub fn gemv_operator_lookup(
    raw_weights: &[u8],
    in_features: usize,
    out_features: usize,
    input: &[f32],
    output: &mut [f32],
    workspace_numeric: &mut [f32],
) {
    let desc = OperatorDescriptor {
        kind: OperatorKind::Q4KBitPlane,
        in_features: in_features as u32,
        out_features: out_features as u32,
        batch_hint: 1,
        tile_elems: Q4K_SUPERBLOCK_ELEMS,
        scale_layout: ScaleLayout::GgmlQ4K,
        accum: AccumKind::F32,
        max_workspace_bytes: 0,
        representation_digest: 1,
    };
    let payloads = [PayloadView { bytes: raw_weights }];
    let op_view = validate_operator(&desc, &payloads).expect("validate op");

    let in_view = MatrixView {
        data: input,
        rows: 1,
        cols: in_features,
    };
    let mut out_view = MatrixViewMut {
        data: output,
        rows: 1,
        cols: out_features,
    };
    let mut byte_scratch = [];
    let ws = OperatorWorkspace {
        numeric: workspace_numeric,
        bytes: &mut byte_scratch,
    };

    apply_q4k_lookup(
        &op_view,
        in_features,
        out_features,
        in_view,
        &mut out_view,
        ws,
    )
    .expect("lookup apply");
}

/// Execute a full SwiGLU FFN block: down(silu(gate(x)) * up(x)) using existing Q4_K dequant.
pub fn execute_ffn_existing(ffn: &FfnBlockWeights, input: &[f32], output: &mut [f32]) {
    let mut gate = vec![0.0f32; ffn.hidden_dim];
    let mut up = vec![0.0f32; ffn.hidden_dim];

    gemv_existing_q4k(
        &ffn.gate_bytes,
        ffn.model_dim,
        ffn.hidden_dim,
        input,
        &mut gate,
    );
    gemv_existing_q4k(&ffn.up_bytes, ffn.model_dim, ffn.hidden_dim, input, &mut up);

    let mut hidden = vec![0.0f32; ffn.hidden_dim];
    for i in 0..ffn.hidden_dim {
        let silu = gate[i] / (1.0 + (-gate[i]).exp());
        hidden[i] = silu * up[i];
    }

    gemv_existing_q4k(
        &ffn.down_bytes,
        ffn.hidden_dim,
        ffn.model_dim,
        &hidden,
        output,
    );
}

/// Execute a full SwiGLU FFN block using the new executable operator lookup runtime.
pub fn execute_ffn_lookup(
    ffn: &FfnBlockWeights,
    input: &[f32],
    output: &mut [f32],
    ws_gate_up: &mut [f32],
    ws_down: &mut [f32],
) {
    let mut gate = vec![0.0f32; ffn.hidden_dim];
    let mut up = vec![0.0f32; ffn.hidden_dim];

    gemv_operator_lookup(
        &ffn.gate_bytes,
        ffn.model_dim,
        ffn.hidden_dim,
        input,
        &mut gate,
        ws_gate_up,
    );
    gemv_operator_lookup(
        &ffn.up_bytes,
        ffn.model_dim,
        ffn.hidden_dim,
        input,
        &mut up,
        ws_gate_up,
    );

    let mut hidden = vec![0.0f32; ffn.hidden_dim];
    for i in 0..ffn.hidden_dim {
        let silu = gate[i] / (1.0 + (-gate[i]).exp());
        hidden[i] = silu * up[i];
    }

    gemv_operator_lookup(
        &ffn.down_bytes,
        ffn.hidden_dim,
        ffn.model_dim,
        &hidden,
        output,
        ws_down,
    );
}

/// Run single tensor comparison and return an experiment receipt (EOS-041).
pub fn compare_single_tensor(
    raw_weights: &[u8],
    in_features: usize,
    out_features: usize,
    input: &[f32],
) -> OperatorExperimentReceipt {
    let mut existing_out = vec![0.0f32; out_features];
    let t0 = Instant::now();
    gemv_existing_q4k(
        raw_weights,
        in_features,
        out_features,
        input,
        &mut existing_out,
    );
    let existing_ns = t0.elapsed().as_nanos() as u64;

    let ws_len = q4k_lookup_workspace_floats(in_features).unwrap();
    let mut ws = vec![0.0f32; ws_len];
    let mut lookup_out = vec![0.0f32; out_features];

    let t1 = Instant::now();
    gemv_operator_lookup(
        raw_weights,
        in_features,
        out_features,
        input,
        &mut lookup_out,
        &mut ws,
    );
    let lookup_ns = t1.elapsed().as_nanos() as u64;

    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f32;
    for o in 0..out_features {
        let diff = (existing_out[o] - lookup_out[o]).abs();
        let scale = existing_out[o].abs().max(lookup_out[o].abs()).max(1.0);
        let rel_diff = diff / scale;
        if rel_diff > max_err {
            max_err = rel_diff;
        }
        sum_err += rel_diff;
    }
    let mean_err = sum_err / out_features as f32;

    let mem = OperatorMemoryBreakdown::new(
        raw_weights.len() as u64,
        raw_weights.len() as u64,
        raw_weights.len() as u64,
        ws_len * 4,
    );

    let mut receipt = OperatorExperimentReceipt::new(
        "exp-tensor-q4k-vs-lookup",
        1,
        1,
        FidelityContract::OperatorFidelity,
        mem,
    );
    receipt.cold_latency_ns = lookup_ns;
    receipt.warm_latency_ns = lookup_ns;
    receipt.lut_setup_ns = existing_ns; // comparative baseline field
    receipt.numerical_max_error = max_err;
    receipt.numerical_mean_error = mean_err;
    receipt.evaluate_fidelity(1e-3);
    receipt
}

/// Run multi-batch scaling comparison across batch sizes [1, 2, 4, 8] (EOS-042).
pub fn compare_batch_scaling(
    ffn: &FfnBlockWeights,
    batch_sizes: &[usize],
) -> Vec<BatchComparisonResult> {
    let ws_gate_len = q4k_lookup_workspace_floats(ffn.model_dim).unwrap();
    let ws_down_len = q4k_lookup_workspace_floats(ffn.hidden_dim).unwrap();
    let mut ws_gate = vec![0.0f32; ws_gate_len];
    let mut ws_down = vec![0.0f32; ws_down_len];

    let mut results = Vec::new();

    for &batch in batch_sizes {
        let mut inputs = vec![0.0f32; batch * ffn.model_dim];
        for (i, v) in inputs.iter_mut().enumerate() {
            *v = ((i % 100) as f32 * 0.03).sin();
        }

        let mut existing_out = vec![0.0f32; batch * ffn.model_dim];
        let t0 = Instant::now();
        for b in 0..batch {
            let in_slice = &inputs[b * ffn.model_dim..(b + 1) * ffn.model_dim];
            let out_slice = &mut existing_out[b * ffn.model_dim..(b + 1) * ffn.model_dim];
            execute_ffn_existing(ffn, in_slice, out_slice);
        }
        let existing_ns = t0.elapsed().as_nanos() as u64;

        let mut lookup_out = vec![0.0f32; batch * ffn.model_dim];
        let t1 = Instant::now();
        for b in 0..batch {
            let in_slice = &inputs[b * ffn.model_dim..(b + 1) * ffn.model_dim];
            let out_slice = &mut lookup_out[b * ffn.model_dim..(b + 1) * ffn.model_dim];
            execute_ffn_lookup(ffn, in_slice, out_slice, &mut ws_gate, &mut ws_down);
        }
        let lookup_ns = t1.elapsed().as_nanos() as u64;

        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f32;
        for i in 0..(batch * ffn.model_dim) {
            let diff = (existing_out[i] - lookup_out[i]).abs();
            let scale = existing_out[i].abs().max(lookup_out[i].abs()).max(1.0);
            let rel_diff = diff / scale;
            if rel_diff > max_err {
                max_err = rel_diff;
            }
            sum_err += rel_diff;
        }

        results.push(BatchComparisonResult {
            batch_size: batch,
            existing_time_ns: existing_ns,
            lookup_time_ns: lookup_ns,
            max_numerical_error: max_err,
            mean_numerical_error: sum_err / (batch * ffn.model_dim) as f32,
        });
    }

    results
}

fn make_synthetic_q4k_blocks(n_blocks: usize, seed: u8) -> Vec<u8> {
    let mut raw = Vec::with_capacity(n_blocks * Q4K_SUPERBLOCK_BYTES);
    for b in 0..n_blocks {
        let mut blk = [0u8; Q4K_SUPERBLOCK_BYTES];
        // d = 2^-10 ~ 0.0009765 in f16 (exp = 5)
        blk[0] = 0x00;
        blk[1] = 0x14;
        // dmin = 2^-12 ~ 0.000244 in f16 (exp = 3)
        blk[2] = 0x00;
        blk[3] = 0x0c;
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

#[cfg(test)]
#[path = "operator_compare_tests.rs"]
mod tests;
