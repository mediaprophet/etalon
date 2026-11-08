use super::types::*;
use etalon_inference_kernel::operators::{
    apply_q4k_lookup, validate_operator, AccumKind, MatrixView, MatrixViewMut, OperatorDescriptor,
    OperatorError, OperatorKind, OperatorWorkspace, PayloadView, ScaleLayout, Q4K_SUPERBLOCK_ELEMS,
};

/// SiLU (Swish) activation function: x * sigmoid(x).
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl LowRankDelta {
    /// Apply delta_W * x into out: out[i] += sum_r A[i, r] * (sum_j B[r, j] * x[j]).
    pub fn apply_add(&self, input: &[f32], rank_buf: &mut [f32], out: &mut [f32]) {
        assert!(rank_buf.len() >= self.rank);
        assert!(input.len() >= self.in_dim);
        assert!(out.len() >= self.out_dim);

        // B * x -> rank_buf
        for r in 0..self.rank {
            let row = &self.b[r * self.in_dim..(r + 1) * self.in_dim];
            let mut acc = 0.0f32;
            for j in 0..self.in_dim {
                acc += row[j] * input[j];
            }
            rank_buf[r] = acc;
        }

        // A * rank_buf -> add to out
        for i in 0..self.out_dim {
            let row = &self.a[i * self.rank..(i + 1) * self.rank];
            let mut acc = 0.0f32;
            for r in 0..self.rank {
                acc += row[r] * rank_buf[r];
            }
            out[i] += acc;
        }
    }
}

/// Execute a single matrix multiplication using Q4_K lookup.
pub fn gemv_q4k(
    raw_weights: &[u8],
    in_features: usize,
    out_features: usize,
    input: &[f32],
    output: &mut [f32],
    workspace_numeric: &mut [f32],
) -> Result<(), OperatorError> {
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
    let op_view = validate_operator(&desc, &payloads)?;

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
}

impl<'a> IndependentExpertWeights<'a> {
    /// Evaluate one SwiGLU expert: y = down(silu(gate(x)) * up(x)).
    pub fn evaluate(
        &self,
        input: &[f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        hidden_buf: &mut [f32],
        out: &mut [f32],
        ws_gate_up: &mut [f32],
        ws_down: &mut [f32],
    ) -> Result<(), OperatorError> {
        if input.len() < self.model_dim || out.len() < self.model_dim {
            return Err(OperatorError::WorkspaceTooSmall);
        }
        if gate_buf.len() < self.hidden_dim
            || up_buf.len() < self.hidden_dim
            || hidden_buf.len() < self.hidden_dim
        {
            return Err(OperatorError::WorkspaceTooSmall);
        }

        gemv_q4k(
            self.gate_bytes,
            self.model_dim,
            self.hidden_dim,
            input,
            gate_buf,
            ws_gate_up,
        )?;
        gemv_q4k(
            self.up_bytes,
            self.model_dim,
            self.hidden_dim,
            input,
            up_buf,
            ws_gate_up,
        )?;

        for i in 0..self.hidden_dim {
            hidden_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }

        gemv_q4k(
            self.down_bytes,
            self.hidden_dim,
            self.model_dim,
            hidden_buf,
            out,
            ws_down,
        )?;
        Ok(())
    }
}

impl ClusteredMoEOperator {
    /// Dispatches.
    pub fn dispatch(
        &self,
        routed_experts: &[RoutedExpert],
        input: &[f32],
        output: &mut [f32],
        scratch: &mut ClusteredMoeScratch<'_>,
    ) -> Result<(), OperatorError> {
        dispatch_clustered_moe_step(&self.clusters, routed_experts, input, output, scratch)
    }
}

/// Execute clustered MoE dispatch step with down-projection pre-accumulation (EOS-071).
///
/// Mathematical identity: y_c = D_c (sum_{e in c} g_e h_e) + sum_{e in c} g_e delta_D_e h_e,
/// where h_e = SiLU(G_e x) * (U_e x) stays strictly expert-specific.
pub fn dispatch_clustered_moe_step(
    clusters: &[ClusterAnchor],
    routed_experts: &[RoutedExpert],
    input: &[f32],
    output: &mut [f32],
    scratch: &mut ClusteredMoeScratch<'_>,
) -> Result<(), OperatorError> {
    output.fill(0.0);

    for cluster in clusters {
        let mut cluster_has_active = false;
        scratch.cluster_accum_hidden[..cluster.hidden_dim].fill(0.0);

        for routed in routed_experts {
            let expert_pos = cluster
                .expert_ids
                .iter()
                .position(|&id| id == routed.expert_id);
            let local_idx = match expert_pos {
                Some(idx) => idx,
                None => continue, // Expert not in this cluster
            };

            cluster_has_active = true;

            // 1. Gate projection: G_e x = G_c x + delta_G_e x
            gemv_q4k(
                &cluster.base_gate,
                cluster.model_dim,
                cluster.hidden_dim,
                input,
                &mut scratch.gate_buf[..cluster.hidden_dim],
                scratch.ws_gate_up,
            )?;
            if let Some(ref delta_g) = cluster.expert_gate_deltas[local_idx] {
                delta_g.apply_add(
                    input,
                    scratch.rank_scratch,
                    &mut scratch.gate_buf[..cluster.hidden_dim],
                );
            }

            // 2. Up projection: U_e x = U_c x + delta_U_e x
            gemv_q4k(
                &cluster.base_up,
                cluster.model_dim,
                cluster.hidden_dim,
                input,
                &mut scratch.up_buf[..cluster.hidden_dim],
                scratch.ws_gate_up,
            )?;
            if let Some(ref delta_u) = cluster.expert_up_deltas[local_idx] {
                delta_u.apply_add(
                    input,
                    scratch.rank_scratch,
                    &mut scratch.up_buf[..cluster.hidden_dim],
                );
            }

            // 3. Nonlinear activation: h_e = SiLU(G_e x) * (U_e x) — strictly expert-specific!
            for i in 0..cluster.hidden_dim {
                let h_e = silu(scratch.gate_buf[i]) * scratch.up_buf[i];
                scratch.hidden_buf[i] = h_e;
                // Accumulate weighted hidden state for down projection: sum_{e in c} g_e h_e
                scratch.cluster_accum_hidden[i] += routed.weight * h_e;
            }

            // 4. Down delta projection if present: add g_e * delta_D_e * h_e directly to output
            if let Some(ref delta_d) = cluster.expert_down_deltas[local_idx] {
                // Reuse the caller-owned down buffer.  The shared down projection below
                // overwrites it after all expert-specific deltas have been accumulated.
                // Keeping this temporary in supplied scratch is essential: this function
                // is called once per decode step and must never allocate.
                scratch.cluster_down_out[..cluster.model_dim].fill(0.0);
                delta_d.apply_add(
                    &scratch.hidden_buf[..cluster.hidden_dim],
                    scratch.rank_scratch,
                    &mut scratch.cluster_down_out[..cluster.model_dim],
                );
                for m in 0..cluster.model_dim {
                    output[m] += routed.weight * scratch.cluster_down_out[m];
                }
            }
        }

        if cluster_has_active {
            // 5. Shared cluster down projection: D_c (sum_{e in c} g_e h_e) — computed ONCE per cluster!
            gemv_q4k(
                &cluster.base_down,
                cluster.hidden_dim,
                cluster.model_dim,
                &scratch.cluster_accum_hidden[..cluster.hidden_dim],
                &mut scratch.cluster_down_out[..cluster.model_dim],
                scratch.ws_down,
            )?;

            for m in 0..cluster.model_dim {
                output[m] += scratch.cluster_down_out[m];
            }
        }
    }

    Ok(())
}
