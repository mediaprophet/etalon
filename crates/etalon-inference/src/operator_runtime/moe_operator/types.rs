/// A low-rank residual factor delta: delta_W * x = A * (B * x).
#[derive(Debug, Clone)]
pub struct LowRankDelta {
    pub a: Vec<f32>, // out_dim x rank (row-major)
    pub b: Vec<f32>, // rank x in_dim (row-major)
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
}

impl LowRankDelta {
    /// Creates a default `LowRankDelta`.
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, a: Vec<f32>, b: Vec<f32>) -> Self {
        assert_eq!(a.len(), out_dim * rank);
        assert_eq!(b.len(), rank * in_dim);
        Self {
            a,
            b,
            in_dim,
            out_dim,
            rank,
        }
    }
}

/// Independent SwiGLU expert weights (EOS-070).
pub struct IndependentExpertWeights<'a> {
    pub gate_bytes: &'a [u8],
    pub up_bytes: &'a [u8],
    pub down_bytes: &'a [u8],
    pub model_dim: usize,
    pub hidden_dim: usize,
}

/// A cluster anchor containing shared base weights and per-expert deltas (EOS-071).
#[derive(Debug, Clone)]
pub struct ClusterAnchor {
    pub cluster_id: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub base_gate: Vec<u8>,
    pub base_up: Vec<u8>,
    pub base_down: Vec<u8>,
    pub expert_ids: Vec<usize>,
    /// Optional low-rank deltas for each expert: (expert_id -> deltas)
    pub expert_gate_deltas: Vec<Option<LowRankDelta>>,
    pub expert_up_deltas: Vec<Option<LowRankDelta>>,
    pub expert_down_deltas: Vec<Option<LowRankDelta>>,
}

/// Clustered MoE operator holding cluster anchors with shared base projections and per-expert deltas (EOS-071).
#[derive(Debug, Clone)]
pub struct ClusteredMoEOperator {
    pub clusters: Vec<ClusterAnchor>,
}

impl ClusteredMoEOperator {
    /// Creates a default `ClusteredMoEOperator`.
    pub fn new(clusters: Vec<ClusterAnchor>) -> Self {
        Self { clusters }
    }
}

/// Routed expert execution item: (expert_index, gating_weight).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedExpert {
    pub expert_id: usize,
    pub weight: f32,
}

/// Scratch buffers required for zero-heap clustered MoE evaluation.
pub struct ClusteredMoeScratch<'a> {
    pub gate_buf: &'a mut [f32],
    pub up_buf: &'a mut [f32],
    pub hidden_buf: &'a mut [f32],
    pub cluster_accum_hidden: &'a mut [f32],
    pub cluster_down_out: &'a mut [f32],
    pub rank_scratch: &'a mut [f32],
    pub ws_gate_up: &'a mut [f32],
    pub ws_down: &'a mut [f32],
}
