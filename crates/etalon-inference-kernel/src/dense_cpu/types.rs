#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseConfig {
    pub embedding: usize,
    pub ffn: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_context: usize,
    pub vocabulary: usize,
    pub rope_base: f32,
    pub rope_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorBinding<H> {
    pub handle: H,
    pub input_width: usize,
    pub output_rows: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DenseLayer<H> {
    pub attn_norm: Option<TensorBinding<H>>,
    pub q: Option<TensorBinding<H>>,
    pub k: Option<TensorBinding<H>>,
    pub v: Option<TensorBinding<H>>,
    pub output: Option<TensorBinding<H>>,
    pub ffn_norm: Option<TensorBinding<H>>,
    pub gate: Option<TensorBinding<H>>,
    pub up: Option<TensorBinding<H>>,
    pub down: Option<TensorBinding<H>>,
}

#[derive(Clone, Copy, Debug)]
pub struct DenseModel<'a, H> {
    pub config: DenseConfig,
    pub layers: &'a [DenseLayer<H>],
    pub embedding: TensorBinding<H>,
    pub output_norm: Option<TensorBinding<H>>,
    pub output: Option<TensorBinding<H>>,
}

/// Implementations must decode the bound immutable tensors and allocate no hot
/// storage. Calls receive the original used buffer lengths, excluding spare capacity.
/// The trait is an execution boundary, not authentication of an implementation.
pub trait TensorKernels<H: Copy> {
    type Error;
    fn decode_row_into(
        &mut self,
        tensor: TensorBinding<H>,
        row: usize,
        out: &mut [f32],
    ) -> Result<usize, Self::Error>;
    fn gemv_into(
        &mut self,
        tensor: TensorBinding<H>,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct DenseWorkspace<'a> {
    pub kv: &'a mut [f32],
    pub hidden: &'a mut [f32],
    pub normed: &'a mut [f32],
    pub norm_weight: &'a mut [f32],
    pub q: &'a mut [f32],
    pub k: &'a mut [f32],
    pub v: &'a mut [f32],
    pub attention: &'a mut [f32],
    pub projection: &'a mut [f32],
    pub gate: &'a mut [f32],
    pub up: &'a mut [f32],
    pub scores: &'a mut [f32],
    pub logits: &'a mut [f32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseBuffer {
    Kv,
    Hidden,
    Normed,
    NormWeight,
    Q,
    K,
    V,
    Attention,
    Projection,
    Gate,
    Up,
    Scores,
    Logits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensePlanError {
    InvalidDimensions,
    EmptyLayers,
    ArithmeticOverflow,
    ByteBudget { required: usize, available: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseWorkspaceError {
    Capacity {
        buffer: DenseBuffer,
        required: usize,
        available: usize,
    },
    ArithmeticOverflow,
    ByteBudget {
        required: usize,
        available: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseRole {
    Embedding,
    AttentionNorm,
    Q,
    K,
    V,
    AttentionDimensions,
    AttentionSoftmax,
    AttentionOutput,
    AttentionOutputDimensions,
    FfnNorm,
    Gate,
    Up,
    Down,
    FfnDimensions,
    DownDimensions,
    OutputNorm,
    Output,
    OutputArgmax,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DenseError<E> {
    ContextExceeded {
        position: usize,
        max_context: usize,
    },
    Workspace(DenseWorkspaceError),
    MissingTensor {
        layer: Option<usize>,
        role: DenseRole,
    },
    Kernel {
        layer: Option<usize>,
        role: DenseRole,
    },
    DecodeCount {
        layer: Option<usize>,
        role: DenseRole,
        expected: usize,
        actual: usize,
    },
    Tensor {
        layer: Option<usize>,
        role: DenseRole,
        source: E,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseStep {
    pub token_id: u32,
    pub max_logit: f32,
}

impl core::fmt::Display for DensePlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dense plan: {self:?}")
    }
}
impl core::error::Error for DensePlanError {}
impl core::fmt::Display for DenseWorkspaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dense workspace: {self:?}")
    }
}
impl core::error::Error for DenseWorkspaceError {}
impl<E: core::fmt::Debug> core::fmt::Display for DenseError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dense execution: {self:?}")
    }
}
impl<E: core::fmt::Debug> core::error::Error for DenseError<E> {}
