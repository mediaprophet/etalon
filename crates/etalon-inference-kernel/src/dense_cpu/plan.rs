use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseRequirements {
    pub kv_plane: usize,
    pub kv: usize,
    pub hidden: usize,
    pub normed: usize,
    pub norm_weight: usize,
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub attention: usize,
    pub projection: usize,
    pub gate: usize,
    pub up: usize,
    pub scores: usize,
    pub logits: usize,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct DensePlan<'a, H> {
    pub(super) model: DenseModel<'a, H>,
    pub(super) required: DenseRequirements,
}

impl<'a, H: Copy> DensePlan<'a, H> {
    /// Validates dimensions and memory sizes, not model-family interpretation.
    /// Missing tensors and numerical failures retain their execution-time order.
    pub fn new(model: DenseModel<'a, H>, byte_budget: usize) -> Result<Self, DensePlanError> {
        let c = model.config;
        if model.layers.is_empty() {
            return Err(DensePlanError::EmptyLayers);
        }
        if c.embedding == 0
            || c.ffn == 0
            || c.heads == 0
            || c.kv_heads == 0
            || c.head_dim == 0
            || c.max_context == 0
            || c.vocabulary == 0
            || c.max_context > u32::MAX as usize
            || c.vocabulary > u32::MAX as usize
            || c.heads.checked_mul(c.head_dim) != Some(c.embedding)
            || !c.heads.is_multiple_of(c.kv_heads)
        {
            return Err(DensePlanError::InvalidDimensions);
        }
        let width = c
            .kv_heads
            .checked_mul(c.head_dim)
            .ok_or(DensePlanError::ArithmeticOverflow)?;
        let plane = model
            .layers
            .len()
            .checked_mul(c.max_context)
            .and_then(|n| n.checked_mul(width))
            .ok_or(DensePlanError::ArithmeticOverflow)?;
        let mut required = DenseRequirements {
            kv_plane: plane,
            kv: plane
                .checked_mul(2)
                .ok_or(DensePlanError::ArithmeticOverflow)?,
            hidden: c.embedding,
            normed: c.embedding,
            norm_weight: c.embedding,
            q: c.embedding,
            k: width,
            v: width,
            attention: c.embedding,
            projection: c.embedding.max(c.ffn),
            gate: c.ffn,
            up: c.ffn,
            scores: c.max_context,
            logits: c.vocabulary,
            bytes: 0,
        };
        required.bytes =
            sum_bytes(&required.lengths()).ok_or(DensePlanError::ArithmeticOverflow)?;
        if required.bytes > byte_budget {
            return Err(DensePlanError::ByteBudget {
                required: required.bytes,
                available: byte_budget,
            });
        }
        Ok(Self { model, required })
    }
    pub fn requirements(&self) -> DenseRequirements {
        self.required
    }
    pub fn config(&self) -> DenseConfig {
        self.model.config
    }
    pub fn layer_count(&self) -> usize {
        self.model.layers.len()
    }
}

fn sum_bytes(lengths: &[usize; 13]) -> Option<usize> {
    lengths
        .iter()
        .try_fold(0usize, |n, length| n.checked_add(*length))?
        .checked_mul(core::mem::size_of::<f32>())
}

impl DenseRequirements {
    fn lengths(&self) -> [usize; 13] {
        [
            self.kv,
            self.hidden,
            self.normed,
            self.norm_weight,
            self.q,
            self.k,
            self.v,
            self.attention,
            self.projection,
            self.gate,
            self.up,
            self.scores,
            self.logits,
        ]
    }
}

impl<'a> DenseWorkspace<'a> {
    pub(super) fn admit(
        self,
        r: DenseRequirements,
        budget: usize,
    ) -> Result<Self, DenseWorkspaceError> {
        let actual = [
            self.kv.len(),
            self.hidden.len(),
            self.normed.len(),
            self.norm_weight.len(),
            self.q.len(),
            self.k.len(),
            self.v.len(),
            self.attention.len(),
            self.projection.len(),
            self.gate.len(),
            self.up.len(),
            self.scores.len(),
            self.logits.len(),
        ];
        let bytes = sum_bytes(&actual).ok_or(DenseWorkspaceError::ArithmeticOverflow)?;
        if bytes > budget {
            return Err(DenseWorkspaceError::ByteBudget {
                required: bytes,
                available: budget,
            });
        }
        let fields = [
            DenseBuffer::Kv,
            DenseBuffer::Hidden,
            DenseBuffer::Normed,
            DenseBuffer::NormWeight,
            DenseBuffer::Q,
            DenseBuffer::K,
            DenseBuffer::V,
            DenseBuffer::Attention,
            DenseBuffer::Projection,
            DenseBuffer::Gate,
            DenseBuffer::Up,
            DenseBuffer::Scores,
            DenseBuffer::Logits,
        ];
        for ((available, required), buffer) in actual.into_iter().zip(r.lengths()).zip(fields) {
            if available < required {
                return Err(DenseWorkspaceError::Capacity {
                    buffer,
                    required,
                    available,
                });
            }
        }
        Ok(Self {
            kv: &mut self.kv[..r.kv],
            hidden: &mut self.hidden[..r.hidden],
            normed: &mut self.normed[..r.normed],
            norm_weight: &mut self.norm_weight[..r.norm_weight],
            q: &mut self.q[..r.q],
            k: &mut self.k[..r.k],
            v: &mut self.v[..r.v],
            attention: &mut self.attention[..r.attention],
            projection: &mut self.projection[..r.projection],
            gate: &mut self.gate[..r.gate],
            up: &mut self.up[..r.up],
            scores: &mut self.scores[..r.scores],
            logits: &mut self.logits[..r.logits],
        })
    }
}
