//! Cold, allocation-free admission of the complete Qwen3.5 layer schedule.
use super::WorkspaceLayout;
use etalon_model::inference::{
    moe_routing::AdmittedMoeRouting,
    qwen_attention::{QkNorm, Qwen35AttentionGeometry, Qwen35AttentionRoles},
    qwen_roles::{Projection, Qwen35Packing, Qwen35Roles},
    LayerTensors, ModelHyperparams, TensorIndex, TensorInfo, WeightEncoding, ARCH_QWEN35_MOE,
};

#[derive(Debug, Clone, Copy)]
pub struct Profile<'a> {
    pub packing: Qwen35Packing,
    pub ordinary_norm: QkNorm,
    pub qk_norm: QkNorm,
    pub epsilon: f32,
    /// Independently verified unscaled text, split-half rotary frequencies.
    pub rotary_inv_freq: &'a [f32],
    pub max_context: usize,
    pub output: OutputContract,
}
#[derive(Debug, Clone, Copy)]
pub enum OutputContract {
    Named { vocab: usize },
    TiedEmbedding { vocab: usize },
}
#[derive(Debug, Clone, Copy)]
pub enum LayerKind {
    Linear,
    FullAttention(Qwen35AttentionGeometry),
}
#[derive(Debug, Clone, Copy)]
pub enum Attention {
    Linear(Qwen35Roles),
    FullAttention(Qwen35AttentionRoles),
}
#[derive(Debug, Clone, Copy)]
pub struct LayerPlan {
    pub attention: Attention,
    pub input_norm: TensorInfo,
    pub post_attention_norm: TensorInfo,
    pub ffn: LayerTensors,
    pub routing: AdmittedMoeRouting,
    pub history_offset: usize,
    pub history_len: usize,
    pub recurrent_offset: usize,
    pub recurrent_len: usize,
    pub kv_offset: usize,
    pub kv_len: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanError {
    Profile,
    Architecture,
    MissingTensor,
    AmbiguousTensor,
    Shape,
    UnsupportedEncoding,
    FfnRowScaleUnsupported,
    MissingRouting,
    Roles,
    Overflow,
    Budget,
    Buffers,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Requirements {
    pub layers: usize,
    pub history: usize,
    pub recurrent: usize,
    pub kv: usize,
    pub scratch: usize,
    pub bytes: usize,
}
pub struct ModelPlan<'a> {
    layers: &'a [Option<LayerPlan>],
    final_norm: TensorInfo,
    profile: Profile<'a>,
    requirements: Requirements,
    output: Projection,
}
impl<'a> ModelPlan<'a> {
    pub fn output(&self) -> Projection {
        self.output
    }
    pub fn layers(&self) -> &[Option<LayerPlan>] {
        self.layers
    }
    pub fn final_norm(&self) -> TensorInfo {
        self.final_norm
    }
    pub fn profile(&self) -> Profile<'a> {
        self.profile
    }
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
}
fn resolve_output<H: Copy + Into<ModelHyperparams> + From<ModelHyperparams>>(
    index: &TensorIndex<H>,
    contract: OutputContract,
    hidden: usize,
) -> Result<Projection, PlanError> {
    let (names, vocab): (&[&[u8]], usize) = match contract {
        OutputContract::Named { vocab } => (&[b"output.weight", b"lm_head.weight"], vocab),
        OutputContract::TiedEmbedding { vocab } => {
            (&[b"token_embd.weight", b"model.embed_tokens.weight"], vocab)
        }
    };
    let mut found = None;
    for name in names {
        if let Some(info) = index.tensor_info(name) {
            if found.is_some_and(|old| old != info) {
                return Err(PlanError::AmbiguousTensor);
            }
            found = Some(info);
        }
    }
    let cached = match contract {
        OutputContract::Named { .. } => index.output_weight,
        OutputContract::TiedEmbedding { .. } => index.token_embd,
    };
    if let Some(info) = cached {
        if found.is_some_and(|old| old != info) {
            return Err(PlanError::AmbiguousTensor);
        }
        found = Some(info);
    }
    let weight = required(found)?;
    shape(weight, &[hidden, vocab])?;
    Ok(Projection {
        weight,
        row_scale: None,
    })
}
fn add(a: usize, b: usize) -> Result<usize, PlanError> {
    a.checked_add(b).ok_or(PlanError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, PlanError> {
    a.checked_mul(b).ok_or(PlanError::Overflow)
}
fn scalar(t: TensorInfo) -> Result<(), PlanError> {
    match t.encoding() {
        WeightEncoding::IeeeF32 | WeightEncoding::IeeeF16 | WeightEncoding::BrainF16 => Ok(()),
        WeightEncoding::Float8E4M3 => Err(PlanError::FfnRowScaleUnsupported),
        _ => Err(PlanError::UnsupportedEncoding),
    }
}
fn shape(t: TensorInfo, dims: &[usize]) -> Result<(), PlanError> {
    if t.n_dims as usize != dims.len()
        || dims
            .iter()
            .enumerate()
            .any(|(i, &d)| d == 0 || t.dims[i] != d as u64)
    {
        return Err(PlanError::Shape);
    }
    let mut count = 1;
    for &d in dims {
        count = mul(count, d)?;
    }
    scalar(t)?;
    let width = if t.encoding() == WeightEncoding::IeeeF32 {
        4
    } else {
        2
    };
    let bytes = u64::try_from(mul(count, width)?).map_err(|_| PlanError::Overflow)?;
    t.byte_offset
        .checked_add(bytes)
        .ok_or(PlanError::Overflow)?;
    Ok(())
}
fn required(t: Option<TensorInfo>) -> Result<TensorInfo, PlanError> {
    t.ok_or(PlanError::MissingTensor)
}
fn layer_tensor<H: Copy + Into<ModelHyperparams> + From<ModelHyperparams>>(
    index: &TensorIndex<H>,
    layer: u32,
    names: &[&[u8]],
) -> Result<TensorInfo, PlanError> {
    let mut result = None;
    let mut name = [0u8; 160];
    for suffix in names {
        for writer in [
            etalon_model::inference::write_hf_layer_tensor_name,
            etalon_model::inference::write_blk_tensor_name,
        ] {
            let n = writer(layer, suffix, &mut name);
            let hash = etalon_model::gguf::fnv1a64(&name[..n]);
            for (_, info) in index.entries.iter().filter(|(key, _)| *key == hash) {
                if result.is_some_and(|old| old != *info) {
                    return Err(PlanError::AmbiguousTensor);
                }
                result = Some(*info);
            }
            if let Some(info) = index.tensor_info(&name[..n]) {
                if result.is_some_and(|old| old != info) {
                    return Err(PlanError::AmbiguousTensor);
                }
                result = Some(info);
            }
        }
    }
    required(result)
}
fn final_norm<H: Copy + Into<ModelHyperparams> + From<ModelHyperparams>>(
    index: &TensorIndex<H>,
    hidden: usize,
) -> Result<TensorInfo, PlanError> {
    let mut found = None;
    for name in [
        b"model.norm.weight".as_slice(),
        b"output_norm.weight".as_slice(),
    ] {
        if let Some(t) = index.tensor_info(name) {
            if found.is_some_and(|old| old != t) {
                return Err(PlanError::AmbiguousTensor);
            }
            found = Some(t);
        }
    }
    let t = required(found)?;
    shape(t, &[hidden])?;
    Ok(t)
}
fn ffn(
    t: &LayerTensors,
    h: ModelHyperparams,
    index_routing: etalon_model::inference::moe_routing::MoeRoutingSettings,
) -> Result<(AdmittedMoeRouting, usize), PlanError> {
    // Only the existing separate-expert SwiGLU evaluator is admitted here.
    if t.ffn_gate.is_some()
        || t.ffn_up.is_some()
        || t.ffn_down.is_some()
        || t.moe_gate_up_exps.is_some()
    {
        return Err(PlanError::Shape);
    }
    let hidden = h.n_embd as usize;
    let gate = required(t.moe_gate_exps)?;
    let inter = usize::try_from(gate.dims[1]).map_err(|_| PlanError::Overflow)?;
    let experts = usize::try_from(gate.dims[2]).map_err(|_| PlanError::Overflow)?;
    if index_routing.expert_count.is_none() || index_routing.experts_per_token.is_none() {
        return Err(PlanError::MissingRouting);
    }
    let routing = index_routing.admit(experts).map_err(|_| PlanError::Shape)?;
    let shared = required(t.moe_shared_gate)?;
    let shared_inter = usize::try_from(shared.dims[1]).map_err(|_| PlanError::Overflow)?;
    if hidden > 4096 || inter > 10240 || shared_inter > 10240 {
        return Err(PlanError::Shape);
    }
    use crate::ordinary::moe_projection::{PlanError as E, ProjectionPlan};
    let admitted = ProjectionPlan::checked(hidden, t, routing).map_err(|e| match e {
        E::MissingTensor => PlanError::MissingTensor,
        E::Shape => PlanError::Shape,
        E::Overflow => PlanError::Overflow,
        E::UnsupportedEncoding => PlanError::UnsupportedEncoding,
        E::ExpertRowScaleUnsupported => PlanError::FfnRowScaleUnsupported,
    })?;
    let scratch = admitted.scratch_len();
    Ok((routing, scratch))
}
fn resolve_ffn<H: Copy + Into<ModelHyperparams> + From<ModelHyperparams>>(
    index: &TensorIndex<H>,
    layer: u32,
    t: &mut LayerTensors,
) -> Result<(), PlanError> {
    for (slot, names) in [
        (
            &mut t.moe_router,
            [
                b"ffn_gate_inp.weight".as_slice(),
                b"mlp.gate.weight".as_slice(),
            ],
        ),
        (
            &mut t.moe_gate_exps,
            [
                b"ffn_gate_exps.weight".as_slice(),
                b"mlp.experts.gate_proj.weight".as_slice(),
            ],
        ),
        (
            &mut t.moe_up_exps,
            [
                b"ffn_up_exps.weight".as_slice(),
                b"mlp.experts.up_proj.weight".as_slice(),
            ],
        ),
        (
            &mut t.moe_down_exps,
            [
                b"ffn_down_exps.weight".as_slice(),
                b"mlp.experts.down_proj.weight".as_slice(),
            ],
        ),
        (
            &mut t.moe_shared_gate,
            [
                b"ffn_gate_shexp.weight".as_slice(),
                b"mlp.shared_expert.gate_proj.weight".as_slice(),
            ],
        ),
        (
            &mut t.moe_shared_up,
            [
                b"ffn_up_shexp.weight".as_slice(),
                b"mlp.shared_expert.up_proj.weight".as_slice(),
            ],
        ),
        (
            &mut t.moe_shared_down,
            [
                b"ffn_down_shexp.weight".as_slice(),
                b"mlp.shared_expert.down_proj.weight".as_slice(),
            ],
        ),
        (
            &mut t.moe_shared_gate_input,
            [
                b"ffn_gate_inp_shexp.weight".as_slice(),
                b"mlp.shared_expert_gate.weight".as_slice(),
            ],
        ),
    ] {
        let info = layer_tensor(index, layer, &names)?;
        if slot.is_some_and(|old| old != info) {
            return Err(PlanError::AmbiguousTensor);
        }
        *slot = Some(info);
    }
    Ok(())
}
fn scan<H: Copy + Into<ModelHyperparams> + From<ModelHyperparams>>(
    index: &TensorIndex<H>,
    kinds: &[LayerKind],
    profile: Profile<'_>,
    mut out: Option<&mut [Option<LayerPlan>]>,
) -> Result<(Requirements, TensorInfo), PlanError> {
    let h: ModelHyperparams = index.hyperparams.into();
    if h.architecture != ARCH_QWEN35_MOE {
        return Err(PlanError::Architecture);
    }
    if h.n_embd == 0
        || kinds.len() != h.n_layer as usize
        || kinds.is_empty()
        || profile.max_context == 0
        || !profile.epsilon.is_finite()
        || profile.epsilon <= 0.0
        || !profile
            .rotary_inv_freq
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    {
        return Err(PlanError::Profile);
    }
    let hidden = h.n_embd as usize;
    let final_norm = final_norm(index, hidden)?;
    let mut r = Requirements {
        layers: kinds.len(),
        ..Default::default()
    };
    for (i, kind) in kinds.iter().enumerate() {
        let mut tensors = index.get_layer_tensors(i as u32);
        resolve_ffn(index, i as u32, &mut tensors)?;
        let input_norm = layer_tensor(
            index,
            i as u32,
            &[b"input_layernorm.weight", b"attn_norm.weight"],
        )?;
        let post_attention_norm = layer_tensor(
            index,
            i as u32,
            &[b"post_attention_layernorm.weight", b"ffn_norm.weight"],
        )?;
        shape(input_norm, &[hidden])?;
        shape(post_attention_norm, &[hidden])?;
        let (routing, ffn_scratch) = ffn(&tensors, h, index.moe_routing)?;
        let (attention, history_len, recurrent_len, kv_len, scratch) = match kind {
            LayerKind::Linear => {
                if tensors.attn_q.is_some() || tensors.attn_k.is_some() || tensors.attn_v.is_some()
                {
                    return Err(PlanError::Shape);
                }
                let roles = Qwen35Roles::resolve(index, i as u32, profile.packing)
                    .map_err(|_| PlanError::Roles)?;
                let w =
                    WorkspaceLayout::checked(roles.geometry).map_err(|_| PlanError::Overflow)?;
                (
                    Attention::Linear(roles),
                    w.history,
                    w.recurrent,
                    0,
                    w.floats,
                )
            }
            LayerKind::FullAttention(g) => {
                if tensors.ssm_conv1d.is_some()
                    || g.hidden != hidden
                    || profile.rotary_inv_freq.len() != g.rotary_dim / 2
                {
                    return Err(PlanError::Profile);
                }
                let roles = Qwen35AttentionRoles::resolve(index, i as u32, *g, profile.qk_norm)
                    .map_err(|_| PlanError::Roles)?;
                let kv = mul(mul(mul(profile.max_context, g.kv_heads)?, g.head_dim)?, 2)?;
                // Use the executor-owned scratch contract.
                let scratch =
                    super::full_attention::WorkspaceLayout::checked(*g, profile.max_context)
                        .map_err(|_| PlanError::Overflow)?
                        .floats;
                (Attention::FullAttention(roles), 0, 0, kv, scratch)
            }
        };
        let plan = LayerPlan {
            attention,
            input_norm,
            post_attention_norm,
            ffn: tensors,
            routing,
            history_offset: r.history,
            history_len,
            recurrent_offset: r.recurrent,
            recurrent_len,
            kv_offset: r.kv,
            kv_len,
        };
        r.history = add(r.history, history_len)?;
        r.recurrent = add(r.recurrent, recurrent_len)?;
        r.kv = add(r.kv, kv_len)?;
        r.scratch = r
            .scratch
            .max(add(scratch.max(ffn_scratch), mul(hidden, 4)?)?);
        if let Some(slots) = out.as_deref_mut() {
            slots[i] = Some(plan);
        }
    }
    let output = resolve_output(index, profile.output, hidden)?;
    r.scratch = add(r.scratch, output.rows())?;
    r.bytes = add(
        mul(r.layers, core::mem::size_of::<Option<LayerPlan>>())?,
        mul(add(add(r.history, r.recurrent)?, add(r.kv, r.scratch)?)?, 4)?,
    )?;
    Ok((r, final_norm))
}
pub fn measure<H: Copy + Into<ModelHyperparams> + From<ModelHyperparams>>(
    index: &TensorIndex<H>,
    kinds: &[LayerKind],
    profile: Profile<'_>,
    byte_budget: usize,
) -> Result<Requirements, PlanError> {
    let (r, _) = scan(index, kinds, profile, None)?;
    if r.bytes > byte_budget {
        return Err(PlanError::Budget);
    }
    Ok(r)
}
pub fn prepare<'a, H: Copy + Into<ModelHyperparams> + From<ModelHyperparams>>(
    index: &TensorIndex<H>,
    kinds: &[LayerKind],
    profile: Profile<'a>,
    byte_budget: usize,
    slots: &'a mut [Option<LayerPlan>],
) -> Result<ModelPlan<'a>, PlanError> {
    let requirements = measure(index, kinds, profile, byte_budget)?;
    if slots.len() != requirements.layers {
        return Err(PlanError::Buffers);
    }
    let (_, final_norm) = scan(index, kinds, profile, Some(slots))?;
    Ok(ModelPlan {
        layers: slots,
        final_norm,
        profile,
        requirements,
        output: resolve_output(
            index,
            profile.output,
            Into::<ModelHyperparams>::into(index.hyperparams).n_embd as usize,
        )?,
    })
}

#[cfg(test)]
#[path = "model_plan_tests.rs"]
mod tests;
