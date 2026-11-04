//! Mixed Qwen3.5 token execution over a sealed cold plan and caller-owned state.
use super::{
    finite, full_attention,
    model_plan::{Attention, ModelPlan},
    ordinary_rms_norm,
    state::{ModelState, SequenceError},
    workspace::take,
    Error, Host,
};
use crate::ordinary::moe_projection::{ProjectionHost, ProjectionPlan};
use etalon_model::inference::{qwen_attention::QkNorm, qwen_roles::Projection, TensorInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Attention,
    Ffn,
    FinalNorm,
    Logits,
}
/// Tentative developer instrumentation; production adapters use NoopObserver.
pub trait Observer {
    fn observe(&mut self, layer: usize, stage: Stage, values: &[f32]);
}
pub struct NoopObserver;
impl Observer for NoopObserver {
    fn observe(&mut self, _: usize, _: Stage, _: &[f32]) {}
}
#[derive(Debug)]
pub enum StepError {
    PlanMismatch,
    Shape,
    Sequence(SequenceError),
    Execution(Error),
    Ffn,
}
impl From<Error> for StepError {
    fn from(e: Error) -> Self {
        Self::Execution(e)
    }
}
struct FfnHost<'a, H>(&'a H);
impl<H: Host> ProjectionHost for FfnHost<'_, H> {
    type Error = Error;
    fn admits(&self, p: &Projection) -> bool {
        self.0.admits(p)
    }
    fn gemm(&self, p: &Projection, input: &[f32], out: &mut [f32]) -> Result<(), Error> {
        self.0.gemm(p, input, out)
    }
}
fn norm(
    host: &impl Host,
    info: TensorInfo,
    input: &[f32],
    raw: bool,
    epsilon: f32,
    weights: &mut [f32],
    out: &mut [f32],
) -> Result<(), Error> {
    weights.fill(f32::NAN);
    host.read(&info, weights)?;
    finite(weights)?;
    ordinary_rms_norm(input, weights, raw, epsilon, out)
}

/// Publishes hidden/logits only after every stage succeeds. Partial token state is invalidated
/// on error or unwind and must be reset before reuse. Source custody belongs to the caller.
#[allow(clippy::too_many_arguments)]
pub fn step(
    host: &impl Host,
    plan: &ModelPlan<'_>,
    state: &mut ModelState<'_>,
    position: usize,
    hidden: &mut [f32],
    logits: &mut [f32],
    observer: &mut impl Observer,
) -> Result<(), StepError> {
    if !state.matches(plan) {
        return Err(StepError::PlanMismatch);
    }
    let projection = plan.output();
    let width = projection.cols();
    if hidden.len() != width || logits.len() != projection.rows() {
        return Err(StepError::Shape);
    }
    finite(hidden)?;
    if !host.admits(&projection) {
        return Err(Error::UnsupportedProjection.into());
    }
    let mut token = state.begin_token(position).map_err(StepError::Sequence)?;
    let (history, recurrent, kv, scratch) = token.buffers();
    let mut rest = scratch;
    let candidate = take(&mut rest, width);
    let normalized = take(&mut rest, width);
    let weights = take(&mut rest, width);
    let ffn_out = take(&mut rest, width);
    let staged_logits = take(&mut rest, projection.rows());
    let workspace = rest;
    candidate.copy_from_slice(hidden);
    let profile = plan.profile();
    let raw = profile.ordinary_norm == QkNorm::RawZeroCentered;
    for (layer, slot) in plan.layers().iter().enumerate() {
        let layer_plan = slot.as_ref().ok_or(StepError::Shape)?;
        norm(
            host,
            layer_plan.input_norm,
            candidate,
            raw,
            profile.epsilon,
            weights,
            normalized,
        )?;
        match &layer_plan.attention {
            Attention::Linear(roles) => super::execute_normalized(
                host,
                roles,
                normalized,
                candidate,
                &mut history
                    [layer_plan.history_offset..layer_plan.history_offset + layer_plan.history_len],
                &mut recurrent[layer_plan.recurrent_offset
                    ..layer_plan.recurrent_offset + layer_plan.recurrent_len],
                profile.epsilon,
                workspace,
            )?,
            Attention::FullAttention(roles) => {
                let cache = &mut kv[layer_plan.kv_offset..layer_plan.kv_offset + layer_plan.kv_len];
                let (keys, values) = cache.split_at_mut(cache.len() / 2);
                let mut cursor = position;
                full_attention::execute_normalized(
                    host,
                    roles,
                    normalized,
                    candidate,
                    keys,
                    values,
                    &mut cursor,
                    full_attention::TextRope {
                        position,
                        inv_freq: profile.rotary_inv_freq,
                    },
                    profile.epsilon,
                    workspace,
                )?;
            }
        }
        observer.observe(layer, Stage::Attention, candidate);
        norm(
            host,
            layer_plan.post_attention_norm,
            candidate,
            raw,
            profile.epsilon,
            weights,
            normalized,
        )?;
        let ffn_plan = ProjectionPlan::checked(width, &layer_plan.ffn, layer_plan.routing)
            .map_err(|_| StepError::Ffn)?;
        ffn_out.fill(f32::NAN);
        crate::ordinary::moe_projection::execute(
            &FfnHost(host),
            &ffn_plan,
            normalized,
            ffn_out,
            workspace,
        )
        .map_err(|_| StepError::Ffn)?;
        finite(ffn_out)?;
        for (x, y) in candidate.iter_mut().zip(ffn_out.iter()) {
            *x += *y;
        }
        finite(candidate)?;
        observer.observe(layer, Stage::Ffn, candidate);
    }
    norm(
        host,
        plan.final_norm(),
        candidate,
        raw,
        profile.epsilon,
        weights,
        normalized,
    )?;
    observer.observe(usize::MAX, Stage::FinalNorm, normalized);
    staged_logits.fill(f32::NAN);
    host.gemm(&projection, normalized, staged_logits)?;
    finite(staged_logits)?;
    observer.observe(usize::MAX, Stage::Logits, staged_logits);
    hidden.copy_from_slice(normalized);
    logits.copy_from_slice(staged_logits);
    token.commit();
    Ok(())
}
