//! Caller-buffered routed and gated shared SwiGLU projection execution.
use etalon_inference_kernel::moe::{dispatch::silu, routing::route_topk};
use etalon_model::inference::{
    moe_routing::AdmittedMoeRouting, projection::Projection, LayerTensors, TensorInfo,
    WeightEncoding,
};

/// Implementations must honor row scales and perform no hot-path allocation.
pub trait ProjectionHost {
    type Error;
    fn admits(&self, projection: &Projection) -> bool;
    fn gemm(
        &self,
        projection: &Projection,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanError {
    MissingTensor,
    Shape,
    Overflow,
    UnsupportedEncoding,
    ExpertRowScaleUnsupported,
}
#[derive(Debug, PartialEq)]
pub enum Error<E> {
    Shape,
    Workspace,
    UnsupportedProjection,
    NonFinite,
    Routing,
    Provider(E),
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectionPlan {
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    routing: AdmittedMoeRouting,
    experts: [TensorInfo; 3],
    router: Projection,
    shared: [Projection; 4],
    scratch: usize,
}
fn add(a: usize, b: usize) -> Result<usize, PlanError> {
    a.checked_add(b).ok_or(PlanError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, PlanError> {
    a.checked_mul(b).ok_or(PlanError::Overflow)
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
    Ok(())
}
fn scalar_bytes(t: TensorInfo) -> Result<usize, PlanError> {
    match t.encoding() {
        WeightEncoding::IeeeF32 => Ok(4),
        WeightEncoding::IeeeF16 | WeightEncoding::BrainF16 => Ok(2),
        WeightEncoding::Float8E4M3 => Err(PlanError::ExpertRowScaleUnsupported),
        _ => Err(PlanError::UnsupportedEncoding),
    }
}
fn bounds(t: TensorInfo, width: usize) -> Result<(), PlanError> {
    let count = t.dims[..t.n_dims as usize]
        .iter()
        .try_fold(1usize, |n, &d| {
            mul(n, usize::try_from(d).map_err(|_| PlanError::Overflow)?)
        })?;
    t.byte_offset
        .checked_add(u64::try_from(mul(count, width)?).map_err(|_| PlanError::Overflow)?)
        .ok_or(PlanError::Overflow)?;
    Ok(())
}
fn dense(p: Projection, cols: usize, rows: usize) -> Result<(), PlanError> {
    shape(p.weight, &[cols, rows])?;
    match (p.weight.encoding(), p.row_scale) {
        (WeightEncoding::Float8E4M3, Some(s)) => {
            shape(s, &[rows])?;
            bounds(s, scalar_bytes(s)?)?;
            bounds(p.weight, 1)
        }
        (WeightEncoding::Float8E4M3, None) | (_, Some(_)) => Err(PlanError::UnsupportedEncoding),
        (_, None) => bounds(p.weight, scalar_bytes(p.weight)?),
    }
}
fn projection(weight: TensorInfo) -> Projection {
    Projection {
        weight,
        row_scale: None,
    }
}
impl ProjectionPlan {
    pub fn checked(
        hidden: usize,
        t: &LayerTensors,
        routing: AdmittedMoeRouting,
    ) -> Result<Self, PlanError> {
        let required = |t: Option<TensorInfo>| t.ok_or(PlanError::MissingTensor);
        if t.moe_gate_up_exps.is_some()
            || t.ffn_gate.is_some()
            || t.ffn_up.is_some()
            || t.ffn_down.is_some()
        {
            return Err(PlanError::Shape);
        }
        let mut shared_gate = required(t.moe_shared_gate_input)?;
        shape(shared_gate, &[hidden])?;
        shared_gate.n_dims = 2;
        shared_gate.dims[1] = 1;
        Self::checked_projections(
            hidden,
            [
                required(t.moe_gate_exps)?,
                required(t.moe_up_exps)?,
                required(t.moe_down_exps)?,
            ],
            projection(required(t.moe_router)?),
            [
                projection(required(t.moe_shared_gate)?),
                projection(required(t.moe_shared_up)?),
                projection(required(t.moe_shared_down)?),
                projection(shared_gate),
            ],
            routing,
        )
    }

    /// Dense FP8 roles carry their row scales explicitly; 3-D expert scales are unsupported.
    pub fn checked_projections(
        hidden: usize,
        experts: [TensorInfo; 3],
        router: Projection,
        shared: [Projection; 4],
        routing: AdmittedMoeRouting,
    ) -> Result<Self, PlanError> {
        let n = routing.expert_count();
        if n == 0
            || n > 512
            || routing.experts_per_token() == 0
            || routing.experts_per_token() > 16
            || routing.experts_per_token() > n
        {
            return Err(PlanError::Shape);
        }
        let intermediate = usize::try_from(experts[0].dims[1]).map_err(|_| PlanError::Overflow)?;
        let shared_intermediate =
            usize::try_from(shared[0].weight.dims[1]).map_err(|_| PlanError::Overflow)?;
        for (t, dims) in experts.iter().zip([
            [hidden, intermediate, n],
            [hidden, intermediate, n],
            [intermediate, hidden, n],
        ]) {
            shape(*t, &dims)?;
            bounds(*t, scalar_bytes(*t)?)?;
        }
        dense(router, hidden, n)?;
        for (p, (cols, rows)) in shared.iter().zip([
            (hidden, shared_intermediate),
            (hidden, shared_intermediate),
            (shared_intermediate, hidden),
            (hidden, 1),
        ]) {
            dense(*p, cols, rows)?;
        }
        let scratch = add(
            add(n, mul(intermediate.max(shared_intermediate), 2)?)?,
            add(mul(hidden, 2)?, 1)?,
        )?;
        mul(scratch, core::mem::size_of::<f32>())?;
        Ok(Self {
            hidden,
            intermediate,
            shared_intermediate,
            routing,
            experts,
            router,
            shared,
            scratch,
        })
    }
    pub const fn scratch_len(&self) -> usize {
        self.scratch
    }
    pub fn workspace_bytes(&self) -> usize {
        self.scratch * core::mem::size_of::<f32>()
    }
    fn expert(&self, role: usize, expert: usize) -> Result<Projection, PlanError> {
        let mut t = self.experts[role];
        let stride = mul(
            mul(t.dims[0] as usize, t.dims[1] as usize)?,
            scalar_bytes(t)?,
        )?;
        let offset = u64::try_from(mul(stride, expert)?).map_err(|_| PlanError::Overflow)?;
        t.byte_offset = t
            .byte_offset
            .checked_add(offset)
            .ok_or(PlanError::Overflow)?;
        t.n_dims = 2;
        t.dims[2] = 0;
        Ok(projection(t))
    }
}
fn finite<E>(v: &[f32]) -> Result<(), Error<E>> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}
fn gemm<H: ProjectionHost>(
    host: &H,
    p: &Projection,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), Error<H::Error>> {
    if !host.admits(p) {
        return Err(Error::UnsupportedProjection);
    }
    y.fill(f32::NAN);
    host.gemm(p, x, y).map_err(Error::Provider)?;
    finite(y)
}
pub fn execute<H: ProjectionHost>(
    host: &H,
    plan: &ProjectionPlan,
    input: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
) -> Result<(), Error<H::Error>> {
    let h = plan.hidden;
    if input.len() != h || out.len() != h {
        return Err(Error::Shape);
    }
    if scratch.len() < plan.scratch {
        return Err(Error::Workspace);
    }
    finite(input)?;
    let (logits, rest) = scratch.split_at_mut(plan.routing.expert_count());
    let width = plan.intermediate.max(plan.shared_intermediate);
    let (gate, rest) = rest.split_at_mut(width);
    let (up, rest) = rest.split_at_mut(width);
    let (expert_out, rest) = rest.split_at_mut(h);
    let (staged, rest) = rest.split_at_mut(h);
    gemm(host, &plan.router, input, logits)?;
    let mut indices = [0u16; 16];
    let mut weights = [0.0f32; 16];
    let selected = route_topk(
        logits,
        plan.routing.experts_per_token(),
        true,
        &mut indices,
        &mut weights,
    )
    .map_err(|_| Error::Routing)?;
    finite(&weights[..selected])?;
    staged.fill(0.0);
    for (&index, &weight) in indices[..selected].iter().zip(&weights[..selected]) {
        let mut roles = [plan.router; 3];
        for (role, p) in roles.iter_mut().enumerate() {
            *p = plan
                .expert(role, usize::from(index))
                .map_err(|_| Error::Shape)?;
        }
        let n = plan.intermediate;
        gemm(host, &roles[0], input, &mut gate[..n])?;
        gemm(host, &roles[1], input, &mut up[..n])?;
        for i in 0..n {
            gate[i] = silu(gate[i]) * up[i];
        }
        finite(&gate[..n])?;
        gemm(host, &roles[2], &gate[..n], expert_out)?;
        for i in 0..h {
            staged[i] += weight * expert_out[i];
        }
        finite(staged)?;
    }
    let n = plan.shared_intermediate;
    gemm(host, &plan.shared[0], input, &mut gate[..n])?;
    gemm(host, &plan.shared[1], input, &mut up[..n])?;
    for i in 0..n {
        gate[i] = silu(gate[i]) * up[i];
    }
    finite(&gate[..n])?;
    gemm(host, &plan.shared[2], &gate[..n], expert_out)?;
    let shared_gate = &mut rest[..1];
    gemm(host, &plan.shared[3], input, shared_gate)?;
    let factor = 1.0 / (1.0 + (-shared_gate[0]).exp());
    for i in 0..h {
        staged[i] += factor * expert_out[i];
    }
    finite(staged)?;
    out.copy_from_slice(staged);
    Ok(())
}

#[cfg(test)]
mod tests;
