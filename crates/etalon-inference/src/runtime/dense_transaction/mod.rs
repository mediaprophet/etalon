//! One tentative dense CPU step with caller-owned, whole-workspace undo.
//!
//! This is numerical state control, not permission to accept a token. Normal
//! applications must obtain the real host/Sentinel decision before `commit`.
//! Developer tools may deliberately use it without a controller. Callbacks must
//! be stateless/read-only model access: their private mutations are not restored.
//! Host position, generation, grammar, sampler and publication state are outside
//! this primitive and must participate in the host's acceptance transaction.

mod state;
pub use state::PendingDense;

use etalon_inference_kernel::dense_cpu::{
    self as dense, DenseBuffer, DenseError, DensePlan, DenseRequirements, DenseWorkspace,
    DenseWorkspaceError, TensorKernels,
};

#[derive(Debug, PartialEq, Eq)]
pub enum TransactionError<E> {
    ArithmeticOverflow,
    Budget { required: usize, available: usize },
    UndoCapacity { required: usize, available: usize },
    Execution(DenseError<E>),
}

fn lengths(w: &DenseWorkspace<'_>) -> [usize; 13] {
    [
        w.kv.len(),
        w.hidden.len(),
        w.normed.len(),
        w.norm_weight.len(),
        w.q.len(),
        w.k.len(),
        w.v.len(),
        w.attention.len(),
        w.projection.len(),
        w.gate.len(),
        w.up.len(),
        w.scores.len(),
        w.logits.len(),
    ]
}

fn requirements(r: DenseRequirements) -> [(DenseBuffer, usize); 13] {
    [
        (DenseBuffer::Kv, r.kv),
        (DenseBuffer::Hidden, r.hidden),
        (DenseBuffer::Normed, r.normed),
        (DenseBuffer::NormWeight, r.norm_weight),
        (DenseBuffer::Q, r.q),
        (DenseBuffer::K, r.k),
        (DenseBuffer::V, r.v),
        (DenseBuffer::Attention, r.attention),
        (DenseBuffer::Projection, r.projection),
        (DenseBuffer::Gate, r.gate),
        (DenseBuffer::Up, r.up),
        (DenseBuffer::Scores, r.scores),
        (DenseBuffer::Logits, r.logits),
    ]
}

fn checked_sizes<E>(
    lengths: &[usize; 13],
    undo_len: usize,
    budget: usize,
) -> Result<(usize, usize), TransactionError<E>> {
    let elements = lengths
        .iter()
        .try_fold(0usize, |n, &v| n.checked_add(v))
        .ok_or(TransactionError::ArithmeticOverflow)?;
    let workspace_bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(TransactionError::ArithmeticOverflow)?;
    let required = elements
        .checked_add(undo_len)
        .and_then(|n| n.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(TransactionError::ArithmeticOverflow)?;
    if required > budget {
        return Err(TransactionError::Budget {
            required,
            available: budget,
        });
    }
    if undo_len < elements {
        return Err(TransactionError::UndoCapacity {
            required: elements,
            available: undo_len,
        });
    }
    Ok((elements, workspace_bytes))
}

/// Execute a proposal while retaining exclusive state until commit or abort.
///
/// Charges the complete lengths of all thirteen workspace buffers and the full
/// supplied undo slice. Only the used snapshot prefix is written; its spare tail
/// is preserved. Failure and unwind restore numerical state; undo retains the
/// captured values for its owner to manage. Successful execution allocates nothing
/// provided the model callbacks also allocate nothing.
pub fn propose_dense<'w, 'u, H: Copy, K: TensorKernels<H>>(
    plan: &DensePlan<'_, H>,
    kernels: &mut K,
    workspace: DenseWorkspace<'w>,
    undo: &'u mut [f32],
    token: u32,
    position: usize,
    total_byte_budget: usize,
) -> Result<PendingDense<'w, 'u>, TransactionError<K::Error>> {
    if position >= plan.config().max_context {
        return Err(TransactionError::Execution(DenseError::ContextExceeded {
            position,
            max_context: plan.config().max_context,
        }));
    }
    let actual = lengths(&workspace);
    for (&available, (buffer, required)) in actual.iter().zip(requirements(plan.requirements())) {
        if available < required {
            return Err(TransactionError::Execution(DenseError::Workspace(
                DenseWorkspaceError::Capacity {
                    buffer,
                    required,
                    available,
                },
            )));
        }
    }
    let (elements, workspace_bytes) = checked_sizes(&actual, undo.len(), total_byte_budget)?;
    let mut pending = PendingDense::capture(workspace, &mut undo[..elements]);
    let step = dense::run_token(
        plan,
        kernels,
        pending.reborrow(),
        token,
        position,
        workspace_bytes,
    )
    .map_err(TransactionError::Execution)?;
    pending.set_proposal(step);
    Ok(pending)
}

#[cfg(test)]
mod tests;
