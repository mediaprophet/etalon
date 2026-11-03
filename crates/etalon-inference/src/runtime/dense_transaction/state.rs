use etalon_inference_kernel::dense_cpu::{DenseStep, DenseWorkspace};

/// An unaccepted numerical proposal. Dropping it restores every borrowed buffer.
/// It cannot be cloned; commit and abort consume it, preventing duplicate resolution.
#[derive(Debug)]
pub struct PendingDense<'w, 'u> {
    workspace: DenseWorkspace<'w>,
    undo: &'u mut [f32],
    proposal: Option<DenseStep>,
    armed: bool,
}

impl<'w, 'u> PendingDense<'w, 'u> {
    pub(super) fn capture(workspace: DenseWorkspace<'w>, undo: &'u mut [f32]) -> Self {
        let pending = Self {
            workspace,
            undo,
            proposal: None,
            armed: true,
        };
        let mut offset = 0;
        for buffer in buffers(&pending.workspace) {
            pending.undo[offset..offset + buffer.len()].copy_from_slice(buffer);
            offset += buffer.len();
        }
        pending
    }

    pub(super) fn reborrow(&mut self) -> DenseWorkspace<'_> {
        let w = &mut self.workspace;
        DenseWorkspace {
            kv: w.kv,
            hidden: w.hidden,
            normed: w.normed,
            norm_weight: w.norm_weight,
            q: w.q,
            k: w.k,
            v: w.v,
            attention: w.attention,
            projection: w.projection,
            gate: w.gate,
            up: w.up,
            scores: w.scores,
            logits: w.logits,
        }
    }

    pub(super) fn set_proposal(&mut self, step: DenseStep) {
        self.proposal = Some(step);
    }

    /// Candidate output for policy inspection; this is not an acceptance receipt.
    pub fn proposal(&self) -> DenseStep {
        self.proposal
            .expect("only completed proposals are returned")
    }

    /// Read-only full buffers, including spare tails, in DenseWorkspace field order.
    /// The borrow prevents resolution while these observations are still in use.
    pub fn observations(&self) -> [&[f32]; 13] {
        buffers(&self.workspace)
    }

    /// Retain numerical state after the owning mode's acceptance decision.
    /// This method does not obtain or authenticate that decision itself.
    pub fn commit(mut self) -> DenseStep {
        let step = self.proposal();
        self.armed = false;
        step
    }

    /// Restore numerical state. Host/controller state remains the owner's obligation.
    pub fn abort(self) {
        drop(self);
    }
}

fn buffers<'a>(w: &'a DenseWorkspace<'_>) -> [&'a [f32]; 13] {
    [
        w.kv,
        w.hidden,
        w.normed,
        w.norm_weight,
        w.q,
        w.k,
        w.v,
        w.attention,
        w.projection,
        w.gate,
        w.up,
        w.scores,
        w.logits,
    ]
}

impl Drop for PendingDense<'_, '_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let w = &mut self.workspace;
        let mut offset = 0;
        for buffer in [
            &mut *w.kv,
            &mut *w.hidden,
            &mut *w.normed,
            &mut *w.norm_weight,
            &mut *w.q,
            &mut *w.k,
            &mut *w.v,
            &mut *w.attention,
            &mut *w.projection,
            &mut *w.gate,
            &mut *w.up,
            &mut *w.scores,
            &mut *w.logits,
        ] {
            buffer.copy_from_slice(&self.undo[offset..offset + buffer.len()]);
            offset += buffer.len();
        }
    }
}
