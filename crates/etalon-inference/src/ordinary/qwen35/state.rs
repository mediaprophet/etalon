//! Caller-owned model state; a failed partially executed token invalidates the sequence.
use super::model_plan::{ModelPlan, PlanError, Requirements};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceError {
    Invalidated,
    Busy,
    Position,
    ContextFull,
    NotRunning,
}
pub struct ModelState<'a> {
    plan: Option<&'a ModelPlan<'a>>,
    history: &'a mut [f32],
    recurrent: &'a mut [f32],
    kv: &'a mut [f32],
    scratch: &'a mut [f32],
    cursor: usize,
    max_context: usize,
    valid: bool,
    running: bool,
}
impl<'a> ModelState<'a> {
    pub fn bind(
        plan: &'a ModelPlan<'a>,
        history: &'a mut [f32],
        recurrent: &'a mut [f32],
        kv: &'a mut [f32],
        scratch: &'a mut [f32],
    ) -> Result<Self, PlanError> {
        let Requirements {
            history: nh,
            recurrent: nr,
            kv: nk,
            scratch: ns,
            ..
        } = plan.requirements();
        if history.len() != nh || recurrent.len() != nr || kv.len() != nk || scratch.len() != ns {
            return Err(PlanError::Buffers);
        }
        let mut state = Self {
            plan: Some(plan),
            history,
            recurrent,
            kv,
            scratch,
            cursor: 0,
            max_context: plan.profile().max_context,
            valid: false,
            running: false,
        };
        state.reset();
        Ok(state)
    }
    pub fn matches(&self, plan: &ModelPlan<'_>) -> bool {
        self.plan.is_some_and(|bound| core::ptr::eq(bound, plan))
    }
    pub fn cache(&self) -> (&[f32], &[f32], &[f32]) {
        (self.history, self.recurrent, self.kv)
    }
    pub fn position(&self) -> Result<usize, SequenceError> {
        if !self.valid {
            Err(SequenceError::Invalidated)
        } else {
            Ok(self.cursor)
        }
    }
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.recurrent.fill(0.0);
        self.kv.fill(0.0);
        self.scratch.fill(0.0);
        self.cursor = 0;
        self.valid = true;
        self.running = false;
    }
    /// Dropping without commit invalidates all partially advanced layer states, including unwind.
    pub(super) fn begin_token(
        &mut self,
        position: usize,
    ) -> Result<TokenStep<'_, 'a>, SequenceError> {
        if !self.valid {
            return Err(SequenceError::Invalidated);
        }
        if self.running {
            return Err(SequenceError::Busy);
        }
        if position != self.cursor {
            return Err(SequenceError::Position);
        }
        if position >= self.max_context {
            return Err(SequenceError::ContextFull);
        }
        self.running = true;
        Ok(TokenStep {
            state: self,
            committed: false,
        })
    }
}
pub(super) struct TokenStep<'s, 'a> {
    state: &'s mut ModelState<'a>,
    committed: bool,
}
impl TokenStep<'_, '_> {
    pub(super) fn buffers(&mut self) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        (
            self.state.history,
            self.state.recurrent,
            self.state.kv,
            self.state.scratch,
        )
    }
    /// Call only after every layer, final norm and output stage succeeds.
    pub(super) fn commit(mut self) {
        self.state.cursor += 1;
        self.state.running = false;
        self.committed = true;
    }
}
impl Drop for TokenStep<'_, '_> {
    fn drop(&mut self) {
        if !self.committed {
            self.state.valid = false;
            self.state.running = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state<'a>(
        a: &'a mut [f32],
        b: &'a mut [f32],
        c: &'a mut [f32],
        d: &'a mut [f32],
    ) -> ModelState<'a> {
        ModelState {
            plan: None,
            history: a,
            recurrent: b,
            kv: c,
            scratch: d,
            cursor: 0,
            max_context: 2,
            valid: true,
            running: false,
        }
    }
    #[test]
    fn partial_token_requires_complete_reset() {
        let (mut a, mut b, mut c, mut d) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
        let mut s = state(&mut a, &mut b, &mut c, &mut d);
        {
            let mut token = s.begin_token(0).unwrap();
            token.buffers().0[0] = 9.0;
        }
        assert_eq!(s.position(), Err(SequenceError::Invalidated));
        assert!(matches!(s.begin_token(0), Err(SequenceError::Invalidated)));
        s.reset();
        assert_eq!(s.history, &[0.0; 2]);
        s.begin_token(0).unwrap().commit();
        assert_eq!(s.position(), Ok(1));
        assert!(matches!(s.begin_token(0), Err(SequenceError::Position)));
        s.begin_token(1).unwrap().commit();
        assert!(matches!(s.begin_token(2), Err(SequenceError::ContextFull)));
    }
    #[test]
    fn unwind_invalidates_sequence() {
        let (mut a, mut b, mut c, mut d) = ([], [], [], []);
        let mut s = state(&mut a, &mut b, &mut c, &mut d);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _token = s.begin_token(0).unwrap();
            panic!("injected layer failure");
        }));
        assert_eq!(s.position(), Err(SequenceError::Invalidated));
    }
}
