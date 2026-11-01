use crate::GpuError;
use core::task::{Context, Waker};
use std::sync::{Mutex, MutexGuard};

pub(super) struct MapCompletion {
    state: Mutex<State>,
}
struct State {
    generation: u64,
    result: Option<bool>,
    error: Option<GpuError>,
    retired: bool,
    waker: Option<Waker>,
}
impl MapCompletion {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                generation: 0,
                result: None,
                error: None,
                retired: false,
                waker: None,
            }),
        }
    }
    pub fn begin(&self) -> Result<u64, GpuError> {
        let mut state = self.lock();
        if state.retired {
            return Err(GpuError::ProviderPoisoned);
        }
        let Some(generation) = state.generation.checked_add(1) else {
            state.retired = true;
            return Err(GpuError::ProviderPoisoned);
        };
        state.generation = generation;
        state.result = None;
        state.error = None;
        state.waker = None;
        Ok(generation)
    }
    pub fn complete(&self, generation: u64, mapped: bool) {
        let waker = {
            let mut state = self.lock();
            if state.retired || state.generation != generation || state.result.is_some() {
                return;
            }
            state.result = Some(mapped);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
    pub fn fail(&self, generation: u64, error: GpuError) {
        let waker = {
            let mut state = self.lock();
            if state.retired || state.generation != generation || state.error.is_some() {
                return;
            }
            state.error = Some(error);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
    #[cfg(any(not(target_arch = "wasm32"), test))]
    pub fn result(&self, generation: u64) -> Option<bool> {
        let state = self.lock();
        if !state.retired && state.generation == generation {
            state.result
        } else {
            None
        }
    }
    pub fn status(&self, generation: u64, cx: &mut Context<'_>) -> Option<Result<(), GpuError>> {
        let mut state = self.lock();
        if state.retired || state.generation != generation {
            return Some(Err(GpuError::ProviderPoisoned));
        }
        state.waker = Some(cx.waker().clone());
        state.error.map(Err).or_else(|| {
            state
                .result
                .map(|ok| if ok { Ok(()) } else { Err(GpuError::MapFailed) })
        })
    }
    pub fn invalidate(&self, generation: u64) {
        let mut state = self.lock();
        if state.generation != generation {
            return;
        }
        state.result = None;
        state.error = None;
        state.waker = None;
        if let Some(next) = state.generation.checked_add(1) {
            state.generation = next;
        } else {
            state.retired = true;
        }
    }
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
