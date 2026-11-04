use std::{env::VarError, ffi::OsString, sync::OnceLock};

#[derive(Clone, Copy)]
pub struct ExecutionPolicy {
    pub prefer_tensor_core_gemm: fn() -> bool,
    pub env_var: fn(&str) -> Result<String, VarError>,
    pub env_var_os: fn(&str) -> Option<OsString>,
}

static POLICY: OnceLock<ExecutionPolicy> = OnceLock::new();

pub fn install(policy: ExecutionPolicy) -> Result<(), ExecutionPolicy> {
    POLICY.set(policy)
}

pub fn prefer_tensor_core_gemm() -> bool {
    #[cfg(test)]
    if let Some(mode) = TEST_MODE.with(|mode| mode.get()) {
        return mode == InferenceMode::CudaTc;
    }
    POLICY
        .get()
        .is_none_or(|policy| (policy.prefer_tensor_core_gemm)())
}
pub fn env_var(key: &str) -> Result<String, VarError> {
    POLICY
        .get()
        .map_or_else(|| std::env::var(key), |policy| (policy.env_var)(key))
}
pub fn env_var_os(key: &str) -> Option<OsString> {
    POLICY
        .get()
        .map_or_else(|| std::env::var_os(key), |policy| (policy.env_var_os)(key))
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceMode {
    Portable,
    CudaTc,
}
#[cfg(test)]
thread_local! { static TEST_MODE: std::cell::Cell<Option<InferenceMode>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub fn set_inference_mode(mode: InferenceMode) {
    TEST_MODE.with(|slot| slot.set(Some(mode)));
}
#[cfg(test)]
pub fn active_inference_mode() -> InferenceMode {
    TEST_MODE.with(|slot| slot.get().unwrap_or(InferenceMode::Portable))
}
