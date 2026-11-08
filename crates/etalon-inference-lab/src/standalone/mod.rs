//! Native developer measurements; no application authority or promotion is implied.
mod artifacts;
mod executor;
mod model_run;
pub use artifacts::BoundedArtifacts;
pub use executor::{StandaloneCpuExecutor, StandaloneOptions};

pub fn dense_cpu_space() -> crate::config_space::ConfigurationSpace {
    crate::config_space::ConfigurationSpace::new("p64-dense-cpu-v1")
        .with("reversible_decode", crate::config_space::ParameterDef::Bool)
}

pub fn runtime_identity() -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    for source in [
        include_str!("../../../etalon-inference/src/cpu_model/native.rs"),
        include_str!("../../../etalon-inference/src/cpu_model/kernels.rs"),
        include_str!("../../../etalon-inference/src/cpu_model/sequence.rs"),
        include_str!("../../../etalon-inference-kernel/src/dense_cpu/forward.rs"),
        include_str!("../../../etalon-inference-kernel/src/dense_cpu/attention.rs"),
        include_str!("../../../etalon-inference-kernel/src/dense_cpu/math.rs"),
        include_str!("../../../etalon-inference-kernel/src/dense_cpu/plan.rs"),
        include_str!("executor.rs"),
        include_str!("model_run.rs"),
    ] {
        digest.update(source.as_bytes());
    }
    format!(
        "etalon:p64-dense:scalar-cpu:source-sha256:{:x}",
        digest.finalize()
    )
}
