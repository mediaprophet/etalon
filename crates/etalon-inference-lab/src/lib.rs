#![forbid(unsafe_code)]
//! Cold experiment orchestration. Execution and activation authority belong to the host.
pub mod ablate;
pub mod auto_improve;
pub mod benchmark;
pub mod campaign;
pub mod config_space;
pub mod device_roof;
pub mod evidence;
pub mod experiment;
pub mod experiment_log;
pub mod hypothesis;
pub mod micro;
pub mod pareto;
pub mod qualification;
pub mod runner;
pub mod search;
#[cfg(all(feature = "standalone-cpu", not(target_arch = "wasm32")))]
pub mod standalone;
pub mod timeline;
#[cfg(not(target_family = "wasm"))]
pub mod worker;

#[cfg(feature = "operator-lab")]
pub mod operator_compare;
#[cfg(feature = "operator-lab")]
pub mod operator_metrics;
