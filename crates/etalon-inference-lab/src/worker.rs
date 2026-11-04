use crate::runner::Cancellation;
pub use etalon_diagnostics::process::WorkerLimits;
use std::process::Command;

pub fn run_isolated_worker(
    command: &mut Command,
    limits: &WorkerLimits,
    cancel: &dyn Cancellation,
) -> Result<crate::experiment::ExperimentResult, String> {
    let output =
        etalon_diagnostics::process::run_worker(command, limits, &|| cancel.cancelled(), &|| {
            Ok(())
        })?;
    serde_json::from_slice(&output.stdout).map_err(|e| format!("invalid worker result: {e}"))
}
