//! Deterministic planning over caller-supplied backend declarations.
//! A plan is not device discovery, an allocation, a dispatch, or a guarantee
//! that a provider is live or can meet a latency budget.
use crate::processing::{OperationId, ProcessingRequest, ProcessorKind, ProcessorPreference};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendCandidate {
    pub provider_id: u64,
    pub processor: ProcessorKind,
    pub operation: OperationId,
    pub required_workspace_bytes: u64,
    /// An explicit caller-supplied estimate, not a measured guarantee.
    pub estimated_latency_ns: Option<u64>,
    pub available: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessingPlan {
    pub candidate_index: usize,
    pub provider_id: u64,
    pub processor: ProcessorKind,
    pub required_workspace_bytes: u64,
    pub estimated_latency_ns: Option<u64>,
}

/// Select the first eligible candidate, preferring the requested processor.
/// A bounded latency request rejects candidates without an estimate. The
/// runtime must independently revalidate availability and enforce budgets.
pub fn select_backend(request: ProcessingRequest, candidates: &[BackendCandidate]) -> Option<ProcessingPlan> {
    let mut fallback = None;
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.provider_id == 0 || !candidate.available
            || candidate.operation != request.operation()
            || !request.preference().permits(candidate.processor)
            || request.check_workspace(candidate.required_workspace_bytes).is_err()
        { continue; }
        if let Some(budget) = request.latency_budget_ns() {
            match candidate.estimated_latency_ns {
                Some(estimate) if estimate <= budget => {},
                _ => continue,
            }
        }
        let plan = ProcessingPlan { candidate_index: index, provider_id: candidate.provider_id,
            processor: candidate.processor, required_workspace_bytes: candidate.required_workspace_bytes,
            estimated_latency_ns: candidate.estimated_latency_ns };
        match request.preference() {
            ProcessorPreference::Prefer(preferred) if candidate.processor != preferred => {
                if fallback.is_none() { fallback = Some(plan); }
            }
            _ => return Some(plan),
        }
    }
    fallback
}

#[cfg(test)]
#[path = "processing_route_tests.rs"]
mod tests;
