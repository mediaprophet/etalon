use crate::ContractError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorKind {
    Cpu,
    Gpu,
    Dsp,
    Npu,
    DeviceFixedFunction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorPreference {
    AnySupported,
    Prefer(ProcessorKind),
    Require(ProcessorKind),
}

impl ProcessorPreference {
    pub const fn permits(self, actual: ProcessorKind) -> bool {
        match self {
            Self::AnySupported | Self::Prefer(_) => true,
            Self::Require(required) => required as u8 == actual as u8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationId {
    id: u64,
    revision: u32,
}

impl OperationId {
    pub const fn new(id: u64, revision: u32) -> Result<Self, ContractError> {
        if id == 0 || revision == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self { id, revision })
    }

    pub const fn id(self) -> u64 {
        self.id
    }

    pub const fn revision(self) -> u32 {
        self.revision
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessingRequest {
    operation: OperationId,
    preference: ProcessorPreference,
    max_workspace_bytes: u64,
    latency_budget_ns: Option<u64>,
}

impl ProcessingRequest {
    pub const fn new(
        operation: OperationId,
        preference: ProcessorPreference,
        max_workspace_bytes: u64,
        latency_budget_ns: Option<u64>,
    ) -> Result<Self, ContractError> {
        if let Some(0) = latency_budget_ns {
            return Err(ContractError::InvalidTime);
        }
        Ok(Self {
            operation,
            preference,
            max_workspace_bytes,
            latency_budget_ns,
        })
    }

    pub const fn operation(self) -> OperationId {
        self.operation
    }

    pub const fn preference(self) -> ProcessorPreference {
        self.preference
    }

    pub const fn max_workspace_bytes(self) -> u64 {
        self.max_workspace_bytes
    }

    /// Optional maximum elapsed duration, not an absolute clock timestamp.
    pub const fn latency_budget_ns(self) -> Option<u64> {
        self.latency_budget_ns
    }

    pub const fn check_workspace(self, required_bytes: u64) -> Result<(), ContractError> {
        if required_bytes > self.max_workspace_bytes {
            Err(ContractError::OutOfBounds)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "processing_tests.rs"]
mod tests;
