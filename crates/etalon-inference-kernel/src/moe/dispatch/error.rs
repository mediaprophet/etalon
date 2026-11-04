use super::super::{nvfp4::Nvfp4Error, routing::MoeError};

/// Nonallocating dispatch errors; host compatibility wrappers may allocate text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeDispatchError {
    InputOrAccumulatorTooSmall,
    TooManyExperts,
    Routing(MoeError),
    RoutedExpert(Nvfp4Error),
    SharedExpert(Nvfp4Error),
}

impl core::fmt::Display for MoeDispatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InputOrAccumulatorTooSmall => {
                write!(f, "MoE dispatch: input or accumulator buffer too small")
            }
            Self::TooManyExperts => {
                write!(f, "MoE dispatch: num_experts exceeds MAX_ROUTED_EXPERTS")
            }
            Self::Routing(error) => write!(f, "route_topk failed: {}", error),
            Self::RoutedExpert(error) => write!(f, "evaluate_swiglu_expert failed: {}", error),
            Self::SharedExpert(error) => write!(f, "evaluate shared expert failed: {}", error),
        }
    }
}

impl core::error::Error for MoeDispatchError {}
