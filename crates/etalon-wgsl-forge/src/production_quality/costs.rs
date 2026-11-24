//! Labeled build/compile/upload/steady costs. Missing values are unsupported, never zero.

use crate::cache::TimingSource;
use crate::ir::ForgeError;

use super::schedule_gate::LabeledTiming;

pub const INTEGRATED_PHASE_COST_NOTE: &str =
    "integrated decode/upload/compaction/draw confirmation belongs to P22 after P18";

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostPresence {
    Measured(LabeledTiming),
    Unsupported { reason: String },
}

impl CostPresence {
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported {
            reason: reason.into(),
        }
    }

    pub fn median_ns(&self, phase: &str) -> Result<u64, ForgeError> {
        match self {
            Self::Measured(timing) => Ok(timing.median_ns),
            Self::Unsupported { reason } => Err(ForgeError::Validation(format!(
                "{phase} phase cost is unsupported, not zero ({reason})"
            ))),
        }
    }

    fn source(&self) -> Option<TimingSource> {
        match self {
            Self::Measured(timing) => Some(timing.source),
            Self::Unsupported { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PhaseCosts {
    pub build: CostPresence,
    pub compile: CostPresence,
    pub upload: CostPresence,
    pub steady: CostPresence,
}

impl PhaseCosts {
    pub fn total_ns(&self, n: u64) -> Result<u64, ForgeError> {
        let build = self.build.median_ns("build")?;
        let compile = self.compile.median_ns("compile")?;
        let upload = self.upload.median_ns("upload")?;
        let steady = self.steady.median_ns("steady")?;
        let fixed = build
            .checked_add(compile)
            .and_then(|value| value.checked_add(upload))
            .ok_or_else(|| ForgeError::Validation("phase cost overflow".to_string()))?;
        let variable = n
            .checked_mul(steady)
            .ok_or_else(|| ForgeError::Validation("phase cost overflow".to_string()))?;
        fixed
            .checked_add(variable)
            .ok_or_else(|| ForgeError::Validation("phase cost overflow".to_string()))
    }

    fn require_uniform_source(&self) -> Result<TimingSource, ForgeError> {
        let sources = [
            self.build.source(),
            self.compile.source(),
            self.upload.source(),
            self.steady.source(),
        ];
        if sources.iter().any(Option::is_none) {
            return Err(ForgeError::Validation(
                "phase cost is unsupported, not zero".to_string(),
            ));
        }
        let first = sources[0].expect("measured");
        if sources.iter().any(|source| *source != Some(first)) {
            return Err(ForgeError::Validation(
                "break-even accounting refuses mixed timing methods".to_string(),
            ));
        }
        Ok(first)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakEvenStatus {
    CandidateWins { from_n: u64, until_n: Option<u64> },
    BaselineRemains,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BreakEvenReport {
    pub status: BreakEvenStatus,
    pub method: Option<TimingSource>,
    pub note: String,
}

pub fn break_even(
    baseline: &PhaseCosts,
    candidate: &PhaseCosts,
) -> Result<BreakEvenReport, ForgeError> {
    let baseline_method = match baseline.require_uniform_source() {
        Ok(method) => method,
        Err(_) => {
            return Ok(BreakEvenReport {
                status: BreakEvenStatus::Inconclusive,
                method: None,
                note: "phase cost is unsupported, not zero".to_string(),
            });
        }
    };
    let candidate_method = match candidate.require_uniform_source() {
        Ok(method) => method,
        Err(_) => {
            return Ok(BreakEvenReport {
                status: BreakEvenStatus::Inconclusive,
                method: None,
                note: "phase cost is unsupported, not zero".to_string(),
            });
        }
    };
    if baseline_method != candidate_method {
        return Ok(BreakEvenReport {
            status: BreakEvenStatus::Inconclusive,
            method: None,
            note: "break-even accounting refuses mixed timing methods".to_string(),
        });
    }

    let b_fixed = add3(
        baseline.build.median_ns("build")?,
        baseline.compile.median_ns("compile")?,
        baseline.upload.median_ns("upload")?,
    )?;
    let c_fixed = add3(
        candidate.build.median_ns("build")?,
        candidate.compile.median_ns("compile")?,
        candidate.upload.median_ns("upload")?,
    )?;
    let b_steady = i128::from(baseline.steady.median_ns("steady")?);
    let c_steady = i128::from(candidate.steady.median_ns("steady")?);
    let ds = c_steady - b_steady;
    let df = i128::from(b_fixed) - i128::from(c_fixed);

    let status = if ds == 0 {
        if df >= 0 {
            BreakEvenStatus::CandidateWins {
                from_n: 0,
                until_n: None,
            }
        } else {
            BreakEvenStatus::BaselineRemains
        }
    } else if ds < 0 {
        if df >= 0 {
            BreakEvenStatus::CandidateWins {
                from_n: 0,
                until_n: None,
            }
        } else {
            let extra = (-df) as u128;
            let saving = (-ds) as u128;
            let from_n = extra.div_ceil(saving) as u64;
            BreakEvenStatus::CandidateWins {
                from_n,
                until_n: None,
            }
        }
    } else if df < 0 {
        BreakEvenStatus::BaselineRemains
    } else {
        let until_n = (df as u128 / ds as u128) as u64;
        BreakEvenStatus::CandidateWins {
            from_n: 0,
            until_n: Some(until_n),
        }
    };

    Ok(BreakEvenReport {
        status,
        method: Some(baseline_method),
        note: INTEGRATED_PHASE_COST_NOTE.to_string(),
    })
}

fn add3(a: u64, b: u64, c: u64) -> Result<u64, ForgeError> {
    a.checked_add(b)
        .and_then(|value| value.checked_add(c))
        .ok_or_else(|| ForgeError::Validation("phase cost overflow".to_string()))
}
