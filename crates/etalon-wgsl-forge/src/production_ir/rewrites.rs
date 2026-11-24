//! Guarded rewrites over existing Forge IR. FP legality is not real algebra.

use serde::{Deserialize, Serialize};

use crate::ir::{ForgeError, Op};

use super::contract::{
    ArithmeticModel, AssumptionKind, Discretization, MathOp, ProductionContract,
};

pub const MAX_REWRITE_REQUESTS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewriteKind {
    RealAlgebra,
    FloatingPoint,
    Memory,
    Geometry,
    Schedule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewriteRequest {
    /// Real: `a+b = b+a`. IEEE add is commutative; recorded as floating-point, not real algebra.
    CommuteAdd,
    /// Real: `(a+b)+c = a+(b+c)`. Not IEEE-legal without `fp_reassoc`.
    ReassociateAdd,
    /// Real: `x/x = 1` for `x ≠ 0`. Domain-invalid under IEEE without a real-algebra domain.
    CancelDiv,
    MulIdentity,
    /// `a*b+c → fma`. Changes rounding; needs `fma_contract`.
    FuseFma,
    /// Always rejected: CAS scatter stays occupancy-bounded.
    UnboundedCasSpin,
    /// Collapse filtered uncertain to a definite class. Needs `geometry_uncertain_policy`.
    CollapseGeometryUncertain,
    /// Always rejected: existing df64 is paired f32, not native WGSL f64.
    PromoteDf64ToNativeF64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AppliedRewrite {
    pub rule: RewriteRequest,
    pub kind: RewriteKind,
    pub assumptions: Vec<String>,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RejectedRewrite {
    pub rule: RewriteRequest,
    pub kind: RewriteKind,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct RewriteTrace {
    pub applied: Vec<AppliedRewrite>,
    pub rejected: Vec<RejectedRewrite>,
}

impl RewriteTrace {
    pub fn canonical_hash(&self) -> Result<String, ForgeError> {
        let bytes = serde_json::to_vec(self)?;
        Ok(super::delimited_blake3(
            super::contract::PRODUCTION_IR_SCHEMA_V1,
            &[bytes.as_slice()],
        ))
    }
}

pub fn apply_guarded(
    mut contract: ProductionContract,
    requests: &[RewriteRequest],
) -> Result<(ProductionContract, RewriteTrace), ForgeError> {
    if requests.len() > MAX_REWRITE_REQUESTS {
        return Err(ForgeError::Validation(format!(
            "rewrite list {} exceeds cold-path bound {MAX_REWRITE_REQUESTS}",
            requests.len()
        )));
    }
    contract.validate()?;
    let mut trace = RewriteTrace::default();
    for request in requests {
        match decide(&contract, *request) {
            Ok(applied) => {
                apply_ir(&mut contract, *request);
                trace.applied.push(applied);
            }
            Err(rejected) => trace.rejected.push(rejected),
        }
    }
    contract.validate()?;
    Ok((contract, trace))
}

fn decide(
    contract: &ProductionContract,
    request: RewriteRequest,
) -> Result<AppliedRewrite, RejectedRewrite> {
    match request {
        RewriteRequest::CommuteAdd => {
            if !has_add(&contract.kernel.ops) {
                return Err(reject(
                    request,
                    RewriteKind::FloatingPoint,
                    "no Add in existing IR",
                ));
            }
            match contract.arithmetic {
                ArithmeticModel::IntegerI32 | ArithmeticModel::IntegerU32 => Ok(applied(
                    request,
                    RewriteKind::RealAlgebra,
                    &[],
                    "integer add is commutative",
                )),
                ArithmeticModel::IeeeF32 | ArithmeticModel::Df64PairedF32 => Ok(applied(
                    request,
                    RewriteKind::FloatingPoint,
                    &["ieee_f32"],
                    "IEEE add is commutative; this is not a real-algebra rewrite",
                )),
                ArithmeticModel::FilteredPredicate => Err(reject(
                    request,
                    RewriteKind::FloatingPoint,
                    "filtered predicates do not commute as arithmetic add",
                )),
            }
        }
        RewriteRequest::ReassociateAdd => {
            if matches!(
                contract.arithmetic,
                ArithmeticModel::IntegerI32 | ArithmeticModel::IntegerU32
            ) {
                return Ok(applied(
                    request,
                    RewriteKind::RealAlgebra,
                    &[],
                    "integer add is associative",
                ));
            }
            if contract.has_assumption("fp_reassoc") {
                return Ok(applied(
                    request,
                    RewriteKind::FloatingPoint,
                    &["fp_reassoc"],
                    "explicit fp_reassoc assumption grants reassociation; results are not IEEE-exact",
                ));
            }
            Err(reject(
                request,
                RewriteKind::FloatingPoint,
                "floating-point reassociation is not real-algebra legal under IEEE f32",
            ))
        }
        RewriteRequest::CancelDiv => {
            if contract.has_assumption("domain_nonzero_finite")
                && contract
                    .assumptions
                    .iter()
                    .any(|item| item.kind == AssumptionKind::RealAlgebra)
            {
                return Ok(applied(
                    request,
                    RewriteKind::RealAlgebra,
                    &["domain_nonzero_finite"],
                    "real-algebra cancellation on a declared nonzero finite domain",
                ));
            }
            Err(reject(
                request,
                RewriteKind::FloatingPoint,
                "x/x → 1 is domain-invalid under IEEE (0, inf, NaN); not a legal FP rewrite",
            ))
        }
        RewriteRequest::MulIdentity => {
            if matches!(
                contract.arithmetic,
                ArithmeticModel::FilteredPredicate
            ) {
                return Err(reject(
                    request,
                    RewriteKind::FloatingPoint,
                    "filtered predicates have no mul identity rewrite",
                ));
            }
            Ok(applied(
                request,
                match contract.arithmetic {
                    ArithmeticModel::IntegerI32 | ArithmeticModel::IntegerU32 => {
                        RewriteKind::RealAlgebra
                    }
                    _ => RewriteKind::FloatingPoint,
                },
                &[],
                "mul-by-one is legal on this arithmetic model",
            ))
        }
        RewriteRequest::FuseFma => {
            if !contract.has_assumption("fma_contract") {
                return Err(reject(
                    request,
                    RewriteKind::FloatingPoint,
                    "fusing mul+add into Fma changes rounding without fma_contract",
                ));
            }
            if !has_mul_add_pair(&contract.kernel.ops) {
                return Err(reject(
                    request,
                    RewriteKind::FloatingPoint,
                    "no mul-add pair in existing IR",
                ));
            }
            Ok(applied(
                request,
                RewriteKind::FloatingPoint,
                &["fma_contract"],
                "fused existing Mul+Add into Op::Fma under fma_contract",
            ))
        }
        RewriteRequest::UnboundedCasSpin => Err(reject(
            request,
            RewriteKind::Memory,
            "CAS scatter remains occupancy-bounded; unbounded retry is not a legal production rewrite",
        )),
        RewriteRequest::CollapseGeometryUncertain => {
            if !matches!(contract.math, MathOp::Geometry { .. })
                || contract.discretization != Discretization::FilteredOrient3d
            {
                return Err(reject(
                    request,
                    RewriteKind::Geometry,
                    "geometry uncertain collapse applies only to filtered orient3d selection",
                ));
            }
            if !contract.has_assumption("geometry_uncertain_policy") {
                return Err(reject(
                    request,
                    RewriteKind::Geometry,
                    "filtered uncertain must be retained without geometry_uncertain_policy",
                ));
            }
            Ok(applied(
                request,
                RewriteKind::Geometry,
                &["geometry_uncertain_policy"],
                "caller policy recorded; selection codes are unchanged",
            ))
        }
        RewriteRequest::PromoteDf64ToNativeF64 => Err(reject(
            request,
            RewriteKind::FloatingPoint,
            "df64 is paired f32 double-single, not native WGSL f64",
        )),
    }
}

fn apply_ir(contract: &mut ProductionContract, request: RewriteRequest) {
    match request {
        RewriteRequest::CommuteAdd => commute_add(&mut contract.kernel.ops),
        RewriteRequest::FuseFma => {
            let _ = fuse_mul_add(&mut contract.kernel.ops);
        }
        _ => {}
    }
}

fn applied(
    rule: RewriteRequest,
    kind: RewriteKind,
    assumptions: &[&str],
    note: &str,
) -> AppliedRewrite {
    AppliedRewrite {
        rule,
        kind,
        assumptions: assumptions.iter().map(|id| (*id).to_string()).collect(),
        note: note.to_string(),
    }
}

fn reject(rule: RewriteRequest, kind: RewriteKind, reason: &str) -> RejectedRewrite {
    RejectedRewrite {
        rule,
        kind,
        reason: reason.to_string(),
    }
}

fn has_add(ops: &[Op]) -> bool {
    ops.iter().any(|op| match op {
        Op::Add { .. } => true,
        Op::Loop { body, .. } => has_add(body),
        _ => false,
    })
}

fn has_mul_add_pair(ops: &[Op]) -> bool {
    if sequential_mul_add(ops) {
        return true;
    }
    ops.iter().any(|op| match op {
        Op::Loop { body, .. } => has_mul_add_pair(body),
        _ => false,
    })
}

fn sequential_mul_add(ops: &[Op]) -> bool {
    ops.windows(2).any(|pair| {
        matches!(
            (&pair[0], &pair[1]),
            (
                Op::Mul { destination, .. },
                Op::Add { left, right, .. },
            ) if left == destination || right == destination
        )
    })
}

fn commute_add(ops: &mut [Op]) {
    for op in ops {
        match op {
            Op::Add { left, right, .. } => std::mem::swap(left, right),
            Op::Loop { body, .. } => commute_add(body),
            _ => {}
        }
    }
}

fn fuse_mul_add(ops: &mut Vec<Op>) -> usize {
    let mut count = 0;
    for op in ops.iter_mut() {
        if let Op::Loop { body, .. } = op {
            count += fuse_mul_add(body);
        }
    }
    let mut i = 0;
    while i + 1 < ops.len() {
        let fused = match (&ops[i], &ops[i + 1]) {
            (
                Op::Mul {
                    left,
                    right,
                    destination,
                },
                Op::Add {
                    left: add_left,
                    right: add_right,
                    destination: dest,
                },
            ) if add_left == destination || add_right == destination => {
                let c = if add_left == destination {
                    add_right.clone()
                } else {
                    add_left.clone()
                };
                Some(Op::Fma {
                    a: left.clone(),
                    b: right.clone(),
                    c,
                    destination: dest.clone(),
                })
            }
            _ => None,
        };
        if let Some(fma) = fused {
            ops.remove(i + 1);
            ops[i] = fma;
            count += 1;
        } else {
            i += 1;
        }
    }
    count
}
