//! Precision, layout, occupancy, subgroup, and shared-memory admission.

use crate::ir::{ForgeError, IntrinsicClass, KernelSpec};
use crate::production_ir::{ArithmeticModel, MathOp, ProductionContract};
use crate::schedule::{AdapterConstraints, Schedule};

use super::PRODUCTION_LOWERING_SCHEMA_VERSION;

pub const PORTABLE_MAX_SHARED_BYTES: u32 = 16_384;
pub const GEOMETRY_PACKED_QUERY_STRIDE: u32 = 15;
pub const CLUSTER_NODE_STRIDE_SCALARS: u32 = 12;
pub const DF64_PAIR_STRIDE: u32 = 2;
pub const MAX_FALLBACKS: usize = 4;
pub const OCCUPANCY_NOTE: &str =
    "occupancy is an admission/report, not a promise of measured GPU timing";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestedPrecision {
    IeeeF32,
    FilteredPredicate,
    Df64PairedF32,
    NativeWgslF64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutKind {
    ScalarF32,
    IntegerI32,
    IntegerU32,
    Df64PairedF32,
    GeometryPacked15,
    ClusterNode12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DeclaredOccupancy {
    pub waves_per_cu: u32,
    pub shared_bytes: Option<u32>,
    pub subgroup_size: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccupancyAssumption {
    None,
    Declared(DeclaredOccupancy),
    Undeclared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackKind {
    Precision,
    Layout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct FallbackRecord {
    pub kind: FallbackKind,
    pub requested: &'static str,
    pub admitted: &'static str,
    pub note: &'static str,
}

impl Default for FallbackRecord {
    fn default() -> Self {
        Self {
            kind: FallbackKind::Precision,
            requested: "",
            admitted: "",
            note: "",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmissionRequest<'a> {
    pub contract: &'a ProductionContract,
    pub constraints: &'a AdapterConstraints,
    pub precision: RequestedPrecision,
    pub layout: LayoutKind,
    pub occupancy: OccupancyAssumption,
    pub subgroup_required: bool,
    pub assumed_subgroup_size: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionAdmission {
    Admitted { model: ArithmeticModel },
    Fallback { model: ArithmeticModel },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct OccupancyReport {
    pub assumed: bool,
    pub workgroup_threads: u32,
    pub shared_bytes: u32,
    pub estimated_waves_in_workgroup: u32,
    #[serde(skip)]
    pub note: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SubgroupAdmission {
    pub required: bool,
    pub adapter_supports: bool,
    pub size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SharedMemoryAdmission {
    pub declared_bytes: u32,
    pub limit_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AdmissionReport {
    pub schema_version: u32,
    pub accepted: bool,
    pub precision: PrecisionAdmission,
    pub layout: LayoutKind,
    pub occupancy: OccupancyReport,
    pub subgroup: SubgroupAdmission,
    pub shared_memory: SharedMemoryAdmission,
    pub fallback_count: usize,
}

pub fn precision_from_contract(contract: &ProductionContract) -> RequestedPrecision {
    match contract.arithmetic {
        ArithmeticModel::IeeeF32 => RequestedPrecision::IeeeF32,
        ArithmeticModel::FilteredPredicate => RequestedPrecision::FilteredPredicate,
        ArithmeticModel::Df64PairedF32 => RequestedPrecision::Df64PairedF32,
        ArithmeticModel::IntegerI32 | ArithmeticModel::IntegerU32 => RequestedPrecision::IeeeF32,
    }
}

pub fn layout_from_contract(contract: &ProductionContract) -> LayoutKind {
    match contract.arithmetic {
        ArithmeticModel::Df64PairedF32 => LayoutKind::Df64PairedF32,
        ArithmeticModel::IntegerI32 => LayoutKind::IntegerI32,
        ArithmeticModel::IntegerU32 => LayoutKind::IntegerU32,
        ArithmeticModel::IeeeF32 | ArithmeticModel::FilteredPredicate => match contract.math {
            MathOp::Geometry { .. } => LayoutKind::GeometryPacked15,
            _ => LayoutKind::ScalarF32,
        },
    }
}

pub fn admit(
    request: &AdmissionRequest<'_>,
    source: Option<&str>,
    fallbacks_out: &mut [FallbackRecord],
) -> Result<AdmissionReport, ForgeError> {
    request.contract.validate_with(request.constraints)?;
    if matches!(request.occupancy, OccupancyAssumption::Undeclared) {
        return Err(ForgeError::Validation(
            "undeclared occupancy assumption is rejected; occupancy is an admission/report, not measured GPU timing".to_string(),
        ));
    }

    let mut fallback_count = 0usize;
    let precision = admit_precision(request, source, fallbacks_out, &mut fallback_count)?;
    let layout = admit_layout(request, fallbacks_out, &mut fallback_count)?;
    let shared = admit_shared_memory(request)?;
    let occupancy = admit_occupancy(request, shared.declared_bytes)?;
    let subgroup = admit_subgroup(request)?;

    Ok(AdmissionReport {
        schema_version: PRODUCTION_LOWERING_SCHEMA_VERSION,
        accepted: true,
        precision,
        layout,
        occupancy,
        subgroup,
        shared_memory: shared,
        fallback_count,
    })
}

fn admit_precision(
    request: &AdmissionRequest<'_>,
    source: Option<&str>,
    fallbacks_out: &mut [FallbackRecord],
    fallback_count: &mut usize,
) -> Result<PrecisionAdmission, ForgeError> {
    if let Some(src) = source {
        if contains_wgsl_f64_type(src) {
            return Err(ForgeError::Validation(
                "native WGSL f64 is not supported; existing df64 is paired f32".to_string(),
            ));
        }
    }
    let contract_model = request.contract.arithmetic;
    match request.precision {
        RequestedPrecision::NativeWgslF64 => {
            let admitted = match contract_model {
                ArithmeticModel::Df64PairedF32 => ArithmeticModel::Df64PairedF32,
                ArithmeticModel::FilteredPredicate => ArithmeticModel::FilteredPredicate,
                _ => ArithmeticModel::IeeeF32,
            };
            push_fallback(
                fallbacks_out,
                fallback_count,
                FallbackRecord {
                    kind: FallbackKind::Precision,
                    requested: "native_wgsl_f64",
                    admitted: arithmetic_name(admitted),
                    note: "WGSL has no f64; df64 is paired f32 double-single, not native f64",
                },
            )?;
            Ok(PrecisionAdmission::Fallback { model: admitted })
        }
        RequestedPrecision::Df64PairedF32 => {
            if contract_model == ArithmeticModel::Df64PairedF32 {
                Ok(PrecisionAdmission::Admitted {
                    model: ArithmeticModel::Df64PairedF32,
                })
            } else {
                push_fallback(
                    fallbacks_out,
                    fallback_count,
                    FallbackRecord {
                        kind: FallbackKind::Precision,
                        requested: "df64_paired_f32",
                        admitted: arithmetic_name(contract_model),
                        note: "requested df64 layout is unsupported on this kernel; admitted f32",
                    },
                )?;
                Ok(PrecisionAdmission::Fallback {
                    model: contract_model,
                })
            }
        }
        RequestedPrecision::IeeeF32 => {
            if contract_model == ArithmeticModel::Df64PairedF32 {
                push_fallback(
                    fallbacks_out,
                    fallback_count,
                    FallbackRecord {
                        kind: FallbackKind::Precision,
                        requested: "ieee_f32",
                        admitted: arithmetic_name(contract_model),
                        note: "contract arithmetic differs from requested ieee_f32",
                    },
                )?;
                Ok(PrecisionAdmission::Fallback {
                    model: contract_model,
                })
            } else {
                Ok(PrecisionAdmission::Admitted {
                    model: contract_model,
                })
            }
        }
        RequestedPrecision::FilteredPredicate => {
            if contract_model == ArithmeticModel::FilteredPredicate {
                Ok(PrecisionAdmission::Admitted {
                    model: ArithmeticModel::FilteredPredicate,
                })
            } else {
                Err(ForgeError::Validation(
                    "filtered predicate precision is unsupported on this contract".to_string(),
                ))
            }
        }
    }
}

fn admit_layout(
    request: &AdmissionRequest<'_>,
    fallbacks_out: &mut [FallbackRecord],
    fallback_count: &mut usize,
) -> Result<LayoutKind, ForgeError> {
    let derived = layout_from_contract(request.contract);
    if request.layout == derived {
        match request.layout {
            LayoutKind::GeometryPacked15 => require_stride(request.contract, 15, "packed_queries")?,
            LayoutKind::ClusterNode12 => {}
            LayoutKind::Df64PairedF32 => {
                if request.contract.arithmetic != ArithmeticModel::Df64PairedF32 {
                    return Err(ForgeError::Validation(format!(
                        "df64 layout requires Df64PairedF32 arithmetic ({DF64_PAIR_STRIDE} f32 scalars per value), not native WGSL f64"
                    )));
                }
            }
            LayoutKind::ScalarF32 | LayoutKind::IntegerI32 | LayoutKind::IntegerU32 => {}
        }
        return Ok(request.layout);
    }
    if request.layout == LayoutKind::Df64PairedF32 && derived != LayoutKind::Df64PairedF32 {
        push_fallback(
            fallbacks_out,
            fallback_count,
            FallbackRecord {
                kind: FallbackKind::Layout,
                requested: "df64_paired_f32",
                admitted: layout_name(derived),
                note: "df64 pair layout is unsupported here; admitted f32 layout",
            },
        )?;
        return Ok(derived);
    }
    if request.layout == LayoutKind::ClusterNode12 && derived == LayoutKind::GeometryPacked15 {
        return Err(ForgeError::Validation(
            "layout mismatch: cluster-node 12-scalar ABI is not geometry packed-15".to_string(),
        ));
    }
    Err(ForgeError::Validation(format!(
        "layout mismatch: requested {:?} is unsupported for this contract ({:?})",
        request.layout, derived
    )))
}

fn admit_shared_memory(
    request: &AdmissionRequest<'_>,
) -> Result<SharedMemoryAdmission, ForgeError> {
    let wg = request.contract.schedule.effective_workgroup_size();
    let declared_bytes = shared_bytes(&request.contract.kernel, wg)?;
    if declared_bytes > PORTABLE_MAX_SHARED_BYTES {
        return Err(ForgeError::InvalidSchedule(format!(
            "shared memory {declared_bytes} bytes exceeds portable limit {PORTABLE_MAX_SHARED_BYTES}"
        )));
    }
    Ok(SharedMemoryAdmission {
        declared_bytes,
        limit_bytes: PORTABLE_MAX_SHARED_BYTES,
    })
}

fn admit_occupancy(
    request: &AdmissionRequest<'_>,
    shared_bytes: u32,
) -> Result<OccupancyReport, ForgeError> {
    let threads = workgroup_threads(&request.contract.schedule);
    let warp = request.constraints.warp_size.max(1);
    let estimated_waves_in_workgroup = threads.div_ceil(warp);
    match request.occupancy {
        OccupancyAssumption::Undeclared => Err(ForgeError::Validation(
            "undeclared occupancy assumption is rejected; occupancy is an admission/report, not measured GPU timing".to_string(),
        )),
        OccupancyAssumption::None => Ok(OccupancyReport {
            assumed: false,
            workgroup_threads: threads,
            shared_bytes,
            estimated_waves_in_workgroup,
            note: OCCUPANCY_NOTE,
        }),
        OccupancyAssumption::Declared(declared) => {
            if declared.waves_per_cu == 0 {
                return Err(ForgeError::Validation(
                    "declared occupancy waves_per_cu must be nonzero".to_string(),
                ));
            }
            if let Some(bytes) = declared.shared_bytes {
                if bytes != shared_bytes {
                    return Err(ForgeError::Validation(format!(
                        "declared occupancy shared_bytes {bytes} does not match kernel {shared_bytes}"
                    )));
                }
            }
            if let Some(size) = declared.subgroup_size {
                if size != warp {
                    return Err(ForgeError::Validation(format!(
                        "declared occupancy subgroup size {size} does not match adapter warp {}",
                        warp
                    )));
                }
            }
            Ok(OccupancyReport {
                assumed: true,
                workgroup_threads: threads,
                shared_bytes,
                estimated_waves_in_workgroup,
                note: OCCUPANCY_NOTE,
            })
        }
    }
}

fn admit_subgroup(request: &AdmissionRequest<'_>) -> Result<SubgroupAdmission, ForgeError> {
    let required = request.subgroup_required
        || request
            .contract
            .kernel
            .required_intrinsics()
            .iter()
            .any(|intrinsic| intrinsic.class() == IntrinsicClass::Subgroup);
    let adapter_supports = request.constraints.supports_subgroups;
    if required && !adapter_supports {
        return Err(ForgeError::InvalidSchedule(
            "kernel requires subgroups unavailable on this adapter".to_string(),
        ));
    }
    let size = request
        .assumed_subgroup_size
        .unwrap_or(request.constraints.warp_size);
    if let Some(assumed) = request.assumed_subgroup_size {
        if assumed == 0 || assumed != request.constraints.warp_size {
            return Err(ForgeError::Validation(format!(
                "assumed subgroup size {assumed} is undeclared relative to adapter warp {}",
                request.constraints.warp_size
            )));
        }
    }
    Ok(SubgroupAdmission {
        required,
        adapter_supports,
        size,
    })
}

fn shared_bytes(kernel: &KernelSpec, workgroup: u32) -> Result<u32, ForgeError> {
    let mut total = 0u32;
    for spec in &kernel.shared_memory {
        let bytes = spec
            .length
            .resolve(workgroup)
            .checked_mul(spec.element.size_bytes())
            .ok_or_else(|| {
                ForgeError::InvalidKernel("shared-memory length overflow".to_string())
            })?;
        total = total.checked_add(bytes).ok_or_else(|| {
            ForgeError::InvalidKernel("shared-memory byte count overflow".to_string())
        })?;
    }
    Ok(total)
}

fn workgroup_threads(schedule: &Schedule) -> u32 {
    schedule
        .workgroup
        .flat_threads()
        .max(schedule.effective_workgroup_size())
}

fn require_stride(
    contract: &ProductionContract,
    stride: u32,
    tensor: &str,
) -> Result<(), ForgeError> {
    let packed = contract
        .tensors
        .iter()
        .find(|item| item.name == tensor)
        .ok_or_else(|| {
            ForgeError::InvalidKernel(format!("layout missing required tensor {tensor:?}"))
        })?;
    match packed.shape.dims.as_slice() {
        [_, n] if *n == stride => Ok(()),
        [n] if packed.shape.dynamic() || *n % stride == 0 => Ok(()),
        _ => Err(ForgeError::Validation(format!(
            "layout mismatch: tensor {tensor:?} is not {stride}-wide"
        ))),
    }
}

fn push_fallback(
    out: &mut [FallbackRecord],
    count: &mut usize,
    record: FallbackRecord,
) -> Result<(), ForgeError> {
    if *count >= out.len() {
        return Err(ForgeError::Validation(
            "caller fallback buffer is too small".to_string(),
        ));
    }
    out[*count] = record;
    *count += 1;
    Ok(())
}

fn arithmetic_name(model: ArithmeticModel) -> &'static str {
    match model {
        ArithmeticModel::IeeeF32 => "ieee_f32",
        ArithmeticModel::Df64PairedF32 => "df64_paired_f32",
        ArithmeticModel::IntegerI32 => "integer_i32",
        ArithmeticModel::IntegerU32 => "integer_u32",
        ArithmeticModel::FilteredPredicate => "filtered_predicate",
    }
}

fn layout_name(kind: LayoutKind) -> &'static str {
    match kind {
        LayoutKind::ScalarF32 => "scalar_f32",
        LayoutKind::IntegerI32 => "integer_i32",
        LayoutKind::IntegerU32 => "integer_u32",
        LayoutKind::Df64PairedF32 => "df64_paired_f32",
        LayoutKind::GeometryPacked15 => "geometry_packed_15",
        LayoutKind::ClusterNode12 => "cluster_node_12",
    }
}

pub(crate) fn contains_wgsl_f64_type(source: &str) -> bool {
    source.lines().any(|line| {
        let code = line.split_once("//").map(|(code, _)| code).unwrap_or(line);
        code.contains("<f64>") || code.contains(" f64")
    })
}
