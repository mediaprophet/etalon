//! Math / discretization / schedule / ABI / evidence layers over existing `KernelSpec`.

use serde::{Deserialize, Serialize};

use crate::ir::{
    BufferAccess, BufferElement, BufferSpec, ForgeError, KernelSpec, Op, ScalarType, SharedLen,
    SharedMemorySpec,
};
use crate::schedule::{AdapterConstraints, Schedule};

use super::delimited_blake3;

pub const PRODUCTION_IR_SCHEMA_V1: &[u8] = b"etalon-production-ir-v1";
pub const PRODUCTION_IR_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathFamily {
    Reduction,
    SparseOperator,
    GeometrySelection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArithmeticModel {
    IeeeF32,
    /// Double-single paired f32. Existing df64 is this model, not native WGSL f64.
    Df64PairedF32,
    IntegerI32,
    IntegerU32,
    FilteredPredicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalUnit {
    Dimensionless,
    Length,
    Area,
    Volume,
    Angle,
    Index,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Effects {
    pub read: bool,
    pub write: bool,
    pub atomic: bool,
    pub barrier: bool,
}

impl Effects {
    pub const fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            atomic: false,
            barrier: false,
        }
    }

    pub const fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            atomic: false,
            barrier: false,
        }
    }

    pub const fn atomic_write() -> Self {
        Self {
            read: true,
            write: true,
            atomic: true,
            barrier: false,
        }
    }

    pub const fn shared_barrier() -> Self {
        Self {
            read: true,
            write: true,
            atomic: false,
            barrier: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeAnnotation {
    /// `0` means runtime-parameterized, matching existing graph IR.
    pub dims: Vec<u32>,
}

impl ShapeAnnotation {
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dynamic(&self) -> bool {
        self.dims.iter().any(|&d| d == 0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorRole {
    pub name: String,
    pub buffer: String,
    pub shape: ShapeAnnotation,
    pub element: BufferElement,
    pub unit: PhysicalUnit,
    pub effects: Effects,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AliasingKind {
    MayAlias,
    MustNotAlias,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AliasingDecl {
    pub left: String,
    pub right: String,
    pub kind: AliasingKind,
}

impl AliasingDecl {
    pub fn covers(&self, a: &str, b: &str) -> bool {
        (self.left == a && self.right == b) || (self.left == b && self.right == a)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssumptionKind {
    RealAlgebra,
    FloatingPoint,
    GeometryFilter,
    Memory,
    Schedule,
    Domain,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Assumption {
    pub id: String,
    pub kind: AssumptionKind,
    pub statement: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReductionKind {
    Sum,
    Max,
    Mean,
    L2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseKind {
    ScatterAdd,
    ScatterMax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryKind {
    PointInTetrahedron,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathOp {
    Reduction {
        kind: ReductionKind,
        axis: u8,
    },
    Sparse {
        kind: SparseKind,
    },
    Geometry {
        kind: GeometryKind,
        uncertain_code: i32,
        outside_code: i32,
        inside_code: i32,
    },
}

impl MathOp {
    pub fn family(&self) -> MathFamily {
        match self {
            Self::Reduction { .. } => MathFamily::Reduction,
            Self::Sparse { .. } => MathFamily::SparseOperator,
            Self::Geometry { .. } => MathFamily::GeometrySelection,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Discretization {
    TreeReduceShared,
    /// Occupancy-bounded CAS on f32 bit patterns. Not a hot-path optimizer.
    AtomicCasBitcast,
    FilteredOrient3d,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductionContract {
    pub schema_version: u32,
    pub kernel: KernelSpec,
    pub math: MathOp,
    pub discretization: Discretization,
    pub schedule: Schedule,
    pub tensors: Vec<TensorRole>,
    pub aliasing: Vec<AliasingDecl>,
    pub assumptions: Vec<Assumption>,
    pub arithmetic: ArithmeticModel,
}

impl ProductionContract {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        kernel: KernelSpec,
        math: MathOp,
        discretization: Discretization,
        schedule: Schedule,
        tensors: Vec<TensorRole>,
        aliasing: Vec<AliasingDecl>,
        assumptions: Vec<Assumption>,
        arithmetic: ArithmeticModel,
    ) -> Result<Self, ForgeError> {
        let contract = Self {
            schema_version: PRODUCTION_IR_SCHEMA_VERSION,
            kernel,
            math,
            discretization,
            schedule,
            tensors,
            aliasing,
            assumptions,
            arithmetic,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn reduction_sum(schedule: Schedule) -> Result<Self, ForgeError> {
        let kernel = KernelSpec {
            id: "production-reduce-sum".to_string(),
            semantic_version: 1,
            entry_point: "reduce_main".to_string(),
            description: "Workgroup tree reduction (sum) over existing Forge IR".to_string(),
            buffers: vec![
                buffer(0, "input", ScalarType::F32, BufferAccess::StorageRead),
                buffer(
                    1,
                    "output",
                    ScalarType::F32,
                    BufferAccess::StorageReadWrite,
                ),
                buffer(2, "params", ScalarType::U32, BufferAccess::StorageRead),
            ],
            ops: vec![
                Op::Loop {
                    induction_var: "i".to_string(),
                    start: "tid".to_string(),
                    end: "n".to_string(),
                    step: "workgroup_size".to_string(),
                    body: vec![
                        Op::Load {
                            buffer: "input".to_string(),
                            index: "i".to_string(),
                            destination: "x".to_string(),
                        },
                        Op::Add {
                            left: "acc".to_string(),
                            right: "x".to_string(),
                            destination: "acc".to_string(),
                        },
                    ],
                },
                Op::Barrier,
                Op::Store {
                    buffer: "output".to_string(),
                    index: "zero".to_string(),
                    value: "acc".to_string(),
                },
            ],
            shared_memory: vec![SharedMemorySpec {
                name: "scratch".to_string(),
                element: ScalarType::F32,
                length: SharedLen::WorkgroupSize,
            }],
        };
        Self::try_new(
            kernel,
            MathOp::Reduction {
                kind: ReductionKind::Sum,
                axis: 0,
            },
            Discretization::TreeReduceShared,
            schedule,
            vec![
                TensorRole {
                    name: "values".to_string(),
                    buffer: "input".to_string(),
                    shape: ShapeAnnotation { dims: vec![0] },
                    element: BufferElement::Scalar(ScalarType::F32),
                    unit: PhysicalUnit::Dimensionless,
                    effects: Effects::read_only(),
                },
                TensorRole {
                    name: "reduced".to_string(),
                    buffer: "output".to_string(),
                    shape: ShapeAnnotation { dims: vec![1] },
                    element: BufferElement::Scalar(ScalarType::F32),
                    unit: PhysicalUnit::Dimensionless,
                    effects: Effects::read_write(),
                },
                TensorRole {
                    name: "count".to_string(),
                    buffer: "params".to_string(),
                    shape: ShapeAnnotation { dims: vec![1] },
                    element: BufferElement::Scalar(ScalarType::U32),
                    unit: PhysicalUnit::Index,
                    effects: Effects::read_only(),
                },
                TensorRole {
                    name: "scratch".to_string(),
                    buffer: "scratch".to_string(),
                    shape: ShapeAnnotation { dims: vec![0] },
                    element: BufferElement::Scalar(ScalarType::F32),
                    unit: PhysicalUnit::Dimensionless,
                    effects: Effects::shared_barrier(),
                },
            ],
            Vec::new(),
            vec![Assumption {
                id: "ieee_f32".to_string(),
                kind: AssumptionKind::FloatingPoint,
                statement: "reduction uses IEEE f32; real-algebra associativity is not assumed"
                    .to_string(),
            }],
            ArithmeticModel::IeeeF32,
        )
    }

    pub fn sparse_scatter_add(schedule: Schedule) -> Result<Self, ForgeError> {
        let kernel = KernelSpec {
            id: "production-scatter-add".to_string(),
            semantic_version: 1,
            entry_point: "scatter_main".to_string(),
            description: "Sparse scatter-add via occupancy-bounded CAS over existing Forge IR"
                .to_string(),
            buffers: vec![
                buffer(0, "src", ScalarType::F32, BufferAccess::StorageRead),
                buffer(1, "idx", ScalarType::U32, BufferAccess::StorageRead),
                buffer(
                    2,
                    "output",
                    ScalarType::U32,
                    BufferAccess::StorageReadWrite,
                ),
                buffer(3, "params", ScalarType::U32, BufferAccess::StorageRead),
            ],
            ops: vec![
                Op::Load {
                    buffer: "src".to_string(),
                    index: "p".to_string(),
                    destination: "val".to_string(),
                },
                Op::Load {
                    buffer: "idx".to_string(),
                    index: "p".to_string(),
                    destination: "t".to_string(),
                },
                Op::Store {
                    buffer: "output".to_string(),
                    index: "t".to_string(),
                    value: "val".to_string(),
                },
            ],
            shared_memory: Vec::new(),
        };
        Self::try_new(
            kernel,
            MathOp::Sparse {
                kind: SparseKind::ScatterAdd,
            },
            Discretization::AtomicCasBitcast,
            schedule,
            vec![
                TensorRole {
                    name: "sources".to_string(),
                    buffer: "src".to_string(),
                    shape: ShapeAnnotation { dims: vec![0] },
                    element: BufferElement::Scalar(ScalarType::F32),
                    unit: PhysicalUnit::Dimensionless,
                    effects: Effects::read_only(),
                },
                TensorRole {
                    name: "indices".to_string(),
                    buffer: "idx".to_string(),
                    shape: ShapeAnnotation { dims: vec![0] },
                    element: BufferElement::Scalar(ScalarType::U32),
                    unit: PhysicalUnit::Index,
                    effects: Effects::read_only(),
                },
                TensorRole {
                    name: "accum".to_string(),
                    buffer: "output".to_string(),
                    shape: ShapeAnnotation { dims: vec![0] },
                    element: BufferElement::Scalar(ScalarType::U32),
                    unit: PhysicalUnit::Dimensionless,
                    effects: Effects::atomic_write(),
                },
                TensorRole {
                    name: "counts".to_string(),
                    buffer: "params".to_string(),
                    shape: ShapeAnnotation { dims: vec![4] },
                    element: BufferElement::Scalar(ScalarType::U32),
                    unit: PhysicalUnit::Index,
                    effects: Effects::read_only(),
                },
            ],
            Vec::new(),
            vec![Assumption {
                id: "cas_occupancy_bounded".to_string(),
                kind: AssumptionKind::Memory,
                statement: "atomicCompareExchangeWeak retries are occupancy-bounded, not an unbounded optimizer".to_string(),
            }],
            ArithmeticModel::IeeeF32,
        )
    }

    pub fn geometry_point_in_tetra(schedule: Schedule) -> Result<Self, ForgeError> {
        let kernel = KernelSpec {
            id: "production-point-in-tetra".to_string(),
            semantic_version: 1,
            entry_point: "point_in_tetra_f32".to_string(),
            description: "Filtered point-in-tetrahedron selection over existing Forge IR"
                .to_string(),
            buffers: vec![
                buffer(0, "data", ScalarType::F32, BufferAccess::StorageRead),
                buffer(
                    1,
                    "result_out",
                    ScalarType::I32,
                    BufferAccess::StorageReadWrite,
                ),
                buffer(2, "params", ScalarType::U32, BufferAccess::Uniform),
            ],
            ops: Vec::new(),
            shared_memory: Vec::new(),
        };
        Self::try_new(
            kernel,
            MathOp::Geometry {
                kind: GeometryKind::PointInTetrahedron,
                uncertain_code: 2,
                outside_code: 0,
                inside_code: 1,
            },
            Discretization::FilteredOrient3d,
            schedule,
            vec![
                TensorRole {
                    name: "packed_queries".to_string(),
                    buffer: "data".to_string(),
                    shape: ShapeAnnotation { dims: vec![0, 15] },
                    element: BufferElement::Scalar(ScalarType::F32),
                    unit: PhysicalUnit::Length,
                    effects: Effects::read_only(),
                },
                TensorRole {
                    name: "selection".to_string(),
                    buffer: "result_out".to_string(),
                    shape: ShapeAnnotation { dims: vec![0] },
                    element: BufferElement::Scalar(ScalarType::I32),
                    unit: PhysicalUnit::Dimensionless,
                    effects: Effects::read_write(),
                },
                TensorRole {
                    name: "query_count".to_string(),
                    buffer: "params".to_string(),
                    shape: ShapeAnnotation { dims: vec![1] },
                    element: BufferElement::Scalar(ScalarType::U32),
                    unit: PhysicalUnit::Index,
                    effects: Effects::read_only(),
                },
            ],
            Vec::new(),
            vec![Assumption {
                id: "filtered_uncertain_retained".to_string(),
                kind: AssumptionKind::GeometryFilter,
                statement: "near-zero orient3d remains uncertain; exact adjudication is the caller"
                    .to_string(),
            }],
            ArithmeticModel::FilteredPredicate,
        )
    }

    pub fn with_assumption(mut self, assumption: Assumption) -> Result<Self, ForgeError> {
        if !self.assumptions.iter().any(|item| item.id == assumption.id) {
            self.assumptions.push(assumption);
        }
        self.validate()?;
        Ok(self)
    }

    pub fn has_assumption(&self, id: &str) -> bool {
        self.assumptions.iter().any(|item| item.id == id)
    }

    pub fn family(&self) -> MathFamily {
        self.math.family()
    }

    pub fn validate(&self) -> Result<(), ForgeError> {
        self.validate_with(&AdapterConstraints::portable())
    }

    pub fn validate_with(&self, constraints: &AdapterConstraints) -> Result<(), ForgeError> {
        if self.schema_version != PRODUCTION_IR_SCHEMA_VERSION {
            return Err(ForgeError::Validation(format!(
                "unsupported production-IR schema {}",
                self.schema_version
            )));
        }
        self.kernel.validate()?;
        self.schedule.validate(&self.kernel, constraints)?;
        if self.tensors.is_empty() {
            return Err(ForgeError::InvalidKernel(
                "production contract requires tensor annotations".to_string(),
            ));
        }
        self.validate_layers()?;
        self.validate_tensors()?;
        self.validate_aliasing()?;
        self.validate_math_shapes()?;
        Ok(())
    }

    pub fn canonical_hash(&self) -> Result<String, ForgeError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        Ok(delimited_blake3(
            PRODUCTION_IR_SCHEMA_V1,
            &[
                &PRODUCTION_IR_SCHEMA_VERSION.to_le_bytes(),
                bytes.as_slice(),
            ],
        ))
    }

    pub fn abi_hash(&self) -> Result<String, ForgeError> {
        let bytes = serde_json::to_vec(&(
            &self.kernel.buffers,
            &self.kernel.shared_memory,
            &self.tensors,
            &self.aliasing,
        ))?;
        Ok(delimited_blake3(
            PRODUCTION_IR_SCHEMA_V1,
            &[bytes.as_slice()],
        ))
    }

    pub fn options_hash(&self) -> Result<String, ForgeError> {
        let bytes = serde_json::to_vec(&(
            &self.schedule,
            self.arithmetic,
            self.discretization,
            &self.assumptions,
        ))?;
        Ok(delimited_blake3(PRODUCTION_IR_SCHEMA_V1, &[bytes.as_slice()]))
    }

    fn validate_layers(&self) -> Result<(), ForgeError> {
        match (&self.math, self.discretization, self.arithmetic) {
            (MathOp::Reduction { .. }, Discretization::TreeReduceShared, ArithmeticModel::IeeeF32)
            | (
                MathOp::Reduction { .. },
                Discretization::TreeReduceShared,
                ArithmeticModel::Df64PairedF32,
            ) => {}
            (MathOp::Sparse { .. }, Discretization::AtomicCasBitcast, ArithmeticModel::IeeeF32) => {}
            (
                MathOp::Geometry { .. },
                Discretization::FilteredOrient3d,
                ArithmeticModel::FilteredPredicate,
            ) => {}
            _ => {
                return Err(ForgeError::InvalidKernel(format!(
                    "math {:?}, discretization {:?}, arithmetic {:?} are not a legal layering",
                    self.math.family(),
                    self.discretization,
                    self.arithmetic
                )));
            }
        }
        if matches!(self.math, MathOp::Reduction { .. }) && !contains_barrier(&self.kernel.ops) {
            return Err(ForgeError::InvalidKernel(
                "tree reduction requires a Barrier in existing IR".to_string(),
            ));
        }
        if matches!(self.math, MathOp::Sparse { .. })
            && !self
                .tensors
                .iter()
                .any(|tensor| tensor.buffer == "output" && tensor.effects.atomic)
        {
            return Err(ForgeError::InvalidKernel(
                "sparse scatter requires an atomic effect on output".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_tensors(&self) -> Result<(), ForgeError> {
        for tensor in &self.tensors {
            if tensor.name.is_empty() || tensor.buffer.is_empty() {
                return Err(ForgeError::InvalidKernel(
                    "tensor name and buffer must not be empty".to_string(),
                ));
            }
            if tensor.buffer == "scratch" {
                if !self
                    .kernel
                    .shared_memory
                    .iter()
                    .any(|shared| shared.name == tensor.buffer)
                {
                    return Err(ForgeError::InvalidKernel(format!(
                        "tensor {:?} has no shared-memory buffer {:?}",
                        tensor.name, tensor.buffer
                    )));
                }
                continue;
            }
            let Some(spec) = self
                .kernel
                .buffers
                .iter()
                .find(|buffer| buffer.name == tensor.buffer)
            else {
                return Err(ForgeError::InvalidKernel(format!(
                    "tensor {:?} references unknown buffer {:?}",
                    tensor.name, tensor.buffer
                )));
            };
            if spec.element != tensor.element {
                return Err(ForgeError::InvalidKernel(format!(
                    "tensor {:?} element {:?} does not match buffer {:?}",
                    tensor.name, tensor.element, spec.element
                )));
            }
            if tensor.effects.write
                && matches!(spec.access, BufferAccess::StorageRead | BufferAccess::Uniform)
            {
                return Err(ForgeError::InvalidKernel(format!(
                    "tensor {:?} writes buffer {:?} with access {:?}",
                    tensor.name, tensor.buffer, spec.access
                )));
            }
            if tensor.effects.atomic && !tensor.effects.write {
                return Err(ForgeError::InvalidKernel(format!(
                    "tensor {:?} declares atomic without write",
                    tensor.name
                )));
            }
        }
        Ok(())
    }

    fn validate_aliasing(&self) -> Result<(), ForgeError> {
        for decl in &self.aliasing {
            if decl.left == decl.right {
                return Err(ForgeError::InvalidKernel(
                    "aliasing declaration must name two tensors".to_string(),
                ));
            }
            if !self.tensors.iter().any(|tensor| tensor.name == decl.left)
                || !self.tensors.iter().any(|tensor| tensor.name == decl.right)
            {
                return Err(ForgeError::InvalidKernel(format!(
                    "aliasing declaration {:?}/{:?} names an unknown tensor",
                    decl.left, decl.right
                )));
            }
        }
        for (i, left) in self.tensors.iter().enumerate() {
            for right in self.tensors.iter().skip(i + 1) {
                let same_named_buffer = left.buffer == right.buffer;
                let same_binding = match (
                    buffer_slot(&self.kernel, &left.buffer),
                    buffer_slot(&self.kernel, &right.buffer),
                ) {
                    (Some(a), Some(b)) => a == b && left.buffer != right.buffer,
                    _ => false,
                };
                if !same_named_buffer && !same_binding {
                    continue;
                }
                let declared = self
                    .aliasing
                    .iter()
                    .find(|decl| decl.covers(&left.name, &right.name));
                match declared.map(|decl| decl.kind) {
                    Some(AliasingKind::MayAlias) => {}
                    Some(AliasingKind::MustNotAlias) => {
                        return Err(ForgeError::InvalidKernel(format!(
                            "tensors {:?} and {:?} share a binding but are declared MustNotAlias",
                            left.name, right.name
                        )));
                    }
                    None => {
                        return Err(ForgeError::InvalidKernel(format!(
                            "undeclared aliasing: tensors {:?} and {:?} share @group/binding without a MayAlias declaration",
                            left.name, right.name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_math_shapes(&self) -> Result<(), ForgeError> {
        match &self.math {
            MathOp::Reduction { axis, .. } => {
                let input = required_tensor(self, "values")?;
                if input.shape.rank() == 0 {
                    return Err(ForgeError::InvalidKernel(
                        "reduction input must have rank >= 1".to_string(),
                    ));
                }
                if usize::from(*axis) >= input.shape.rank() {
                    return Err(ForgeError::InvalidKernel(format!(
                        "reduction axis {axis} is outside rank {}",
                        input.shape.rank()
                    )));
                }
            }
            MathOp::Sparse { .. } => {
                let src = required_tensor(self, "sources")?;
                let idx = required_tensor(self, "indices")?;
                if src.shape.dims != idx.shape.dims {
                    return Err(ForgeError::InvalidKernel(
                        "sparse sources and indices must have identical shapes".to_string(),
                    ));
                }
                if idx.unit != PhysicalUnit::Index {
                    return Err(ForgeError::InvalidKernel(
                        "sparse indices must be annotated as Index".to_string(),
                    ));
                }
            }
            MathOp::Geometry {
                uncertain_code,
                outside_code,
                inside_code,
                ..
            } => {
                if uncertain_code == outside_code
                    || uncertain_code == inside_code
                    || outside_code == inside_code
                {
                    return Err(ForgeError::InvalidKernel(
                        "geometry selection codes must be distinct".to_string(),
                    ));
                }
                let packed = required_tensor(self, "packed_queries")?;
                match packed.shape.dims.as_slice() {
                    [_, 15] => {}
                    [n] if packed.shape.dynamic() || *n % 15 == 0 => {}
                    _ => {
                        return Err(ForgeError::InvalidKernel(
                            "point-in-tetra packed queries must be 15-wide (p,a,b,c,d)".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn buffer(binding: u32, name: &str, scalar: ScalarType, access: BufferAccess) -> BufferSpec {
    BufferSpec {
        group: 0,
        binding,
        name: name.to_string(),
        element: BufferElement::Scalar(scalar),
        access,
    }
}

fn buffer_slot(kernel: &KernelSpec, name: &str) -> Option<(u32, u32)> {
    kernel
        .buffers
        .iter()
        .find(|buffer| buffer.name == name)
        .map(|buffer| (buffer.group, buffer.binding))
}

fn required_tensor<'a>(
    contract: &'a ProductionContract,
    name: &str,
) -> Result<&'a TensorRole, ForgeError> {
    contract
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| {
            ForgeError::InvalidKernel(format!("contract missing required tensor {name:?}"))
        })
}

fn contains_barrier(ops: &[Op]) -> bool {
    ops.iter().any(|op| match op {
        Op::Barrier => true,
        Op::Loop { body, .. } => contains_barrier(body),
        _ => false,
    })
}
