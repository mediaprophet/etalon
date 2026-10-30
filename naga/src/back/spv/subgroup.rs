use super::{Block, BlockContext, Error, Instruction, NumericType};
use crate::{arena::Handle, TypeInner};

fn is_own_subgroup_id(
    module: &crate::Module,
    function: &crate::Function,
    expression: Handle<crate::Expression>,
) -> bool {
    let binding = match function.expressions[expression] {
        crate::Expression::FunctionArgument(index) => {
            function.arguments[index as usize].binding.as_ref()
        }
        crate::Expression::AccessIndex { base, index } => {
            let crate::Expression::FunctionArgument(argument) = function.expressions[base] else {
                return false;
            };
            let TypeInner::Struct { ref members, .. } =
                module.types[function.arguments[argument as usize].ty].inner
            else {
                return false;
            };
            members[index as usize].binding.as_ref()
        }
        _ => None,
    };
    matches!(
        binding,
        Some(crate::Binding::BuiltIn(
            crate::BuiltIn::SubgroupInvocationId
        ))
    )
}

impl BlockContext<'_> {
    pub(super) fn write_subgroup_ballot(
        &mut self,
        predicate: &Option<Handle<crate::Expression>>,
        result: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        self.writer.require_any(
            "GroupNonUniformBallot",
            &[spirv::Capability::GroupNonUniformBallot],
        )?;
        let vec4_u32_type_id = self.get_numeric_type_id(NumericType::Vector {
            size: crate::VectorSize::Quad,
            scalar: crate::Scalar::U32,
        });
        let exec_scope_id = self.get_index_constant(spirv::Scope::Subgroup as u32);
        let predicate = if let Some(predicate) = *predicate {
            self.cached[predicate]
        } else {
            self.writer.get_constant_scalar(crate::Literal::Bool(true))
        };
        let id = self.gen_id();
        block.body.push(Instruction::group_non_uniform_ballot(
            vec4_u32_type_id,
            id,
            exec_scope_id,
            predicate,
        ));
        self.cached[result] = id;
        Ok(())
    }
    pub(super) fn write_subgroup_operation(
        &mut self,
        op: &crate::SubgroupOperation,
        collective_op: &crate::CollectiveOperation,
        argument: Handle<crate::Expression>,
        result: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        use crate::SubgroupOperation as sg;
        match *op {
            sg::All | sg::Any => {
                self.writer.require_any(
                    "GroupNonUniformVote",
                    &[spirv::Capability::GroupNonUniformVote],
                )?;
            }
            _ => {
                self.writer.require_any(
                    "GroupNonUniformArithmetic",
                    &[spirv::Capability::GroupNonUniformArithmetic],
                )?;
            }
        }

        let id = self.gen_id();
        let result_ty = &self.fun_info[result].ty;
        let result_type_id = self.get_expression_type_id(result_ty);
        let result_ty_inner = result_ty.inner_with(&self.ir_module.types);

        let (is_scalar, scalar) = match *result_ty_inner {
            TypeInner::Scalar(kind) => (true, kind),
            TypeInner::Vector { scalar: kind, .. } => (false, kind),
            _ => unimplemented!(),
        };

        use crate::ScalarKind as sk;
        let spirv_op = match (scalar.kind, *op) {
            (sk::Bool, sg::All) if is_scalar => spirv::Op::GroupNonUniformAll,
            (sk::Bool, sg::Any) if is_scalar => spirv::Op::GroupNonUniformAny,
            (_, sg::All | sg::Any) => unimplemented!(),

            (sk::Sint | sk::Uint, sg::Add) => spirv::Op::GroupNonUniformIAdd,
            (sk::Float, sg::Add) => spirv::Op::GroupNonUniformFAdd,
            (sk::Sint | sk::Uint, sg::Mul) => spirv::Op::GroupNonUniformIMul,
            (sk::Float, sg::Mul) => spirv::Op::GroupNonUniformFMul,
            (sk::Sint, sg::Max) => spirv::Op::GroupNonUniformSMax,
            (sk::Uint, sg::Max) => spirv::Op::GroupNonUniformUMax,
            (sk::Float, sg::Max) => spirv::Op::GroupNonUniformFMax,
            (sk::Sint, sg::Min) => spirv::Op::GroupNonUniformSMin,
            (sk::Uint, sg::Min) => spirv::Op::GroupNonUniformUMin,
            (sk::Float, sg::Min) => spirv::Op::GroupNonUniformFMin,
            (_, sg::Add | sg::Mul | sg::Min | sg::Max) => unimplemented!(),

            (sk::Sint | sk::Uint, sg::And) => spirv::Op::GroupNonUniformBitwiseAnd,
            (sk::Sint | sk::Uint, sg::Or) => spirv::Op::GroupNonUniformBitwiseOr,
            (sk::Sint | sk::Uint, sg::Xor) => spirv::Op::GroupNonUniformBitwiseXor,
            (sk::Bool, sg::And) => spirv::Op::GroupNonUniformLogicalAnd,
            (sk::Bool, sg::Or) => spirv::Op::GroupNonUniformLogicalOr,
            (sk::Bool, sg::Xor) => spirv::Op::GroupNonUniformLogicalXor,
            (_, sg::And | sg::Or | sg::Xor) => unimplemented!(),
        };

        let exec_scope_id = self.get_index_constant(spirv::Scope::Subgroup as u32);

        use crate::CollectiveOperation as c;
        let group_op = match *op {
            sg::All | sg::Any => None,
            _ => Some(match *collective_op {
                c::Reduce => spirv::GroupOperation::Reduce,
                c::InclusiveScan => spirv::GroupOperation::InclusiveScan,
                c::ExclusiveScan => spirv::GroupOperation::ExclusiveScan,
            }),
        };

        let arg_id = self.cached[argument];
        block.body.push(Instruction::group_non_uniform_arithmetic(
            spirv_op,
            result_type_id,
            id,
            exec_scope_id,
            group_op,
            arg_id,
        ));
        self.cached[result] = id;
        Ok(())
    }
    pub(super) fn write_subgroup_gather(
        &mut self,
        mode: &crate::GatherMode,
        argument: Handle<crate::Expression>,
        result: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        match *mode {
            crate::GatherMode::Shuffle(index)
                if is_own_subgroup_id(self.ir_module, self.ir_function, index) =>
            {
                // An active invocation gathering itself already has the evaluated scalar/vector value.
                self.cached[result] = self.cached[argument];
                return Ok(());
            }
            _ => {}
        }
        match *mode {
            crate::GatherMode::BroadcastFirst => {
                self.writer.require_any(
                    "GroupNonUniformBallot",
                    &[spirv::Capability::GroupNonUniformBallot],
                )?;
            }
            crate::GatherMode::Shuffle(_)
            | crate::GatherMode::ShuffleXor(_)
            | crate::GatherMode::Broadcast(_) => {
                self.writer.require_any(
                    "GroupNonUniformShuffle",
                    &[spirv::Capability::GroupNonUniformShuffle],
                )?;
            }
            crate::GatherMode::ShuffleDown(_) | crate::GatherMode::ShuffleUp(_) => {
                self.writer.require_any(
                    "GroupNonUniformShuffleRelative",
                    &[spirv::Capability::GroupNonUniformShuffleRelative],
                )?;
            }
            crate::GatherMode::QuadBroadcast(_) | crate::GatherMode::QuadSwap(_) => {
                self.writer.require_any(
                    "GroupNonUniformQuad",
                    &[spirv::Capability::GroupNonUniformQuad],
                )?;
            }
        }

        let id = self.gen_id();
        let result_ty = &self.fun_info[result].ty;
        let result_type_id = self.get_expression_type_id(result_ty);

        let exec_scope_id = self.get_index_constant(spirv::Scope::Subgroup as u32);

        let arg_id = self.cached[argument];
        match *mode {
            crate::GatherMode::BroadcastFirst => {
                block
                    .body
                    .push(Instruction::group_non_uniform_broadcast_first(
                        result_type_id,
                        id,
                        exec_scope_id,
                        arg_id,
                    ));
            }
            crate::GatherMode::Broadcast(index)
            | crate::GatherMode::Shuffle(index)
            | crate::GatherMode::ShuffleDown(index)
            | crate::GatherMode::ShuffleUp(index)
            | crate::GatherMode::ShuffleXor(index)
            | crate::GatherMode::QuadBroadcast(index) => {
                let index_id = self.cached[index];
                let op = match *mode {
                    crate::GatherMode::BroadcastFirst => unreachable!(),
                    // Use shuffle to emit broadcast to allow the index to
                    // be dynamically uniform on Vulkan 1.1. The argument to
                    // OpGroupNonUniformBroadcast must be a constant pre SPIR-V
                    // 1.5 (vulkan 1.2)
                    crate::GatherMode::Broadcast(_) => spirv::Op::GroupNonUniformShuffle,
                    crate::GatherMode::Shuffle(_) => spirv::Op::GroupNonUniformShuffle,
                    crate::GatherMode::ShuffleDown(_) => spirv::Op::GroupNonUniformShuffleDown,
                    crate::GatherMode::ShuffleUp(_) => spirv::Op::GroupNonUniformShuffleUp,
                    crate::GatherMode::ShuffleXor(_) => spirv::Op::GroupNonUniformShuffleXor,
                    crate::GatherMode::QuadBroadcast(_) => spirv::Op::GroupNonUniformQuadBroadcast,
                    crate::GatherMode::QuadSwap(_) => unreachable!(),
                };
                block.body.push(Instruction::group_non_uniform_gather(
                    op,
                    result_type_id,
                    id,
                    exec_scope_id,
                    arg_id,
                    index_id,
                ));
            }
            crate::GatherMode::QuadSwap(direction) => {
                let direction = self.get_index_constant(match direction {
                    crate::Direction::X => 0,
                    crate::Direction::Y => 1,
                    crate::Direction::Diagonal => 2,
                });
                block.body.push(Instruction::group_non_uniform_quad_swap(
                    result_type_id,
                    id,
                    exec_scope_id,
                    arg_id,
                    direction,
                ));
            }
        }
        self.cached[result] = id;
        Ok(())
    }
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use crate::back::spv::{write_vec, Options, PipelineOptions};
    use alloc::vec::Vec;

    fn opcodes(words: &[u32]) -> Vec<u16> {
        let mut result = Vec::new();
        let mut cursor = 5;
        while cursor < words.len() {
            result.push(words[cursor] as u16);
            cursor += (words[cursor] >> 16) as usize;
        }
        result
    }

    #[test]
    fn own_lane_shuffle_preserves_values_and_input_evaluation() {
        for parameter in ["@builtin(subgroup_invocation_id) lane: u32", "input: Input"] {
            let lane = if parameter == "input: Input" {
                "input.lane"
            } else {
                "lane"
            };
            let source = alloc::format!(
                r#"
                struct Input {{ @builtin(subgroup_invocation_id) lane: u32 }};
                @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
                @group(0) @binding(1) var<storage, read_write> output: array<vec4<u32>>;
                fn value() -> u32 {{ return atomicAdd(&counter, 1u); }}
                @compute @workgroup_size(32) fn main({parameter}) {{
                    let scalar = subgroupShuffle(value(), {lane});
                    let vector = subgroupShuffle(vec4<u32>(value(), 3u, 5u, 7u), {lane});
                    let other = subgroupShuffle(vec4<u32>({lane}), {lane} ^ 1u);
                    output[{lane}] = vector + other + vec4<u32>(scalar);
                }}
            "#
            );
            let module = crate::front::wgsl::parse_str(&source).unwrap();
            let info = crate::valid::Validator::new(
                crate::valid::ValidationFlags::all(),
                crate::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let words = write_vec(
                &module,
                &info,
                &Options {
                    lang_version: (1, 3),
                    ..Default::default()
                },
                Some(&PipelineOptions {
                    shader_stage: crate::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
            let ops = opcodes(&words);
            assert_eq!(
                ops.iter()
                    .filter(|&&op| op == spirv::Op::GroupNonUniformShuffle as u16)
                    .count(),
                1
            );
            assert_eq!(
                ops.iter()
                    .filter(|&&op| op == spirv::Op::FunctionCall as u16)
                    .count(),
                2
            );
            assert_eq!(
                ops.iter()
                    .filter(|&&op| op == spirv::Op::AtomicIAdd as u16)
                    .count(),
                1
            );
        }
    }
}
