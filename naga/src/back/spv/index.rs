/*!
Bounds-checking for SPIR-V output.
*/

use super::{
    helpers::{global_needs_wrapper, map_storage_class},
    selection::Selection,
    Block, BlockContext, Error, IdGenerator, Instruction, Word,
};
use crate::{
    arena::Handle,
    proc::{index::GuardedIndex, BoundsCheckPolicy},
};

/// The results of performing a bounds check.
///
/// On success, [`write_bounds_check`](BlockContext::write_bounds_check)
/// returns a value of this type. The caller can assume that the right
/// policy has been applied, and simply do what the variant says.
#[derive(Debug)]
pub(super) enum BoundsCheckResult {
    /// The index is statically known and in bounds, with the given value.
    KnownInBounds(u32),

    /// The given instruction computes the index to be used.
    ///
    /// When [`BoundsCheckPolicy::Restrict`] is in force, this is a
    /// clamped version of the index the user supplied.
    ///
    /// When [`BoundsCheckPolicy::Unchecked`] is in force, this is
    /// simply the index the user supplied. This variant indicates
    /// that we couldn't prove statically that the index was in
    /// bounds; otherwise we would have returned [`KnownInBounds`].
    ///
    /// [`KnownInBounds`]: BoundsCheckResult::KnownInBounds
    Computed(Word),

    /// The given instruction computes a boolean condition which is true
    /// if the index is in bounds.
    ///
    /// This is returned when [`BoundsCheckPolicy::ReadZeroSkipWrite`]
    /// is in force.
    Conditional {
        /// The access should only be permitted if this value is true.
        condition_id: Word,

        /// The access should use this index value.
        index_id: Word,
    },
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use crate::{
        back::spv,
        proc::{BoundsCheckPolicies, BoundsCheckPolicy},
    };
    use alloc::vec::Vec;

    fn instructions(words: &[u32]) -> Vec<&[u32]> {
        let mut remaining = &words[5..];
        let mut instructions = Vec::new();
        while !remaining.is_empty() {
            let count = (remaining[0] >> 16) as usize;
            assert!(count != 0 && count <= remaining.len());
            let (instruction, rest) = remaining.split_at(count);
            instructions.push(instruction);
            remaining = rest;
        }
        instructions
    }

    fn is_op(instruction: &[u32], op: spirv::Op) -> bool {
        instruction[0] & 0xffff == op as u32
    }

    #[test]
    fn binding_array_runtime_bounds_use_buffer_length_and_guard_metadata() {
        for binding_size in [", 8", ""] {
            for policy in [
                BoundsCheckPolicy::Restrict,
                BoundsCheckPolicy::ReadZeroSkipWrite,
            ] {
                let source = alloc::format!(
                    r"
                    enable etalon_binding_array;
                    struct Data {{ inner: array<u32> }}
                    @group(0) @binding(0)
                    var<storage, read_write> buffers: binding_array<Data{binding_size}>;
                    @compute @workgroup_size(1)
                    fn main(@builtin(global_invocation_id) id: vec3u) {{
                        let descriptor = id.x + 8u;
                        let element = id.y + 16u;
                        buffers[descriptor].inner[element] = buffers[descriptor].inner[element] + 1u;
                    }}
                "
                );
                let module = crate::front::wgsl::parse_str(&source).unwrap();
                let info = crate::valid::Validator::new(
                    crate::valid::ValidationFlags::all(),
                    crate::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                let mut options = spv::Options {
                    bounds_check_policies: BoundsCheckPolicies {
                        buffer: policy,
                        binding_array: policy,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                options.binding_map.insert(
                    crate::ResourceBinding {
                        group: 0,
                        binding: 0,
                    },
                    spv::BindingInfo {
                        descriptor_set: 0,
                        binding: 0,
                        binding_array_size: Some(8),
                        ..Default::default()
                    },
                );
                let words = spv::write_vec(&module, &info, &options, None).unwrap();
                let instructions = instructions(&words);
                let lengths: Vec<_> = instructions
                    .iter()
                    .copied()
                    .filter(|inst| is_op(inst, spirv::Op::ArrayLength))
                    .collect();
                assert!(
                    !lengths.is_empty(),
                    "inner arrays require OpArrayLength, not descriptor count"
                );
                for length in lengths {
                    let pointer = instructions
                        .iter()
                        .copied()
                        .find(|inst| is_op(inst, spirv::Op::AccessChain) && inst[2] == length[3])
                        .unwrap();
                    assert!(instructions
                        .iter()
                        .any(|inst| is_op(inst, spirv::Op::Decorate)
                            && inst[1] == pointer[2]
                            && inst[2] == spirv::Decoration::NonUniform as u32));
                    match policy {
                        BoundsCheckPolicy::Restrict => {
                            let clamp = instructions
                                .iter()
                                .copied()
                                .find(|inst| {
                                    is_op(inst, spirv::Op::ExtInst) && inst[2] == pointer[4]
                                })
                                .unwrap();
                            assert_eq!(clamp[4], spirv::GlslStd450Op::UMin as u32);
                            assert!(instructions.iter().any(|inst| is_op(
                                inst,
                                spirv::Op::Constant
                            ) && inst[2] == clamp[6]
                                && inst[3] == 7));
                            let max = instructions
                                .iter()
                                .copied()
                                .find(|inst| is_op(inst, spirv::Op::ISub) && inst[3] == length[2])
                                .unwrap();
                            assert!(instructions.iter().any(|inst| is_op(
                                inst,
                                spirv::Op::ExtInst
                            ) && inst[4]
                                == spirv::GlslStd450Op::UMin as u32
                                && inst[6] == max[2]));
                        }
                        BoundsCheckPolicy::ReadZeroSkipWrite => {
                            let pointer_position = instructions
                                .iter()
                                .position(|inst| inst.as_ptr() == pointer.as_ptr())
                                .unwrap();
                            let label = instructions[..pointer_position]
                                .iter()
                                .rev()
                                .find(|inst| is_op(inst, spirv::Op::Label))
                                .unwrap()[1];
                            let branch = instructions
                                .iter()
                                .copied()
                                .find(|inst| {
                                    is_op(inst, spirv::Op::BranchConditional) && inst[2] == label
                                })
                                .unwrap();
                            let check = instructions
                                .iter()
                                .copied()
                                .find(|inst| {
                                    is_op(inst, spirv::Op::ULessThan) && inst[2] == branch[1]
                                })
                                .unwrap();
                            assert_eq!(check[3], pointer[4]);
                            assert!(instructions.iter().any(|inst| is_op(
                                inst,
                                spirv::Op::Constant
                            ) && inst[2] == check[4]
                                && inst[3] == 8));
                            let merged = instructions
                                .iter()
                                .copied()
                                .find(|inst| {
                                    is_op(inst, spirv::Op::Phi)
                                        && inst[3..]
                                            .chunks_exact(2)
                                            .any(|pair| pair[0] == length[2])
                                })
                                .unwrap();
                            assert!(instructions
                                .iter()
                                .any(|inst| is_op(inst, spirv::Op::ULessThan)
                                    && inst[4] == merged[2]));
                        }
                        BoundsCheckPolicy::Unchecked => unreachable!(),
                    }
                }
            }
        }
    }
}

/// A value that we either know at translation time, or need to compute at runtime.
#[derive(Copy, Clone)]
pub(super) enum MaybeKnown<T> {
    /// The value is known at shader translation time.
    Known(T),

    /// The value is computed by the instruction with the given id.
    Computed(Word),
}

impl BlockContext<'_> {
    /// Emit code to compute the length of a run-time array.
    ///
    /// Given `array`, an expression referring a runtime-sized array, return the
    /// instruction id for the array's length.
    ///
    /// Runtime-sized arrays may only appear in the values of global
    /// variables, which must have one of the following Naga types:
    ///
    /// 1. A runtime-sized array.
    /// 2. A struct whose last member is a runtime-sized array.
    /// 3. A binding array of 2.
    ///
    /// Thus, the expression `array` has the form of:
    ///
    /// - An optional [`AccessIndex`], for case 2, applied to...
    /// - An optional [`Access`] or [`AccessIndex`], for case 3, applied to...
    /// - A [`GlobalVariable`].
    ///
    /// The generated SPIR-V takes into account wrapped globals; see
    /// [`back::spv::GlobalVariable`] for details.
    ///
    /// [`GlobalVariable`]: crate::Expression::GlobalVariable
    /// [`AccessIndex`]: crate::Expression::AccessIndex
    /// [`Access`]: crate::Expression::Access
    /// [`base`]: crate::Expression::Access::base
    /// [`back::spv::GlobalVariable`]: super::GlobalVariable
    pub(super) fn write_runtime_array_length(
        &mut self,
        array: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        // The index into the binding array, if any.
        let binding_array_index: Option<(Handle<crate::Expression>, GuardedIndex)>;

        // The handle to the Naga IR global we're referring to.
        let global_handle: Handle<crate::GlobalVariable>;

        // At the Naga type level, if the runtime-sized array is the final member of a
        // struct, this is that member's index.
        //
        // This does not cover wrappers: if this backend wrapped the Naga global's
        // type in a synthetic SPIR-V struct (see `global_needs_wrapper`), this is
        // `None`.
        let opt_last_member_index: Option<u32>;

        // Inspect `array` and decide whether we have a binding array and/or an
        // enclosing struct.
        match self.ir_function.expressions[array] {
            crate::Expression::AccessIndex { base, index } => {
                match self.ir_function.expressions[base] {
                    crate::Expression::AccessIndex {
                        base: base_outer,
                        index: index_outer,
                    } => match self.ir_function.expressions[base_outer] {
                        // An `AccessIndex` of an `AccessIndex` must be a
                        // binding array holding structs whose last members are
                        // runtime-sized arrays.
                        crate::Expression::GlobalVariable(handle) => {
                            binding_array_index =
                                Some((base_outer, GuardedIndex::Known(index_outer)));
                            global_handle = handle;
                            opt_last_member_index = Some(index);
                        }
                        _ => {
                            return Err(Error::Validation(
                                "array length expression: AccessIndex(AccessIndex(Global))",
                            ))
                        }
                    },
                    crate::Expression::Access {
                        base: base_outer,
                        index: index_outer,
                    } => match self.ir_function.expressions[base_outer] {
                        // Similarly, an `AccessIndex` of an `Access` must be a
                        // binding array holding structs whose last members are
                        // runtime-sized arrays.
                        crate::Expression::GlobalVariable(handle) => {
                            binding_array_index =
                                Some((base_outer, GuardedIndex::Expression(index_outer)));
                            global_handle = handle;
                            opt_last_member_index = Some(index);
                        }
                        _ => {
                            return Err(Error::Validation(
                                "array length expression: AccessIndex(Access(Global))",
                            ))
                        }
                    },
                    crate::Expression::GlobalVariable(handle) => {
                        // An outer `AccessIndex` applied directly to a
                        // `GlobalVariable`. Since binding arrays can only contain
                        // structs, this must be referring to the last member of a
                        // struct that is a runtime-sized array.
                        binding_array_index = None;
                        global_handle = handle;
                        opt_last_member_index = Some(index);
                    }
                    _ => {
                        return Err(Error::Validation(
                            "array length expression: AccessIndex(<unexpected>)",
                        ))
                    }
                }
            }
            crate::Expression::GlobalVariable(handle) => {
                // A direct reference to a global variable. This must hold the
                // runtime-sized array directly.
                binding_array_index = None;
                global_handle = handle;
                opt_last_member_index = None;
            }
            _ => return Err(Error::Validation("array length expression case-4")),
        };

        // The verifier should have checked this, but make sure the inspection above
        // agrees with the type about whether a binding array is involved.
        //
        // Eventually we do want to support `binding_array<array<T>>`. This check
        // ensures that whoever relaxes the validator will get an error message from
        // us, not just bogus SPIR-V.
        let global = &self.ir_module.global_variables[global_handle];
        match (&self.ir_module.types[global.ty].inner, binding_array_index) {
            (&crate::TypeInner::BindingArray { .. }, Some(_)) => {}
            (_, None) => {}
            _ => {
                return Err(Error::Validation(
                    "array length expression: bad binding array inference",
                ))
            }
        }

        // SPIR-V allows runtime-sized arrays to appear only as the last member of a
        // struct. Determine this member's index.
        let gvar = self.writer.global_variables[global_handle].clone();
        let global = &self.ir_module.global_variables[global_handle];
        let needs_wrapper = global_needs_wrapper(self.ir_module, global);
        let (last_member_index, gvar_id) = match (opt_last_member_index, needs_wrapper) {
            (Some(index), false) => {
                // At the Naga type level, the runtime-sized array appears as the
                // final member of a struct, whose index is `index`. We didn't need to
                // wrap this, since the Naga type meets SPIR-V's requirements already.
                (index, gvar.access_id)
            }
            (None, true) => {
                // At the Naga type level, the runtime-sized array does not appear
                // within a struct. We wrapped this in an OpTypeStruct with nothing
                // else in it, so the index is zero. OpArrayLength wants the pointer
                // to the wrapper struct, so use `gvar.var_id`.
                (0, gvar.var_id)
            }
            _ => {
                return Err(Error::Validation(
                    "array length expression: bad SPIR-V wrapper struct inference",
                ));
            }
        };

        let (binding_array_index_id, condition, non_uniform) = match binding_array_index {
            Some((base, index)) => {
                let non_uniform = match index {
                    GuardedIndex::Expression(index) => {
                        self.fun_info[index].uniformity.non_uniform_result.is_some()
                    }
                    GuardedIndex::Known(_) => false,
                };
                match self.write_bounds_check(base, index, block)? {
                    BoundsCheckResult::KnownInBounds(index) => {
                        (Some(self.get_index_constant(index)), None, non_uniform)
                    }
                    BoundsCheckResult::Computed(index) => (Some(index), None, non_uniform),
                    BoundsCheckResult::Conditional {
                        condition_id,
                        index_id,
                    } => (Some(index_id), Some(condition_id), non_uniform),
                }
            }
            None => (None, None, false),
        };
        let u32_type_id = self.writer.get_u32_type_id();
        let mut selection = Selection::start(block, u32_type_id);
        if let Some(condition) = condition {
            let zero = self.writer.get_constant_null(u32_type_id);
            selection.if_true(self, condition, zero);
        }
        let block = selection.block();
        let structure_id = match binding_array_index_id {
            // We are indexing inside a binding array, generate the access op.
            Some(index_id) => {
                let element_type_id = match self.ir_module.types[global.ty].inner {
                    crate::TypeInner::BindingArray { base, size: _ } => {
                        let base_id = self.get_handle_type_id(base);
                        let class = map_storage_class(global.space);
                        self.get_pointer_type_id(base_id, class)
                    }
                    _ => return Err(Error::Validation("array length expression case-5")),
                };
                let structure_id = self.gen_id();
                block.body.push(Instruction::access_chain(
                    element_type_id,
                    structure_id,
                    gvar_id,
                    &[index_id],
                ));
                if non_uniform {
                    self.writer
                        .decorate_non_uniform_binding_array_access(structure_id)?;
                }
                structure_id
            }
            None => gvar_id,
        };
        let length_id = self.gen_id();
        block.body.push(Instruction::array_length(
            u32_type_id,
            length_id,
            structure_id,
            last_member_index,
        ));

        Ok(selection.finish(self, length_id))
    }

    /// If `sequence` refers to an unbounded binding array global, return its
    /// layout size from the SPIR-V binding map.
    fn binding_array_layout_size(&self, sequence: Handle<crate::Expression>) -> Option<u32> {
        let sequence_ty = self.fun_info[sequence].ty.inner_with(&self.ir_module.types);
        let sequence_ty = match *sequence_ty {
            crate::TypeInner::Pointer { base, .. } => &self.ir_module.types[base].inner,
            _ => sequence_ty,
        };
        // A buffer's runtime array must use its own length, not its descriptor array's length.
        if !matches!(sequence_ty, crate::TypeInner::BindingArray { .. }) {
            return None;
        }
        let global_handle = match self.ir_function.expressions[sequence] {
            crate::Expression::GlobalVariable(handle) => handle,
            _ => self.ir_function.originating_global(sequence)?,
        };
        let global = &self.ir_module.global_variables[global_handle];
        let crate::TypeInner::BindingArray { .. } = self.ir_module.types[global.ty].inner else {
            return None;
        };
        let binding = global.binding?;
        let bind_target = self.writer.resolve_resource_binding(&binding).ok()?;
        bind_target.binding_array_size
    }

    /// Compute the length of a subscriptable value.
    ///
    /// Given `sequence`, an expression referring to some indexable type, return
    /// its length. The result may either be computed by SPIR-V instructions, or
    /// known at shader translation time.
    ///
    /// `sequence` may be a `Vector`, `Matrix`, or `Array`, a `Pointer` to any
    /// of those, or a `ValuePointer`. An array may be fixed-size, dynamically
    /// sized, or use a specializable constant as its length.
    fn write_sequence_length(
        &mut self,
        sequence: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<MaybeKnown<u32>, Error> {
        let sequence_ty = self.fun_info[sequence].ty.inner_with(&self.ir_module.types);
        match sequence_ty.indexable_length_resolved(self.ir_module) {
            Ok(crate::proc::IndexableLength::Known(known_length)) => {
                Ok(MaybeKnown::Known(known_length))
            }
            Ok(crate::proc::IndexableLength::Dynamic) => {
                if let Some(size) = self.binding_array_layout_size(sequence) {
                    return Ok(MaybeKnown::Known(size));
                }
                let length_id = self.write_runtime_array_length(sequence, block)?;
                Ok(MaybeKnown::Computed(length_id))
            }
            Err(err) => {
                log::error!("Sequence length for {sequence:?} failed: {err}");
                Err(Error::Validation("indexable length"))
            }
        }
    }

    /// Compute the maximum valid index of a subscriptable value.
    ///
    /// Given `sequence`, an expression referring to some indexable type, return
    /// its maximum valid index - one less than its length. The result may
    /// either be computed, or known at shader translation time.
    ///
    /// `sequence` may be a `Vector`, `Matrix`, or `Array`, a `Pointer` to any
    /// of those, or a `ValuePointer`. An array may be fixed-size, dynamically
    /// sized, or use a specializable constant as its length.
    fn write_sequence_max_index(
        &mut self,
        sequence: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<MaybeKnown<u32>, Error> {
        match self.write_sequence_length(sequence, block)? {
            MaybeKnown::Known(known_length) => {
                // We should have thrown out all attempts to subscript zero-length
                // sequences during validation, so the following subtraction should never
                // underflow.
                assert!(known_length > 0);
                // Compute the max index from the length now.
                Ok(MaybeKnown::Known(known_length - 1))
            }
            MaybeKnown::Computed(length_id) => {
                // Emit code to compute the max index from the length.
                let const_one_id = self.get_index_constant(1);
                let max_index_id = self.gen_id();
                block.body.push(Instruction::binary(
                    spirv::Op::ISub,
                    self.writer.get_u32_type_id(),
                    max_index_id,
                    length_id,
                    const_one_id,
                ));
                Ok(MaybeKnown::Computed(max_index_id))
            }
        }
    }

    /// Restrict an index to be in range for a vector, matrix, or array.
    ///
    /// This is used to implement `BoundsCheckPolicy::Restrict`. An in-bounds
    /// index is left unchanged. An out-of-bounds index is replaced with some
    /// arbitrary in-bounds index. Note,this is not necessarily clamping; for
    /// example, negative indices might be changed to refer to the last element
    /// of the sequence, not the first, as clamping would do.
    ///
    /// Either return the restricted index value, if known, or add instructions
    /// to `block` to compute it, and return the id of the result. See the
    /// documentation for `BoundsCheckResult` for details.
    ///
    /// The `sequence` expression may be a `Vector`, `Matrix`, or `Array`, a
    /// `Pointer` to any of those, or a `ValuePointer`. An array may be
    /// fixed-size, dynamically sized, or use a specializable constant as its
    /// length.
    pub(super) fn write_restricted_index(
        &mut self,
        sequence: Handle<crate::Expression>,
        index: GuardedIndex,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        let max_index = self.write_sequence_max_index(sequence, block)?;

        // If both are known, we can compute the index to be used
        // right now.
        if let (GuardedIndex::Known(index), MaybeKnown::Known(max_index)) = (index, max_index) {
            let restricted = core::cmp::min(index, max_index);
            return Ok(BoundsCheckResult::KnownInBounds(restricted));
        }

        let index_id = match index {
            GuardedIndex::Known(value) => self.get_index_constant(value),
            GuardedIndex::Expression(expr) => self.cached[expr],
        };

        let max_index_id = match max_index {
            MaybeKnown::Known(value) => self.get_index_constant(value),
            MaybeKnown::Computed(id) => id,
        };

        // One or the other of the index or length is dynamic, so emit code for
        // BoundsCheckPolicy::Restrict.
        let restricted_index_id = self.gen_id();
        block.body.push(Instruction::ext_inst_gl_op(
            self.writer.gl450_ext_inst_id,
            spirv::GlslStd450Op::UMin,
            self.writer.get_u32_type_id(),
            restricted_index_id,
            &[index_id, max_index_id],
        ));
        Ok(BoundsCheckResult::Computed(restricted_index_id))
    }

    /// Write an index bounds comparison to `block`, if needed.
    ///
    /// This is used to implement [`BoundsCheckPolicy::ReadZeroSkipWrite`].
    ///
    /// If we're able to determine statically that `index` is in bounds for
    /// `sequence`, return `KnownInBounds(value)`, where `value` is the actual
    /// value of the index. (In principle, one could know that the index is in
    /// bounds without knowing its specific value, but in our simple-minded
    /// situation, we always know it.)
    ///
    /// If instead we must generate code to perform the comparison at run time,
    /// return `Conditional(comparison_id)`, where `comparison_id` is an
    /// instruction producing a boolean value that is true if `index` is in
    /// bounds for `sequence`.
    ///
    /// The `sequence` expression may be a `Vector`, `Matrix`, or `Array`, a
    /// `Pointer` to any of those, or a `ValuePointer`. An array may be
    /// fixed-size, dynamically sized, or use a specializable constant as its
    /// length.
    fn write_index_comparison(
        &mut self,
        sequence: Handle<crate::Expression>,
        index: GuardedIndex,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        let length = self.write_sequence_length(sequence, block)?;

        // If both are known, we can decide whether the index is in
        // bounds right now.
        if let (GuardedIndex::Known(index), MaybeKnown::Known(length)) = (index, length) {
            if index < length {
                return Ok(BoundsCheckResult::KnownInBounds(index));
            }

            // In theory, when `index` is bad, we could return a new
            // `KnownOutOfBounds` variant here. But it's simpler just to fall
            // through and let the bounds check take place. The shader is broken
            // anyway, so it doesn't make sense to invest in emitting the ideal
            // code for it.
        }

        let index_id = match index {
            GuardedIndex::Known(value) => self.get_index_constant(value),
            GuardedIndex::Expression(expr) => self.cached[expr],
        };

        let length_id = match length {
            MaybeKnown::Known(value) => self.get_index_constant(value),
            MaybeKnown::Computed(id) => id,
        };

        // Compare the index against the length.
        let condition_id = self.gen_id();
        block.body.push(Instruction::binary(
            spirv::Op::ULessThan,
            self.writer.get_bool_type_id(),
            condition_id,
            index_id,
            length_id,
        ));

        // Indicate that we did generate the check.
        Ok(BoundsCheckResult::Conditional {
            condition_id,
            index_id,
        })
    }

    /// Emit a conditional load for `BoundsCheckPolicy::ReadZeroSkipWrite`.
    ///
    /// Generate code to load a value of `result_type` if `condition` is true,
    /// and generate a null value of that type if it is false. Call `emit_load`
    /// to emit the instructions to perform the load. Return the id of the
    /// merged value of the two branches.
    pub(super) fn write_conditional_indexed_load<F>(
        &mut self,
        result_type: Word,
        condition: Word,
        block: &mut Block,
        emit_load: F,
    ) -> Word
    where
        F: FnOnce(&mut IdGenerator, &mut Block) -> Word,
    {
        // For the out-of-bounds case, we produce a zero value.
        let null_id = self.writer.get_constant_null(result_type);

        let mut selection = Selection::start(block, result_type);

        // As it turns out, we don't actually need a full 'if-then-else'
        // structure for this: SPIR-V constants are declared up front, so the
        // 'else' block would have no instructions. Instead we emit something
        // like this:
        //
        //     result = zero;
        //     if in_bounds {
        //         result = do the load;
        //     }
        //     use result;

        // Continue only if the index was in bounds. Otherwise, branch to the
        // merge block.
        selection.if_true(self, condition, null_id);

        // The in-bounds path. Perform the access and the load.
        let loaded_value = emit_load(&mut self.writer.id_gen, selection.block());

        selection.finish(self, loaded_value)
    }

    /// Emit code for bounds checks for an array, vector, or matrix access.
    ///
    /// This tries to handle all the critical steps for bounds checks:
    ///
    /// - First, select the appropriate bounds check policy for `base`,
    ///   depending on its address space.
    ///
    /// - Next, analyze `index` to see if its value is known at
    ///   compile time, in which case we can decide statically whether
    ///   the index is in bounds.
    ///
    /// - If the index's value is not known at compile time, emit code to:
    ///
    ///     - restrict its value (for [`BoundsCheckPolicy::Restrict`]), or
    ///
    ///     - check whether it's in bounds (for
    ///       [`BoundsCheckPolicy::ReadZeroSkipWrite`]).
    ///
    /// Return a [`BoundsCheckResult`] indicating how the index should be
    /// consumed. See that type's documentation for details.
    pub(super) fn write_bounds_check(
        &mut self,
        base: Handle<crate::Expression>,
        mut index: GuardedIndex,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        // If the value of `index` is known at compile time, find it now.
        index.try_resolve_to_constant(&self.ir_function.expressions, self.ir_module);

        let policy = self.writer.bounds_check_policies.choose_policy(
            base,
            &self.ir_module.types,
            self.fun_info,
        );

        Ok(match policy {
            BoundsCheckPolicy::Restrict => self.write_restricted_index(base, index, block)?,
            BoundsCheckPolicy::ReadZeroSkipWrite => {
                self.write_index_comparison(base, index, block)?
            }
            BoundsCheckPolicy::Unchecked => match index {
                GuardedIndex::Known(value) => BoundsCheckResult::KnownInBounds(value),
                GuardedIndex::Expression(expr) => BoundsCheckResult::Computed(self.cached[expr]),
            },
        })
    }

    /// Emit code to subscript a vector by value with a computed index.
    ///
    /// Return the id of the element value.
    ///
    /// If `base_id_override` is provided, it is used as the vector expression
    /// to be subscripted into, rather than the cached value of `base`.
    pub(super) fn write_vector_access(
        &mut self,
        result_type_id: Word,
        base: Handle<crate::Expression>,
        base_id_override: Option<Word>,
        index: GuardedIndex,
        block: &mut Block,
    ) -> Result<Word, Error> {
        let base_id = base_id_override.unwrap_or_else(|| self.cached[base]);

        let result_id = match self.write_bounds_check(base, index, block)? {
            BoundsCheckResult::KnownInBounds(known_index) => {
                let result_id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    result_type_id,
                    result_id,
                    base_id,
                    &[known_index],
                ));
                result_id
            }
            BoundsCheckResult::Computed(computed_index_id) => {
                let result_id = self.gen_id();
                block.body.push(Instruction::vector_extract_dynamic(
                    result_type_id,
                    result_id,
                    base_id,
                    computed_index_id,
                ));
                result_id
            }
            BoundsCheckResult::Conditional {
                condition_id,
                index_id,
            } => {
                // Run-time bounds checks were required. Emit
                // conditional load.
                self.write_conditional_indexed_load(
                    result_type_id,
                    condition_id,
                    block,
                    |id_gen, block| {
                        // The in-bounds path. Generate the access.
                        let element_id = id_gen.next();
                        block.body.push(Instruction::vector_extract_dynamic(
                            result_type_id,
                            element_id,
                            base_id,
                            index_id,
                        ));
                        element_id
                    },
                )
            }
        };

        Ok(result_id)
    }
}
