//! `enable …;` extensions in WGSL.
//!
//! The focal point of this module is the [`EnableExtension`] API.

use crate::front::wgsl::{Error, Result};
use crate::Span;

use alloc::boxed::Box;

/// Tracks the status of every enable-extension known to Naga.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EnableExtensions {
    etalon_mesh_shader: bool,
    etalon_ray_query: bool,
    etalon_ray_query_vertex_return: bool,
    etalon_ray_tracing_pipeline: bool,
    dual_source_blending: bool,
    /// Whether `enable f16;` was written earlier in the shader module.
    f16: bool,
    /// Whether `enable etalon_int16;` was written earlier in the shader module.
    etalon_int16: bool,
    clip_distances: bool,
    etalon_cooperative_matrix: bool,
    draw_index: bool,
    primitive_index: bool,
    per_vertex: bool,
    etalon_binding_array: bool,
    debug_printf: bool,
}

impl EnableExtensions {
    pub(crate) const fn empty() -> Self {
        Self {
            etalon_mesh_shader: false,
            etalon_ray_query: false,
            etalon_ray_query_vertex_return: false,
            etalon_ray_tracing_pipeline: false,
            f16: false,
            etalon_int16: false,
            dual_source_blending: false,
            clip_distances: false,
            etalon_cooperative_matrix: false,
            draw_index: false,
            primitive_index: false,
            per_vertex: false,
            etalon_binding_array: false,
            debug_printf: false,
        }
    }

    /// Add an enable-extension to the set requested by a module.
    pub(crate) const fn add(&mut self, ext: ImplementedEnableExtension) {
        let field = match ext {
            ImplementedEnableExtension::etalonMeshShader => &mut self.etalon_mesh_shader,
            ImplementedEnableExtension::etalonRayQuery => &mut self.etalon_ray_query,
            ImplementedEnableExtension::etalonRayQueryVertexReturn => {
                &mut self.etalon_ray_query_vertex_return
            }
            ImplementedEnableExtension::etalonRayTracingPipeline => {
                &mut self.etalon_ray_tracing_pipeline
            }
            ImplementedEnableExtension::DualSourceBlending => &mut self.dual_source_blending,
            ImplementedEnableExtension::F16 => &mut self.f16,
            ImplementedEnableExtension::etalonInt16 => &mut self.etalon_int16,
            ImplementedEnableExtension::ClipDistances => &mut self.clip_distances,
            ImplementedEnableExtension::etalonCooperativeMatrix => &mut self.etalon_cooperative_matrix,
            ImplementedEnableExtension::DrawIndex => &mut self.draw_index,
            ImplementedEnableExtension::PrimitiveIndex => &mut self.primitive_index,
            ImplementedEnableExtension::etalonPerVertex => &mut self.per_vertex,
            ImplementedEnableExtension::etalonBindingArray => &mut self.etalon_binding_array,
            ImplementedEnableExtension::etalonDebugPrintf => &mut self.debug_printf,
        };
        *field = true;
    }

    /// Query whether an enable-extension tracked here has been requested.
    pub(crate) const fn contains(&self, ext: ImplementedEnableExtension) -> bool {
        match ext {
            ImplementedEnableExtension::etalonMeshShader => self.etalon_mesh_shader,
            ImplementedEnableExtension::etalonRayQuery => self.etalon_ray_query,
            ImplementedEnableExtension::etalonRayQueryVertexReturn => {
                self.etalon_ray_query_vertex_return
            }
            ImplementedEnableExtension::etalonRayTracingPipeline => self.etalon_ray_tracing_pipeline,
            ImplementedEnableExtension::DualSourceBlending => self.dual_source_blending,
            ImplementedEnableExtension::F16 => self.f16,
            ImplementedEnableExtension::etalonInt16 => self.etalon_int16,
            ImplementedEnableExtension::ClipDistances => self.clip_distances,
            ImplementedEnableExtension::etalonCooperativeMatrix => self.etalon_cooperative_matrix,
            ImplementedEnableExtension::DrawIndex => self.draw_index,
            ImplementedEnableExtension::PrimitiveIndex => self.primitive_index,
            ImplementedEnableExtension::etalonPerVertex => self.per_vertex,
            ImplementedEnableExtension::etalonBindingArray => self.etalon_binding_array,
            ImplementedEnableExtension::etalonDebugPrintf => self.debug_printf,
        }
    }

    pub(crate) fn require(
        &self,
        ext: ImplementedEnableExtension,
        span: Span,
    ) -> Result<'static, ()> {
        if !self.contains(ext) {
            Err(Box::new(Error::EnableExtensionNotEnabled {
                span,
                kind: ext.into(),
            }))
        } else {
            Ok(())
        }
    }
}

impl Default for EnableExtensions {
    fn default() -> Self {
        Self::empty()
    }
}

/// An enable-extension not guaranteed to be present in all environments.
///
/// WGSL spec.: <https://www.w3.org/TR/WGSL/#enable-extensions-sec>
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum EnableExtension {
    Implemented(ImplementedEnableExtension),
    Unimplemented(UnimplementedEnableExtension),
}

impl From<ImplementedEnableExtension> for EnableExtension {
    fn from(value: ImplementedEnableExtension) -> Self {
        Self::Implemented(value)
    }
}

impl EnableExtension {
    const F16: &'static str = "f16";
    const CLIP_DISTANCES: &'static str = "clip_distances";
    const DUAL_SOURCE_BLENDING: &'static str = "dual_source_blending";
    const MESH_SHADER: &'static str = "etalon_mesh_shader";
    const RAY_QUERY: &'static str = "etalon_ray_query";
    const RAY_QUERY_VERTEX_RETURN: &'static str = "etalon_ray_query_vertex_return";
    const RAY_TRACING_PIPELINE: &'static str = "etalon_ray_tracing_pipeline";
    const COOPERATIVE_MATRIX: &'static str = "etalon_cooperative_matrix";
    const SUBGROUPS: &'static str = "subgroups";
    const PRIMITIVE_INDEX: &'static str = "primitive_index";
    const DRAW_INDEX: &'static str = "draw_index";
    const PER_VERTEX: &'static str = "etalon_per_vertex";
    const BINDING_ARRAY: &'static str = "etalon_binding_array";
    const INT16: &'static str = "etalon_int16";
    const DEBUG_PRINTF: &'static str = "etalon_debug_printf";

    /// Convert from a sentinel word in WGSL into its associated [`EnableExtension`], if possible.
    pub(crate) fn from_ident(word: &str, span: Span) -> Result<'_, Self> {
        Ok(match word {
            Self::F16 => Self::Implemented(ImplementedEnableExtension::F16),
            Self::CLIP_DISTANCES => Self::Implemented(ImplementedEnableExtension::ClipDistances),
            Self::DUAL_SOURCE_BLENDING => {
                Self::Implemented(ImplementedEnableExtension::DualSourceBlending)
            }
            Self::MESH_SHADER => Self::Implemented(ImplementedEnableExtension::etalonMeshShader),
            Self::RAY_QUERY => Self::Implemented(ImplementedEnableExtension::etalonRayQuery),
            Self::RAY_QUERY_VERTEX_RETURN => {
                Self::Implemented(ImplementedEnableExtension::etalonRayQueryVertexReturn)
            }
            Self::RAY_TRACING_PIPELINE => {
                Self::Implemented(ImplementedEnableExtension::etalonRayTracingPipeline)
            }
            Self::COOPERATIVE_MATRIX => {
                Self::Implemented(ImplementedEnableExtension::etalonCooperativeMatrix)
            }
            Self::SUBGROUPS => Self::Unimplemented(UnimplementedEnableExtension::Subgroups),
            Self::DRAW_INDEX => Self::Implemented(ImplementedEnableExtension::DrawIndex),
            Self::PRIMITIVE_INDEX => Self::Implemented(ImplementedEnableExtension::PrimitiveIndex),
            Self::PER_VERTEX => Self::Implemented(ImplementedEnableExtension::etalonPerVertex),
            Self::BINDING_ARRAY => Self::Implemented(ImplementedEnableExtension::etalonBindingArray),
            Self::INT16 => Self::Implemented(ImplementedEnableExtension::etalonInt16),
            Self::DEBUG_PRINTF => Self::Implemented(ImplementedEnableExtension::etalonDebugPrintf),
            _ => return Err(Box::new(Error::UnknownEnableExtension(span, word))),
        })
    }

    /// Maps this [`EnableExtension`] into the sentinel word associated with it in WGSL.
    pub const fn to_ident(self) -> &'static str {
        match self {
            Self::Implemented(kind) => match kind {
                ImplementedEnableExtension::etalonMeshShader => Self::MESH_SHADER,
                ImplementedEnableExtension::etalonRayQuery => Self::RAY_QUERY,
                ImplementedEnableExtension::etalonRayQueryVertexReturn => {
                    Self::RAY_QUERY_VERTEX_RETURN
                }
                ImplementedEnableExtension::etalonCooperativeMatrix => Self::COOPERATIVE_MATRIX,
                ImplementedEnableExtension::DualSourceBlending => Self::DUAL_SOURCE_BLENDING,
                ImplementedEnableExtension::F16 => Self::F16,
                ImplementedEnableExtension::ClipDistances => Self::CLIP_DISTANCES,
                ImplementedEnableExtension::DrawIndex => Self::DRAW_INDEX,
                ImplementedEnableExtension::PrimitiveIndex => Self::PRIMITIVE_INDEX,
                ImplementedEnableExtension::etalonRayTracingPipeline => Self::RAY_TRACING_PIPELINE,
                ImplementedEnableExtension::etalonPerVertex => Self::PER_VERTEX,
                ImplementedEnableExtension::etalonBindingArray => Self::BINDING_ARRAY,
                ImplementedEnableExtension::etalonInt16 => Self::INT16,
                ImplementedEnableExtension::etalonDebugPrintf => Self::DEBUG_PRINTF,
            },
            Self::Unimplemented(kind) => match kind {
                UnimplementedEnableExtension::Subgroups => Self::SUBGROUPS,
            },
        }
    }
}

/// A variant of [`EnableExtension::Implemented`].
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(test, derive(strum::VariantArray))]
pub enum ImplementedEnableExtension {
    /// Enables `f16`/`half` primitive support in all shader languages.
    ///
    /// In the WGSL standard, this corresponds to [`enable f16;`].
    ///
    /// [`enable f16;`]: https://www.w3.org/TR/WGSL/#extension-f16
    F16,
    /// Enables the `blend_src` attribute in WGSL.
    ///
    /// In the WGSL standard, this corresponds to [`enable dual_source_blending;`].
    ///
    /// [`enable dual_source_blending;`]: https://www.w3.org/TR/WGSL/#extension-dual_source_blending
    DualSourceBlending,
    /// Enables the `clip_distances` variable in WGSL.
    ///
    /// In the WGSL standard, this corresponds to [`enable clip_distances;`].
    ///
    /// [`enable clip_distances;`]: https://www.w3.org/TR/WGSL/#extension-clip_distances
    ClipDistances,
    /// Enables the `etalon_mesh_shader` extension, native only
    etalonMeshShader,
    /// Enables the `etalon_ray_query` extension, native only.
    etalonRayQuery,
    /// Enables the `etalon_ray_query_vertex_return` extension, native only.
    etalonRayQueryVertexReturn,
    /// Enables the `etalon_ray_tracing_pipeline` extension, native only.
    etalonRayTracingPipeline,
    /// Enables the `etalon_cooperative_matrix` extension, native only.
    etalonCooperativeMatrix,
    /// Enables the `draw_index` builtin. Not currently part of the WGSL spec but probably will be at some point.
    DrawIndex,
    /// Enables the `@builtin(primitive_index)` attribute in WGSL.
    ///
    /// In the WGSL standard, this corresponds to [`enable primitive-index;`].
    ///
    /// [`enable primitive-index;`]: https://www.w3.org/TR/WGSL/#extension-primitive_index
    PrimitiveIndex,
    /// Enables the `etalon_per_vertex` extension, allows using `@interpolate(per_vertex)` attribute in WGSL, native only.
    etalonPerVertex,
    /// Enables the `etalon_binding_array` extension, native only.
    etalonBindingArray,
    /// Enables `i16`/`u16` 16-bit integer support in WGSL, native only.
    etalonInt16,
    /// Enables the `etalon_debug_printf` extension, allows using `debugPrintf`, native only.
    etalonDebugPrintf,
}

impl ImplementedEnableExtension {
    /// A slice of all variants of [`ImplementedEnableExtension`].
    pub const VARIANTS: &'static [Self] = &[
        Self::F16,
        Self::DualSourceBlending,
        Self::ClipDistances,
        Self::etalonMeshShader,
        Self::etalonRayQuery,
        Self::etalonRayQueryVertexReturn,
        Self::etalonRayTracingPipeline,
        Self::etalonCooperativeMatrix,
        Self::DrawIndex,
        Self::PrimitiveIndex,
        Self::etalonPerVertex,
        Self::etalonBindingArray,
        Self::etalonInt16,
        Self::etalonDebugPrintf,
    ];

    /// Returns slice of all variants of [`ImplementedEnableExtension`].
    pub const fn all() -> &'static [Self] {
        Self::VARIANTS
    }

    /// Returns the capability required for this enable extension.
    pub const fn capability(self) -> crate::valid::Capabilities {
        use crate::valid::Capabilities as C;
        match self {
            Self::F16 => C::SHADER_FLOAT16,
            Self::DualSourceBlending => C::DUAL_SOURCE_BLENDING,
            Self::ClipDistances => C::CLIP_DISTANCES,
            Self::etalonMeshShader => C::MESH_SHADER,
            Self::etalonRayQuery => C::RAY_QUERY,
            Self::etalonRayQueryVertexReturn => C::RAY_HIT_VERTEX_POSITION,
            Self::etalonCooperativeMatrix => C::COOPERATIVE_MATRIX,
            Self::etalonRayTracingPipeline => C::RAY_TRACING_PIPELINE,
            Self::DrawIndex => C::DRAW_INDEX,
            Self::PrimitiveIndex => C::PRIMITIVE_INDEX,
            Self::etalonPerVertex => C::PER_VERTEX,
            Self::etalonBindingArray => C::BUFFER_BINDING_ARRAY
                .union(C::BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING)
                .union(C::STORAGE_BUFFER_BINDING_ARRAY)
                .union(C::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING)
                .union(C::STORAGE_TEXTURE_BINDING_ARRAY)
                .union(C::STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING)
                .union(C::TEXTURE_AND_SAMPLER_BINDING_ARRAY)
                .union(C::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING)
                .union(C::ACCELERATION_STRUCTURE_BINDING_ARRAY),
            Self::etalonInt16 => C::SHADER_INT16,
            Self::etalonDebugPrintf => C::DEBUG_PRINTF,
        }
    }
}

#[test]
/// Asserts that the manual implementation of VARIANTS is the same as the derived strum version would be
/// while still allowing strum to be a dev-only dependency
fn test_manual_variants_array_is_correct() {
    assert_eq!(
        <ImplementedEnableExtension as strum::VariantArray>::VARIANTS,
        ImplementedEnableExtension::VARIANTS
    );
}

/// A variant of [`EnableExtension::Unimplemented`].
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum UnimplementedEnableExtension {
    /// Enables subgroup built-ins in all languages.
    ///
    /// In the WGSL standard, this corresponds to [`enable subgroups;`].
    ///
    /// [`enable subgroups;`]: https://www.w3.org/TR/WGSL/#extension-subgroups
    Subgroups,
}

impl UnimplementedEnableExtension {
    pub(crate) const fn tracking_issue_num(self) -> u16 {
        match self {
            Self::Subgroups => 5555,
        }
    }
}
