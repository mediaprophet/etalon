use super::GeneratedShader;
use crate::numerical::{FORGE_SCHEMA_VERSION, ForgeError, KernelSpec, Schedule};
use std::fmt::Write;
pub fn emit_wgsl(kernel: &KernelSpec, schedule: Schedule) -> Result<GeneratedShader, ForgeError> {
    super::emit_canonical(kernel, schedule, super::TargetBackend::Wgsl)
}
struct WgslGraphLowerer<'a> {
    source: &'a mut String,
}

impl crate::numerical::ir::graph::Lowerer for WgslGraphLowerer<'_> {
    fn reduce(&mut self, node: &crate::numerical::ir::graph::GraphNode) -> Result<(), ForgeError> {
        use crate::numerical::ir::graph::OpNode;
        if let OpNode::Reduce { op, .. } = node.op {
            self.source
                .push_str(&crate::numerical::graph_ops::reduce::reduce_wgsl(
                    op,
                    node.sched.workgroup_size,
                ));
            Ok(())
        } else {
            Err(ForgeError::Emission(
                "WgslGraphLowerer::reduce received a non-Reduce node".to_string(),
            ))
        }
    }

    fn broadcast(
        &mut self,
        node: &crate::numerical::ir::graph::GraphNode,
    ) -> Result<(), ForgeError> {
        self.source
            .push_str(&crate::numerical::graph_ops::broadcast::broadcast_wgsl(
                node.sched.workgroup_size,
            ));
        Ok(())
    }

    fn elementwise(
        &mut self,
        node: &crate::numerical::ir::graph::GraphNode,
    ) -> Result<(), ForgeError> {
        use crate::numerical::ir::graph::OpNode;
        if let OpNode::Elementwise { f } = node.op {
            let src = crate::numerical::graph_ops::elementwise::elementwise_wgsl(
                f,
                node.sched.workgroup_size,
            )?;
            self.source.push_str(&src);
            Ok(())
        } else {
            Err(ForgeError::Emission(
                "WgslGraphLowerer::elementwise received a non-Elementwise node".to_string(),
            ))
        }
    }

    fn gather_dequant(
        &mut self,
        node: &crate::numerical::ir::graph::GraphNode,
    ) -> Result<(), ForgeError> {
        self.source.push_str(
            &crate::numerical::graph_ops::gather_dequant::gather_dequant_ternary_wgsl(
                node.sched.workgroup_size,
            ),
        );
        Ok(())
    }

    fn scatter_accum(
        &mut self,
        node: &crate::numerical::ir::graph::GraphNode,
    ) -> Result<(), ForgeError> {
        self.source
            .push_str(&crate::numerical::graph_ops::scatter::scatter_wgsl(
                node.sched.workgroup_size,
            ));
        Ok(())
    }

    fn slice(&mut self, node: &crate::numerical::ir::graph::GraphNode) -> Result<(), ForgeError> {
        self.source
            .push_str(&crate::numerical::graph_ops::slice::slice_wgsl(
                node.sched.workgroup_size,
            ));
        Ok(())
    }

    fn stencil(&mut self, node: &crate::numerical::ir::graph::GraphNode) -> Result<(), ForgeError> {
        use crate::numerical::ir::graph::OpNode;
        if let OpNode::Stencil { kind, .. } = node.op {
            let src = crate::numerical::graph_ops::stencil::stencil_wgsl(
                kind,
                node.sched.workgroup_size,
            )?;
            self.source.push_str(&src);
            Ok(())
        } else {
            Err(ForgeError::Emission(
                "WgslGraphLowerer::stencil received a non-Stencil node".to_string(),
            ))
        }
    }

    fn rope(&mut self, node: &crate::numerical::ir::graph::GraphNode) -> Result<(), ForgeError> {
        let src = crate::numerical::graph_ops::stencil::rope_wgsl(node.sched.workgroup_size)?;
        self.source.push_str(&src);
        Ok(())
    }

    fn pool2d(&mut self, node: &crate::numerical::ir::graph::GraphNode) -> Result<(), ForgeError> {
        self.source
            .push_str(&crate::numerical::graph_ops::vision::max_pool2d_wgsl(
                node.sched.workgroup_size,
            ));
        Ok(())
    }

    fn resize2d(
        &mut self,
        node: &crate::numerical::ir::graph::GraphNode,
    ) -> Result<(), ForgeError> {
        self.source
            .push_str(&crate::numerical::graph_ops::vision::resize2d_wgsl(
                node.sched.workgroup_size,
            ));
        Ok(())
    }

    fn conv2d(&mut self, node: &crate::numerical::ir::graph::GraphNode) -> Result<(), ForgeError> {
        self.source
            .push_str(&crate::numerical::graph_ops::vision::conv2d_wgsl(
                node.sched.workgroup_size,
            ));
        Ok(())
    }
}

/// Emit a complete WGSL module for a **pure compute-graph** (no backing `KernelSpec`) —
/// the native-node path (Reduce/Broadcast in Phase 2). A single-node graph produces one
/// kernel module; multi-node graph emission (per-node functions + a driver) is Phase 4.
pub fn emit_graph_wgsl(
    graph: &crate::numerical::ir::graph::ComputeGraph,
    schedule: Schedule,
) -> Result<GeneratedShader, ForgeError> {
    let mut source = String::with_capacity(1_024);
    writeln!(
        source,
        "// Generated by Qualia WGSL Forge schema {FORGE_SCHEMA_VERSION} (compute-graph)."
    )
    .map_err(|error| ForgeError::Emission(error.to_string()))?;
    let mut lowerer = WgslGraphLowerer {
        source: &mut source,
    };
    crate::numerical::ir::graph::lower_graph(graph, &mut lowerer)?;
    let source_hash = blake3::hash(source.as_bytes()).to_hex().to_string();
    Ok(GeneratedShader {
        kernel_id: "graph".to_string(),
        semantic_hash: source_hash.clone(),
        source_hash,
        schedule,
        source,
    })
}

#[cfg(test)]
#[path = "migration_legacy_wgsl.rs"]
mod migration_legacy_wgsl;
#[cfg(test)]
include!("wgsl_migration_tests.rs");
