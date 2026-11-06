//! The reusable GPU graph executor ([`ForgeGraphExecutor`]) and its one-shot free function
//! ([`execute_graph`]), plus the [`ResidentWeights`] handle for once-uploaded weights. This is
//! the throughput-pass driver: context reuse + single-encoder deferred submit.

use super::nodes::prepare_node;
use crate::numerical::ForgeError;
use crate::numerical::execute::{BindingUsage, BufferView, GraphPass, WgpuComputeContext};
use crate::numerical::ir::graph::{ComputeGraph, MAX_IN, NodeId};

/// Context capacity (per slab). Large enough for a decode block's tensors held at once.
const EXEC_CAPACITY: usize = 64 << 20;

/// Execute the whole graph on the GPU and read back the final tensor (one-shot). Differs from
/// the composed CPU floor only by f32 GPU arithmetic; certified against
/// [`execute_graph_cpu`](super::execute_graph_cpu).
///
/// This builds a throwaway [`ForgeGraphExecutor`] (one device + slab per call). A caller that
/// runs many graphs — e.g. one decode block per generated token — should instead hold a
/// [`ForgeGraphExecutor`] and call [`ForgeGraphExecutor::run`] per step, so device/slab
/// creation is paid once rather than per call (the throughput pass, plan §8.1).
pub fn execute_graph(graph: &ComputeGraph, externals: &[Vec<f32>]) -> Result<Vec<f32>, ForgeError> {
    ForgeGraphExecutor::new()?.run(graph, externals)
}

/// A reusable GPU graph executor: owns one [`WgpuComputeContext`] (device, queue, and the two
/// slabs created **once**) so running many graphs does not recreate the device per call. The
/// slab is reset between calls ([`ForgeGraphExecutor::run`]), so each call gets the full
/// capacity back (the slab is a bump ring; buffers are not freed mid-run).
pub struct ForgeGraphExecutor {
    ctx: WgpuComputeContext,
    ext_views: Vec<BufferView>,
    node_out: Vec<Option<BufferView>>,
    passes: Vec<GraphPass>,
}

impl ForgeGraphExecutor {
    /// Create an executor with the default per-slab capacity (`EXEC_CAPACITY`). Acquires its
    /// **own** GPU adapter + device once; reuse the returned value across calls/decode steps.
    /// To instead run on the process-wide LLM device, use [`Self::on_shared_gpu`].
    pub fn new() -> Result<Self, ForgeError> {
        Self::with_capacity(EXEC_CAPACITY)
    }

    /// Create an executor with an explicit per-slab capacity in bytes, on its own device.
    pub fn with_capacity(capacity_bytes: usize) -> Result<Self, ForgeError> {
        Ok(Self::with_context(WgpuComputeContext::new(capacity_bytes)?))
    }

    /// Wrap an executor around a **caller-supplied** context — e.g. one built on the process-wide
    /// [`etalon_resources::device_caps::shared_gpu`] device via [`WgpuComputeContext::from_device`]. This is
    /// how the forge is made to run on the SAME device that owns the resident LLM weights / KV
    /// cache instead of spinning up a second device (LLM-on-forge plan, Phase 1a).
    pub fn with_context(ctx: WgpuComputeContext) -> Self {
        Self {
            ctx,
            ext_views: Vec::new(),
            node_out: Vec::new(),
            passes: Vec::new(),
        }
    }

    /// Build an executor on the process-wide shared GPU device ([`etalon_resources::device_caps::shared_gpu`]),
    /// at the default per-slab capacity, so the forge shares the device that owns LLM weights +
    /// KV cache. Native only (`shared_gpu` is host-side). The keystone entry point for decode-on-forge.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn on_shared_gpu() -> Result<Self, ForgeError> {
        Self::on_shared_gpu_with_capacity(EXEC_CAPACITY)
    }

    /// [`Self::on_shared_gpu`] with an explicit per-slab capacity in bytes.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn on_shared_gpu_with_capacity(capacity_bytes: usize) -> Result<Self, ForgeError> {
        let (device, queue, caps) =
            crate::numerical::device_options::shared_device().ok_or_else(|| {
                ForgeError::GpuUnavailable("no application shared-device provider installed".into())
            })?;
        let ctx = WgpuComputeContext::from_device(device, queue, &caps, capacity_bytes)?;
        Ok(Self::with_context(ctx))
    }

    /// Borrow the underlying context (e.g. to read adapter identity / profile).
    pub fn context(&self) -> &WgpuComputeContext {
        &self.ctx
    }

    /// Execute `graph` on the GPU with `externals` as the graph inputs, returning the final
    /// output tensor. Intermediates are kept device-side; every node's dispatch and its GPU→GPU
    /// hand-off copy are recorded into ONE command encoder and submitted once
    /// ([`WgpuComputeContext::submit_graph`], Option B). Re-uses the device/slab across calls.
    /// **Every** external is (re)uploaded into the transient ring this call; for the decode loop
    /// where the big matrices are constant across tokens, use [`Self::load_weights`] +
    /// [`Self::run_resident`] instead so they upload once.
    pub fn run(
        &mut self,
        graph: &ComputeGraph,
        externals: &[Vec<f32>],
    ) -> Result<Vec<f32>, ForgeError> {
        // Per-call slab reset: free the previous call's transient allocations so this call starts
        // with the full capacity. Safe — the previous run's readback fully synchronized the device.
        self.ctx.clear_transient_allocations();

        // Upload every external into the transient READ slab (binding overwritten per consumer).
        self.ext_views.clear();
        self.ext_views.reserve(externals.len());
        for data in externals {
            let view = self.ctx.allocate_and_write(
                bytemuck::cast_slice(data),
                0,
                0,
                BindingUsage::StorageRead,
            )?;
            self.ext_views.push(view);
        }
        self.run_prepared(graph)
    }

    /// Caller-buffered variant of [`Self::run`]: accepts borrowed `&[&[f32]]` external input
    /// slices and writes the output tensor directly into `out`, returning the number of `f32`
    /// elements written. Fails closed with [`ForgeError::GpuValidation`] if `out` is smaller than
    /// the graph output tensor.
    pub fn run_into(
        &mut self,
        graph: &ComputeGraph,
        externals: &[&[f32]],
        out: &mut [f32],
    ) -> Result<usize, ForgeError> {
        self.ctx.clear_transient_allocations();
        self.ext_views.clear();
        self.ext_views.reserve(externals.len());
        for &data in externals {
            let view = self.ctx.allocate_and_write(
                bytemuck::cast_slice(data),
                0,
                0,
                BindingUsage::StorageRead,
            )?;
            self.ext_views.push(view);
        }
        self.run_prepared_into(graph, out)
    }

    /// Like [`Self::run`], but external indices already uploaded into the persistent weight region
    /// (via [`Self::load_weights`]) are bound to their **resident** on-device buffers instead of
    /// being re-uploaded — only the *activation* externals (those not in `resident`) are written
    /// this call. This is the decode-step usage: the big projection / FFN matrices live on-device
    /// across every token; each token uploads just `x` (+ tiny scalars). `externals[i]` is ignored
    /// for any `i` that is resident (pass an empty `vec![]` there for clarity).
    pub fn run_resident(
        &mut self,
        graph: &ComputeGraph,
        externals: &[Vec<f32>],
        resident: &ResidentWeights,
    ) -> Result<Vec<f32>, ForgeError> {
        self.ctx.clear_transient_allocations();
        self.ext_views.clear();
        self.ext_views.reserve(externals.len());
        for (i, data) in externals.iter().enumerate() {
            let view = if let Some(view) = resident.view_for(i) {
                view
            } else {
                self.ctx.allocate_and_write(
                    bytemuck::cast_slice(data),
                    0,
                    0,
                    BindingUsage::StorageRead,
                )?
            };
            self.ext_views.push(view);
        }
        self.run_prepared(graph)
    }

    /// Caller-buffered variant of [`Self::run_resident`]: accepts borrowed `&[&[f32]]` activation
    /// slices (with `&[]` placeholders for resident weight indices) and writes the output tensor
    /// into caller-owned `out`, returning the number of `f32` elements written. Designed for hot
    /// decode loops so callers never allocate `Vec<Vec<f32>>` inputs or `Vec<f32>` outputs per token.
    pub fn run_resident_into(
        &mut self,
        graph: &ComputeGraph,
        externals: &[&[f32]],
        resident: &ResidentWeights,
        out: &mut [f32],
    ) -> Result<usize, ForgeError> {
        self.ctx.clear_transient_allocations();
        self.ext_views.clear();
        self.ext_views.reserve(externals.len());
        for (i, &data) in externals.iter().enumerate() {
            let view = if let Some(view) = resident.view_for(i) {
                view
            } else {
                self.ctx.allocate_and_write(
                    bytemuck::cast_slice(data),
                    0,
                    0,
                    BindingUsage::StorageRead,
                )?
            };
            self.ext_views.push(view);
        }
        self.run_prepared_into(graph, out)
    }

    /// Upload a set of `(external_index, data)` weights ONCE into the executor's persistent weight
    /// region and return a [`ResidentWeights`] handle mapping those indices to their on-device
    /// views. Additive — successive calls accumulate (use [`Self::clear_weights`] to start over).
    /// Pass the handle to [`Self::run_resident`] so those externals are referenced by offset, not
    /// re-uploaded per token (the LLM-on-forge weight-residency lever).
    pub fn load_weights(
        &mut self,
        weights: &[(usize, Vec<f32>)],
    ) -> Result<ResidentWeights, ForgeError> {
        let mut views = std::collections::HashMap::with_capacity(weights.len());
        for (idx, data) in weights {
            let view = self.ctx.allocate_weight(bytemuck::cast_slice(data), 0, 0)?;
            views.insert(*idx, view);
        }
        Ok(ResidentWeights { views })
    }

    /// Drop all resident weights, freeing the persistent region for a new model/layer set. Any
    /// [`ResidentWeights`] handles from before this call are stale and must not be reused.
    pub fn clear_weights(&mut self) {
        self.ctx.clear_weights();
    }

    /// The shared execution core: given `self.ext_views` already resolved to device [`BufferView`]s
    /// (freshly uploaded and/or resident), prepare every node, record the whole graph into one
    /// encoder, submit once, and read back the output.
    fn run_prepared(&mut self, graph: &ComputeGraph) -> Result<Vec<f32>, ForgeError> {
        let out_view = self.prepare_and_resolve_output(graph)?;
        self.ctx.submit_graph(&self.passes)?;
        self.ctx.read_buffer_f32(&out_view)
    }

    /// Shared execution core for caller-buffered [`Self::run_into`] and [`Self::run_resident_into`].
    /// Validates `out` capacity before submitting GPU work (fail-closed), then copies the final
    /// output tensor into `out[..elems]`.
    fn run_prepared_into(
        &mut self,
        graph: &ComputeGraph,
        out: &mut [f32],
    ) -> Result<usize, ForgeError> {
        let out_view = self.prepare_and_resolve_output(graph)?;
        let elems = out_view.length_bytes / core::mem::size_of::<f32>();
        if out.len() < elems {
            return Err(ForgeError::GpuValidation(format!(
                "caller output buffer too small: graph produces {elems} f32s, buffer has {}",
                out.len()
            )));
        }
        self.ctx.submit_graph(&self.passes)?;
        self.ctx.read_buffer_f32_into(&out_view, out)
    }

    /// Prepare all nodes in topological order using reusable scratch storage and a stack-allocated
    /// `[BufferView; MAX_IN]` input array per node, returning the final output [`BufferView`].
    fn prepare_and_resolve_output(
        &mut self,
        graph: &ComputeGraph,
    ) -> Result<BufferView, ForgeError> {
        self.passes.clear();
        self.passes.reserve(graph.nodes.len());
        self.node_out.clear();
        self.node_out.resize(graph.nodes.len(), None);

        let dummy_view = BufferView {
            offset: 0,
            length_bytes: 0,
            binding: 0,
            group: 0,
            usage: BindingUsage::StorageRead,
        };

        for (i, node) in graph.nodes.iter().enumerate() {
            let n_in = node.n_in as usize;
            let mut ins = [dummy_view; MAX_IN];
            for (slot, maybe_tr) in ins.iter_mut().zip(node.ins.iter()).take(n_in) {
                let tr = maybe_tr.expect("declared input present");
                *slot = if tr.producer == NodeId::EXTERNAL {
                    *self
                        .ext_views
                        .get(tr.tensor.0 as usize)
                        .ok_or_else(|| ForgeError::Emission("missing external input".to_string()))?
                } else {
                    self.node_out[tr.producer.0 as usize]
                        .ok_or_else(|| ForgeError::Emission("input from unrun node".to_string()))?
                };
            }
            let (pass, out_view) = prepare_node(&mut self.ctx, node, &ins[..n_in])?;
            self.node_out[i] = Some(out_view);
            self.passes.push(pass);
        }

        let id = graph
            .outputs
            .last()
            .copied()
            .unwrap_or(NodeId(graph.nodes.len().saturating_sub(1) as u32));
        self.node_out
            .get(id.0 as usize)
            .and_then(|v| *v)
            .ok_or_else(|| ForgeError::Emission("graph has no output node".to_string()))
    }
}

/// A handle to weights uploaded once into a [`ForgeGraphExecutor`]'s persistent weight region by
/// [`ForgeGraphExecutor::load_weights`]. Maps a graph's **external input index** to its resident
/// on-device [`BufferView`]; passed to [`ForgeGraphExecutor::run_resident`] so those inputs are
/// referenced by offset across many runs instead of re-uploaded per call.
#[derive(Debug, Clone, Default)]
pub struct ResidentWeights {
    views: std::collections::HashMap<usize, BufferView>,
}

impl ResidentWeights {
    /// The resident device view for external index `i`, if it was loaded.
    pub fn view_for(&self, i: usize) -> Option<BufferView> {
        self.views.get(&i).copied()
    }
    /// Number of resident weight tensors held.
    pub fn len(&self) -> usize {
        self.views.len()
    }
    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}
