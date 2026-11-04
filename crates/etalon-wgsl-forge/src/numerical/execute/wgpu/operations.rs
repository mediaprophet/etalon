use super::*;

impl WgpuComputeContext {
    pub(super) fn submit_commands(
        &self,
        command_buffer: wgpu::CommandBuffer,
    ) -> Result<(), ForgeError> {
        self.ensure_live()?;
        etalon_resources::submission::submit_checked(&self.queue, [command_buffer]).map_err(
            |_| ForgeError::GpuValidation("native queue rejected the command submission".into()),
        )?;
        Ok(())
    }

    /// The physical buffer backing a view, chosen by its binding usage.
    pub(super) fn slab_for(&self, usage: BindingUsage) -> &wgpu::Buffer {
        match usage {
            BindingUsage::StorageReadWrite => &self.out_slab,
            BindingUsage::StorageReadResident => &self.weight_slab,
            BindingUsage::StorageRead | BindingUsage::Uniform => &self.slab,
        }
    }

    /// Allocate a transient slab sub-range and upload `data` into it.
    ///
    /// # Topology note (honest scope, plan §2)
    ///
    /// This upload uses `queue.write_buffer` **uniformly on every topology**
    /// (unified and discrete alike); readback in [`Self::read_buffer_f32`]
    /// likewise uses `copy_buffer_to_buffer` uniformly. The
    /// `MemoryTopology::{Unified, Discrete}` classification on the allocator is
    /// *recorded but not yet acted upon here*: the plan-§2 differentiated paths
    /// (zero-copy persistent-mapped slabs for unified memory; a pinned staging ring
    /// with async `copy_buffer` for discrete PCIe) are NOT implemented. The current
    /// uniform path is correct on both topologies but unoptimised; the unified
    /// zero-copy benefit cannot be measured on this discrete-only host (RTX A2000),
    /// so it is left as documented future work rather than shipped unverified. See
    /// [`MemoryTopology`] for the full rationale.
    pub fn allocate_and_write(
        &mut self,
        data: &[u8],
        binding: u32,
        group: u32,
        usage: BindingUsage,
    ) -> Result<BufferView, ForgeError> {
        self.ensure_live()?;
        let view = self
            .allocator
            .allocate_transient(data.len(), binding, group, usage)?;
        if !data.is_empty() {
            let slab = self.slab_for(usage);
            self.queue
                .write_buffer(slab, view.offset as wgpu::BufferAddress, data);
        }
        Ok(view)
    }

    /// Bump-allocate `data` into the **persistent weight region** (`weight_slab`) and upload it
    /// once, returning a [`BufferView`] tagged [`BindingUsage::StorageReadResident`]. Unlike
    /// [`Self::allocate_and_write`] (transient ring), this view **survives**
    /// [`Self::clear_transient_allocations`], so a decode layer's projection / FFN matrices are
    /// uploaded a single time and referenced by offset across every token's `run` — eliminating
    /// the per-call weight re-upload. Offsets are 256-aligned for direct bind-group use.
    pub fn allocate_weight(
        &mut self,
        data: &[u8],
        binding: u32,
        group: u32,
    ) -> Result<BufferView, ForgeError> {
        self.ensure_live()?;
        let offset =
            self.weight_cursor.div_ceil(DEFAULT_BINDING_ALIGNMENT) * DEFAULT_BINDING_ALIGNMENT;
        let end = offset + data.len();
        let cap = self.weight_slab.size() as usize;
        if end > cap {
            return Err(ForgeError::GpuValidation(format!(
                "weight region overflow: need {end} bytes but weight slab is {cap} (raise capacity)"
            )));
        }
        if !data.is_empty() {
            self.queue
                .write_buffer(&self.weight_slab, offset as wgpu::BufferAddress, data);
        }
        self.weight_cursor = end;
        Ok(BufferView {
            offset,
            length_bytes: data.len(),
            binding,
            group,
            usage: BindingUsage::StorageReadResident,
        })
    }

    /// Reset the persistent weight region so it can be reused for a different model/layer set.
    /// Any [`BufferView`]s previously returned by [`Self::allocate_weight`] become stale — drop
    /// the corresponding handles and re-load. (Weights are write-once; no per-tensor free.)
    pub fn clear_weights(&mut self) {
        self.weight_cursor = 0;
    }

    /// Bytes currently consumed in the persistent weight region (for tests / introspection).
    pub fn resident_weight_bytes(&self) -> usize {
        self.weight_cursor
    }

    /// `WgpuComputeContext::allocate_transient`.
    pub fn allocate_transient(
        &mut self,
        size_bytes: usize,
        binding: u32,
        group: u32,
        usage: BindingUsage,
    ) -> Result<BufferView, ForgeError> {
        self.ensure_live()?;
        self.allocator
            .allocate_transient(size_bytes, binding, group, usage)
    }

    /// Advances the transient read head by `offset`.
    pub fn advance_read_head(&mut self, offset: usize) {
        self.allocator.advance_read_head(offset);
    }

    /// Releases all transient allocations.
    pub fn clear_transient_allocations(&mut self) {
        self.allocator.clear();
    }

    /// Builds a bottom-level (BLAS) + top-level (TLAS) acceleration structure for a
    /// triangle soup and returns both, ready to bind to a ray-query shader. `vertices`
    /// is a flat list of `f32` triples (3 per vertex, 3 vertices per triangle),
    /// row-major. The BLAS geometry is marked `OPAQUE` (required — naga's ray-query
    /// has no candidate/any-hit path, so non-opaque geometry yields no committed hits),
    /// and the single TLAS instance uses the identity transform. Both structures are
    /// built and the queue drained before returning. Requires the adapter to support
    /// (and the device to have enabled) `EXPERIMENTAL_RAY_QUERY`.
    ///
    /// The returned `Blas` must be kept alive alongside the `Tlas` for the lifetime of
    /// any bind group referencing the TLAS (the `TlasInstance` borrows the BLAS).
    pub fn build_triangle_scene(
        &self,
        vertices: &[f32],
    ) -> Result<(wgpu::Blas, wgpu::Tlas), ForgeError> {
        self.ensure_live()?;
        if !self.constraints.supports_rt_cores {
            return Err(ForgeError::GpuUnavailable(
                "adapter lacks ray-query (RT) support".to_string(),
            ));
        }
        if vertices.is_empty() || vertices.len() % 9 != 0 {
            return Err(ForgeError::GpuValidation(format!(
                "triangle scene needs a non-empty multiple of 9 floats (3 verts x xyz); got {}",
                vertices.len()
            )));
        }
        let vertex_count = (vertices.len() / 3) as u32;

        let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forge-blas-vertices"),
            size: (vertices.len() * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(vertices));

        let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count,
            index_format: None,
            index_count: None,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = self.device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("forge-blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size_desc.clone()],
            },
        );

        let mut tlas = self.device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("forge-tlas"),
            max_instances: 1,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        // 3x4 row-major identity (scene vertices are already in world space).
        let identity: [f32; 12] = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        tlas[0] = Some(wgpu::TlasInstance::new(&blas, identity, 0, 0xFF));

        let geometry = wgpu::BlasTriangleGeometry {
            size: &size_desc,
            vertex_buffer: &vertex_buffer,
            first_vertex: 0,
            vertex_stride: (3 * size_of::<f32>()) as wgpu::BufferAddress,
            index_buffer: None,
            first_index: None,
            transform_buffer: None,
            transform_buffer_offset: None,
        };
        let entry = wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![geometry]),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forge-build-accel"),
            });
        encoder.build_acceleration_structures(std::iter::once(&entry), std::iter::once(&tlas));
        self.submit_commands(encoder.finish())?;
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| {
                ForgeError::DeviceLost(format!("device poll failed building accel: {e:?}"))
            })?;

        self.ensure_live()?;
        Ok((blas, tlas))
    }

    /// Compile a WGSL compute pipeline and return the **owned** `wgpu::ComputePipeline`
    /// (no borrow of `self`), wrapped in a validation error scope. This is the building
    /// block the multi-node graph executor uses to compile every node's kernel up front
    /// before recording them into a single command encoder ([`Self::submit_graph`]).
    /// [`WgpuPipeline::compile`] delegates here.
    pub fn compile_pipeline(
        &self,
        source: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline, ForgeError> {
        self.ensure_live()?;
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("qualia-wgsl-forge"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("qualia-wgsl-forge-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(ForgeError::GpuValidation(error.to_string()));
        }
        Ok(pipeline)
    }

    /// Like `compile_pipeline()` but accepts pre-compiled SPIR-V bytes instead of
    /// WGSL source. This is the execution bridge for native shader profiles that
    /// compile to SPIR-V (notably HLSL via DXC `–spirv`): the forge emits HLSL,
    /// DXC produces a SPIR-V binary, and this method feeds it into the same wgpu
    /// pipeline (bind groups, slab, dispatch — all unchanged).
    pub fn compile_pipeline_spirv(
        &self,
        spirv: &[u8],
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline, ForgeError> {
        self.ensure_live()?;
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let spirv_words: &[u32] = bytemuck::cast_slice(spirv);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("qualia-forge-spirv"),
                source: wgpu::ShaderSource::SpirV(std::borrow::Cow::Borrowed(spirv_words)),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("qualia-forge-spirv-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(ForgeError::GpuValidation(error.to_string()));
        }
        Ok(pipeline)
    }

    /// [`compile_pipeline`](Self::compile_pipeline) with a process-lifetime cache keyed by
    /// `entry\0source` — the same `(source, entry)` returns the previously-built pipeline
    /// (a cheap `Arc`-clone) instead of recompiling. This is what makes a re-run of a fixed
    /// graph (e.g. one decode block per generated token, via a held `ForgeGraphExecutor()`)
    /// pay shader compilation **once**, not per call. The graph executor records its nodes
    /// through this path; one-shot callers see a cold cache (built + dropped with the context).
    pub fn compile_pipeline_cached(
        &self,
        source: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline, ForgeError> {
        self.ensure_live()?;
        let key = format!("{entry_point}\u{0}{source}");
        if let Some(pipeline) = self.pipeline_cache.borrow().get(&key) {
            return Ok(pipeline.clone());
        }
        let pipeline = self.compile_pipeline(source, entry_point)?;
        self.pipeline_cache
            .borrow_mut()
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// Number of distinct pipelines currently cached (for tests / introspection).
    pub fn cached_pipeline_count(&self) -> usize {
        self.pipeline_cache.borrow().len()
    }

    /// Build a bind group binding each [`BufferView`] at its `binding` slot, choosing the
    /// physical slab per the view's usage (`Self::slab_for`). Shared by the per-node
    /// [`WgpuPipeline::dispatch`] path and the deferred-submit graph path.
    pub fn create_compute_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[BufferView],
    ) -> wgpu::BindGroup {
        let mut entries = Vec::with_capacity(buffers.len());
        for view in buffers {
            let size = wgpu::BufferSize::new(view.length_bytes as u64);
            entries.push(wgpu::BindGroupEntry {
                binding: view.binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.slab_for(view.usage),
                    offset: view.offset as wgpu::BufferAddress,
                    size,
                }),
            });
        }
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forge-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Record **all** of a graph's node dispatches — and the GPU→GPU hand-off copies
    /// between them — into ONE [`wgpu::CommandEncoder`] and submit it **once**, instead of
    /// one `queue.submit()` per node. This is the single-encoder deferred-submit fusion
    /// (plan §8.1 "Option B"): within one command buffer wgpu preserves command order and
    /// inserts the necessary buffer hazard barriers, so a producer's compute pass, its
    /// `copy_buffer_to_buffer` hand-off, and the consumer's dispatch are correctly ordered
    /// with no host round-trip and no per-node submit latency. The caller (the executor)
    /// has already encoded each node's data dependencies in `passes` (insertion/topological
    /// order) and built each bind group, so this loop is pure recording. Blocks on device
    /// completion and surfaces any validation error.
    pub fn submit_graph(&self, passes: &[GraphPass]) -> Result<(), ForgeError> {
        self.ensure_live()?;
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forge-graph-encoder"),
            });
        for pass in passes {
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forge-graph-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&pass.pipeline);
                cpass.set_bind_group(0, &pass.bind_group, &[]);
                cpass.dispatch_workgroups(pass.workgroups, 1, 1);
            }
            // GPU→GPU hand-off: copy this node's read_write output into the fresh
            // read-slab buffer a downstream node will bind read-only. Recorded in the
            // SAME encoder, so it is ordered after the pass that produced it.
            if let Some((src, dst)) = &pass.copy {
                let len = src.length_bytes.min(dst.length_bytes) as u64;
                if len > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.slab_for(src.usage),
                        src.offset as u64,
                        self.slab_for(dst.usage),
                        dst.offset as u64,
                        len,
                    );
                }
            }
        }
        self.submit_commands(encoder.finish())?;
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| {
                ForgeError::DeviceLost(format!("device poll failed submitting graph: {e:?}"))
            })?;
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(ForgeError::GpuValidation(error.to_string()));
        }
        self.ensure_live()?;
        Ok(())
    }

    /// Copy `src`'s bytes to `dst` on the device (GPU→GPU, no host readback), honouring
    /// each view's slab. Used by the multi-node graph executor to move a node's output out
    /// of the read_write slab into the read slab, so a downstream node can bind it as a
    /// read-only input without aliasing its own read_write output (wgpu forbids the same
    /// buffer being bound read-write and read-only within one dispatch). Submits on the
    /// shared queue, so it is ordered before any later dispatch that reads `dst`.
    pub fn copy_view(&self, src: &BufferView, dst: &BufferView) -> Result<(), ForgeError> {
        self.ensure_live()?;
        let len = src.length_bytes.min(dst.length_bytes) as u64;
        if len == 0 {
            return Ok(());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forge-view-copy"),
            });
        encoder.copy_buffer_to_buffer(
            self.slab_for(src.usage),
            src.offset as u64,
            self.slab_for(dst.usage),
            dst.offset as u64,
            len,
        );
        self.submit_commands(encoder.finish())?;
        Ok(())
    }
}
