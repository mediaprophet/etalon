use super::*;
// ─── GEMM operator ───────────────────────────────────────────────────────────

pub struct DmlGemmOp {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

impl DmlGemmOp {
    /// Execute C = A(m×k) × B(k×n) on the GPU via DirectML.
    pub fn execute(
        &self,
        device: &DmlDevice,
        a_data: &[f32],
        b_data: &[f32],
    ) -> windows::core::Result<Vec<f32>> {
        let (m, k, n) = (self.m as usize, self.k as usize, self.n as usize);
        let out_bytes = (m * n * 4) as u64;

        unsafe {
            let d3d = &device.d3d12;

            // ── Buffers ───────────────────────────────────────────────────────
            let buf_a = upload_buffer(d3d, bytemuck::cast_slice(a_data))?;
            let buf_b = upload_buffer(d3d, bytemuck::cast_slice(b_data))?;
            let buf_c = gpu_buffer(d3d, out_bytes)?;

            // ── DML tensor descriptors ─────────────────────────────────────────
            let sizes_a = [m as u32, k as u32];
            let sizes_b = [k as u32, n as u32];
            let sizes_c = [m as u32, n as u32];
            // Descriptor backing lives through CreateOperator without leaking heap boxes.
            let inner_a = make_buffer_tensor_desc(&sizes_a, (m * k * 4) as u64);
            let inner_b = make_buffer_tensor_desc(&sizes_b, (k * n * 4) as u64);
            let inner_c = make_buffer_tensor_desc(&sizes_c, out_bytes);
            let td_a = make_tensor_desc(&inner_a);
            let td_b = make_tensor_desc(&inner_b);
            let td_c = make_tensor_desc(&inner_c);

            let gemm_inner = DML_GEMM_OPERATOR_DESC {
                ATensor: &td_a,
                BTensor: &td_b,
                CTensor: std::ptr::null(),
                OutputTensor: &td_c,
                TransA: DML_MATRIX_TRANSFORM_NONE,
                TransB: DML_MATRIX_TRANSFORM_NONE,
                Alpha: 1.0,
                Beta: 0.0,
                FusedActivation: std::ptr::null(),
            };
            let op_desc = DML_OPERATOR_DESC {
                Type: DML_OPERATOR_GEMM,
                Desc: &gemm_inner as *const _ as *const _,
            };

            let mut op_opt: Option<IDMLOperator> = None;
            device.dml.CreateOperator(&op_desc, &mut op_opt)?;
            let op = op_opt.unwrap();

            let mut compiled_opt: Option<IDMLCompiledOperator> = None;
            device
                .dml
                .CompileOperator(&op, DML_EXECUTION_FLAGS(0), &mut compiled_opt)?;
            let compiled = compiled_opt.unwrap();

            // ── Descriptor heap ───────────────────────────────────────────────
            let bind_props = compiled.GetBindingProperties();
            let desc_count = bind_props.RequiredDescriptorCount.max(1);
            let heap: ID3D12DescriptorHeap =
                d3d.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: desc_count,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let cpu_h = heap.GetCPUDescriptorHandleForHeapStart();
            let gpu_h = heap.GetGPUDescriptorHandleForHeapStart();

            // ── Initialise ────────────────────────────────────────────────────
            let init_op: IDMLOperatorInitializer = device
                .dml
                .CreateOperatorInitializer(Some(&[Some(compiled.clone())]))?;

            let init_dispatchable: IDMLDispatchable = init_op.cast()?;
            let init_td = DML_BINDING_TABLE_DESC {
                Dispatchable: ManuallyDrop::new(Some(init_dispatchable)),
                CPUDescriptorHandle: cpu_h,
                GPUDescriptorHandle: gpu_h,
                SizeInDescriptors: desc_count,
            };
            let init_table: IDMLBindingTable =
                device.dml.CreateBindingTable(Some(&init_td as *const _))?;

            let recorder: IDMLCommandRecorder = device.dml.CreateCommandRecorder()?;
            let alloc: ID3D12CommandAllocator =
                d3d.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let cmd: ID3D12GraphicsCommandList =
                d3d.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            cmd.SetDescriptorHeaps(&[Some(heap.clone())]);
            recorder.RecordDispatch(&cmd, &init_op, &init_table);
            submit_and_wait(d3d, &device.queue, &alloc, &cmd, 1)?;

            // ── Execute ───────────────────────────────────────────────────────
            let exec_dispatchable: IDMLDispatchable = compiled.cast()?;
            let exec_td = DML_BINDING_TABLE_DESC {
                Dispatchable: ManuallyDrop::new(Some(exec_dispatchable)),
                CPUDescriptorHandle: cpu_h,
                GPUDescriptorHandle: gpu_h,
                SizeInDescriptors: desc_count,
            };
            let exec_table: IDMLBindingTable =
                device.dml.CreateBindingTable(Some(&exec_td as *const _))?;

            let bb_a = DML_BUFFER_BINDING {
                Buffer: ManuallyDrop::new(Some(buf_a.clone())),
                Offset: 0,
                SizeInBytes: (m * k * 4) as u64,
            };
            let bb_b = DML_BUFFER_BINDING {
                Buffer: ManuallyDrop::new(Some(buf_b.clone())),
                Offset: 0,
                SizeInBytes: (k * n * 4) as u64,
            };
            let bb_c = DML_BUFFER_BINDING {
                Buffer: ManuallyDrop::new(Some(buf_c.clone())),
                Offset: 0,
                SizeInBytes: out_bytes,
            };

            let in_binds = [
                DML_BINDING_DESC {
                    Type: DML_BINDING_TYPE_BUFFER,
                    Desc: &bb_a as *const _ as _,
                },
                DML_BINDING_DESC {
                    Type: DML_BINDING_TYPE_BUFFER,
                    Desc: &bb_b as *const _ as _,
                },
            ];
            let out_bind = DML_BINDING_DESC {
                Type: DML_BINDING_TYPE_BUFFER,
                Desc: &bb_c as *const _ as _,
            };

            exec_table.BindInputs(Some(&in_binds));
            exec_table.BindOutputs(Some(std::slice::from_ref(&out_bind)));

            let cmd2: ID3D12GraphicsCommandList =
                d3d.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            cmd2.SetDescriptorHeaps(&[Some(heap.clone())]);
            recorder.RecordDispatch(&cmd2, &compiled, &exec_table);
            submit_and_wait(d3d, &device.queue, &alloc, &cmd2, 2)?;

            // ── Readback ──────────────────────────────────────────────────────
            let staging = readback_buffer(d3d, out_bytes)?;
            let cmd3: ID3D12GraphicsCommandList =
                d3d.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            cmd3.CopyResource(&staging, &buf_c);
            submit_and_wait(d3d, &device.queue, &alloc, &cmd3, 3)?;

            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            staging.Map(0, None, Some(&mut ptr))?;
            let result = std::slice::from_raw_parts(ptr as *const f32, m * n).to_vec();
            staging.Unmap(0, None);
            crate::backends::observation::record(8);
            Ok(result)
        }
    }
}

/// Backing descriptor borrows caller-owned dimensions; both remain alive during CreateOperator.
fn make_buffer_tensor_desc(sizes: &[u32; 2], byte_size: u64) -> DML_BUFFER_TENSOR_DESC {
    DML_BUFFER_TENSOR_DESC {
        DataType: DML_TENSOR_DATA_TYPE_FLOAT32,
        Flags: DML_TENSOR_FLAG_NONE,
        DimensionCount: 2,
        Sizes: sizes.as_ptr(),
        Strides: std::ptr::null(),
        TotalTensorSizeInBytes: byte_size,
        GuaranteedBaseOffsetAlignment: 0,
    }
}
fn make_tensor_desc(inner: &DML_BUFFER_TENSOR_DESC) -> DML_TENSOR_DESC {
    DML_TENSOR_DESC {
        Type: DML_TENSOR_TYPE_BUFFER,
        Desc: inner as *const _ as *const _,
    }
}
