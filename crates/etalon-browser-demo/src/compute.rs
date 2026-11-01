use serde_json::{json, Value};

use crate::{browser::failure, readback};

const SOURCE: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&input)) { output[id.x] = input[id.x] * 3u + 7u; }
}
"#;

struct Kernel {
    pipeline: etalon::ComputePipeline,
    group: etalon::BindGroup,
    input: etalon::Buffer,
    output: etalon::Buffer,
    staging: etalon::Buffer,
    count: u32,
}

impl Kernel {
    fn new(device: &etalon::Device, count: u32) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("vector oracle shader"),
            source: etalon::ShaderSource::Wgsl(SOURCE.into()),
        });
        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let size = u64::from(count) * 4;
        let input = readback::buffer(
            device,
            size,
            etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
        );
        let output = readback::buffer(
            device,
            size,
            etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_SRC,
        );
        let staging = readback::buffer(
            device,
            size,
            etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        );
        let group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        Self {
            pipeline,
            group,
            input,
            output,
            staging,
            count,
        }
    }

    async fn check(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        seed: u32,
    ) -> Result<(), Value> {
        let values: Vec<u32> = (0..self.count).map(|i| i * 17 + seed).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        queue.write_buffer(&self.input, 0, &bytes);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, u64::from(self.count) * 4);
        queue.submit([encoder.finish()]);
        let actual = readback::read(&self.staging).await?;
        if actual.len() != values.len() * 4 {
            return Err(failure("compute-length", "Unexpected readback length"));
        }
        for (i, bytes) in actual.chunks_exact(4).enumerate() {
            let got = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let expected = values[i] * 3 + 7;
            if got != expected {
                return Err(failure(
                    "compute-mismatch",
                    &format!("Element {i}: expected {expected}, got {got}"),
                ));
            }
        }
        Ok(())
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        self.input.destroy();
        self.output.destroy();
        self.staging.destroy();
    }
}

pub(crate) async fn run(
    device: &etalon::Device,
    queue: &etalon::Queue,
    count: u32,
) -> Result<Value, Value> {
    if ![1, 63, 64, 65, 257].contains(&count) {
        return Err(failure("invalid-size", "Size outside bounded case catalog"));
    }
    let kernel = Kernel::new(device, count);
    kernel.check(device, queue, 5).await?;
    Ok(
        json!({"elements": count, "workgroup_size": 64, "cpu_oracle": "u32: input * 3 + 7", "mismatches": 0}),
    )
}

pub(crate) async fn changed_inputs(
    device: &etalon::Device,
    queue: &etalon::Queue,
) -> Result<Value, Value> {
    let kernel = Kernel::new(device, 257);
    kernel.check(device, queue, 5).await?;
    kernel.check(device, queue, 123).await?;
    Ok(
        json!({"elements_per_run": 257, "runs": 2, "reused_pipeline_and_buffers": true, "input_seeds": [5, 123], "mismatches": 0}),
    )
}
