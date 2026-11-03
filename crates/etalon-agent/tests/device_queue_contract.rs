//! Native hardware contract: identities plus compute and drawn RGBA readback.
#![cfg(not(target_family = "wasm"))]
use etalon_agent::{
    ComputeLane, DeviceQueueError, HalComputeLane, HalPreparedPlan, HalSubmission, QueueId,
    SharedDeviceQueue,
};
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

fn wait<F: Future>(future: F) -> F::Output {
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    let start = Instant::now();
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "native GPU request deadline"
        );
        std::thread::park_timeout(Duration::from_millis(10));
    }
}
fn mapped(device: &etalon::Device, buffer: &etalon::Buffer) -> Vec<u8> {
    let (send, recv) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(etalon::MapMode::Read, move |result| {
            send.send(result).unwrap();
        });
    device
        .poll(etalon::PollType::wait_or_timeout(Duration::from_secs(30)))
        .unwrap();
    recv.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap().to_vec();
    buffer.unmap();
    data
}
fn buffer(device: &etalon::Device, size: u64, usage: etalon::BufferUsages) -> etalon::Buffer {
    device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
    })
}
#[test]
#[ignore = "requires explicitly selected physical GPU; run with --ignored"]
fn actual_device_identity_and_shared_compute_render_readbacks() {
    let mut desc = etalon::InstanceDescriptor::new_without_display_handle();
    let requested = std::env::var("ETALON_PAIR_TEST_BACKEND").unwrap_or_else(|_| {
        if cfg!(windows) {
            "dx12".into()
        } else {
            "vulkan".into()
        }
    });
    desc.backends = match requested.as_str() {
        "dx12" => etalon::Backends::DX12,
        "vulkan" => etalon::Backends::VULKAN,
        "gl" => etalon::Backends::GL,
        "metal" => etalon::Backends::METAL,
        _ => panic!("unsupported explicit test backend"),
    };
    let instance = etalon::Instance::new(desc);
    let adapter = wait(instance.request_adapter(&etalon::RequestAdapterOptions {
        power_preference: etalon::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("requested native adapter unavailable; this is not a skip/pass");
    let info = adapter.get_info();
    let expected_backend = match requested.as_str() {
        "dx12" => etalon::Backend::Dx12,
        "vulkan" => etalon::Backend::Vulkan,
        "gl" => etalon::Backend::Gl,
        "metal" => etalon::Backend::Metal,
        _ => unreachable!(),
    };
    assert_eq!(
        info.backend, expected_backend,
        "requested backend was substituted"
    );
    assert_ne!(
        info.device_type,
        etalon::DeviceType::Cpu,
        "CPU substitution is not hardware qualification"
    );
    println!("device_queue_contract requested={requested} adapter={info:?}");
    let desc = etalon::DeviceDescriptor {
        label: Some("pair contract"),
        required_features: etalon::Features::empty(),
        required_limits: etalon::Limits::downlevel_defaults(),
        default_queue: etalon::QueueDescriptor { label: None },
        experimental_features: etalon::ExperimentalFeatures::disabled(),
        memory_hints: etalon::MemoryHints::MemoryUsage,
        trace: etalon::Trace::Off,
    };
    let (device, queue) = wait(adapter.request_device(&desc)).unwrap();
    let (other_device, other_queue) = wait(adapter.request_device(&desc)).unwrap();
    assert_eq!(device.owns_queue(&queue), Some(true));
    assert_eq!(device.clone().owns_queue(&queue.clone()), Some(true));
    assert_eq!(device.owns_queue(&other_queue), Some(false));
    assert_eq!(other_device.owns_queue(&queue), Some(false));
    assert_eq!(
        SharedDeviceQueue::try_new(Arc::new(device.clone()), Arc::new(other_queue)).unwrap_err(),
        DeviceQueueError::Mismatch
    );
    assert_eq!(
        SharedDeviceQueue::try_new(Arc::new(other_device), Arc::new(queue.clone())).unwrap_err(),
        DeviceQueueError::Mismatch
    );
    let pair =
        SharedDeviceQueue::try_new(Arc::new(device.clone()), Arc::new(queue.clone())).unwrap();
    let clone = pair.clone();
    assert!(Arc::ptr_eq(pair.device(), clone.device()));
    assert!(Arc::ptr_eq(pair.queue(), clone.queue()));
    drop(device);
    drop(queue);
    drop(pair);
    compute(&clone);
    render(&clone);
    println!("device_queue_contract compute_words=4 exact=true rendered_pixels=16 exact=true same_adapter_distinct_device_rejections=2");
}
fn compute(pair: &SharedDeviceQueue) {
    let device = pair.device();
    let module=device.create_shader_module(etalon::ShaderModuleDescriptor{label:None,source:etalon::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> result_words: array<u32>; @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id: vec3<u32>) { result_words[id.x] = 13u + id.x * 17u; }".into())});
    let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let storage = buffer(
        device,
        16,
        etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_SRC,
    );
    let download = buffer(
        device,
        16,
        etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
    );
    let bind = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });
    let mut lane = HalComputeLane::from_shared_pair(pair.clone(), QueueId(77));
    lane.prepare(&HalPreparedPlan {
        pipeline: Arc::new(pipeline),
        default_workgroups: [1, 1, 1],
    })
    .unwrap();
    let token = lane
        .submit(HalSubmission {
            bind_groups: vec![Arc::new(bind)],
            workgroups_override: None,
        })
        .unwrap();
    assert_eq!(token.queue, QueueId(77));
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&storage, 0, &download, 0, 16);
    assert!(pair.queue().submit([encoder.finish()]).is_some());
    let actual = mapped(device, &download);
    let expected: [u32; 4] = [13, 30, 47, 64];
    let expected: Vec<u8> = expected.into_iter().flat_map(u32::to_le_bytes).collect();
    assert_eq!(actual, expected);
    assert!(lane.poll_completion(token));
    lane.verify_submission(token, &expected, &actual).unwrap();
    let mut wrong = actual.clone();
    wrong[0] ^= 1;
    assert!(lane.verify_submission(token, &expected, &wrong).is_err());
}
fn render(pair: &SharedDeviceQueue) {
    let device = pair.device();
    let size = etalon::Extent3d {
        width: 4,
        height: 4,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let module = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: None,
        source: etalon::ShaderSource::Wgsl(
            r#"
@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{
 let p=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));return vec4(p[i],0.0,1.0);
}
@fragment fn fragment()->@location(0) vec4<f32>{return vec4(1.0,0.0,1.0,1.0);}
"#
            .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: etalon::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(etalon::FragmentState {
            module: &module,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(etalon::ColorTargetState {
                format: etalon::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: etalon::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let download = buffer(
        device,
        1024,
        etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        etalon::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        etalon::TexelCopyBufferInfo {
            buffer: &download,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        size,
    );
    assert!(pair.queue().submit([encoder.finish()]).is_some());
    let actual = mapped(device, &download);
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(
                &actual[y * 256 + x * 4..y * 256 + x * 4 + 4],
                &[255, 0, 255, 255]
            );
        }
    }
}
