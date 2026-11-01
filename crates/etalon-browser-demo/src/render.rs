use serde_json::Value;

use crate::{browser::failure, readback};

const SOURCE: &str = r#"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var points = array<vec2<f32>, 3>(vec2(-0.75, -0.75), vec2(0.75, -0.75), vec2(0.0, 0.75));
    return vec4(points[i], 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4(1.0, 0.0, 0.0, 1.0); }
"#;

pub(crate) async fn run(device: &etalon::Device, queue: &etalon::Queue) -> Result<Vec<u8>, Value> {
    run_backend(device, queue, false).await
}

pub(crate) async fn run_webgl(
    device: &etalon::Device,
    queue: &etalon::Queue,
) -> Result<Vec<u8>, Value> {
    run_backend(device, queue, true).await
}

async fn run_backend(
    device: &etalon::Device,
    queue: &etalon::Queue,
    webgl: bool,
) -> Result<Vec<u8>, Value> {
    let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: Some("bounded offscreen triangle"),
        source: etalon::ShaderSource::Wgsl(SOURCE.into()),
    });
    let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: etalon::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(etalon::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(etalon::TextureFormat::Rgba8Unorm.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let size = etalon::Extent3d {
        width: 64,
        height: 64,
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
    let staging = readback::buffer(
        device,
        16384,
        etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
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
                    load: etalon::LoadOp::Clear(etalon::Color::BLUE),
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
            buffer: &staging,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(64),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    let mapped = if webgl {
        crate::webgl_readback::read(device, &staging).await
    } else {
        readback::read(&staging).await
    };
    staging.destroy();
    texture.destroy();
    let pixels = mapped?;
    let red = [255, 0, 0, 255];
    let blue = [0, 0, 255, 255];
    if pixels.len() != 16384 || pixels.chunks_exact(4).any(|p| p != red && p != blue) {
        return Err(failure(
            "render-pixel-values",
            "Expected only opaque red and blue pixels",
        ));
    }
    for (x, y, expected) in [
        (32, 32, red),
        (32, 20, red),
        (20, 48, red),
        (0, 0, blue),
        (63, 63, blue),
        (8, 8, blue),
    ] {
        let offset = (y * 64 + x) * 4;
        if pixels[offset..offset + 4] != expected {
            return Err(failure(
                "render-sample-mismatch",
                &format!("Unexpected pixel at ({x}, {y})"),
            ));
        }
    }
    Ok(pixels)
}
