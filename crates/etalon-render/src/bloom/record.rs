use super::*;
pub fn run_bloom_passes<R>(
    encoder: &mut etalon::CommandEncoder,
    bloom: &BloomChain<R>,
    queue: &etalon::Queue,
    device: &etalon::Device,
    surface_view: &etalon::TextureView,
    clear_color: [f64; 4],
) {
    let (blur_w, blur_h) = bloom.blur_extent();
    let _ = bloom.texture_handles();
    debug_assert_eq!(blur_w, bloom.blur_a.width());
    debug_assert_eq!(blur_h, bloom.blur_b.height());

    write_bloom_uniform(
        queue,
        &bloom.uniform_buf,
        1.0,
        bloom.exposure_scale,
        bloom.surface_is_srgb,
        bloom.white_balance_gains,
    );
    let extract_bind = make_bloom_bind_group(
        device,
        &bloom.bind_layout,
        &bloom.sampler,
        &bloom.hdr_view,
        &bloom.dummy_view,
        &bloom.uniform_buf,
    );
    {
        let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("portal-bloom-extract"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &bloom.blur_a_view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&bloom.extract_pipeline);
        pass.set_bind_group(0, &extract_bind, &[]);
        pass.draw(0..3, 0..1);
    }

    let mut read = &bloom.blur_a_view;
    let mut write = &bloom.blur_b_view;
    for &offset in &KAWASE_OFFSETS {
        write_bloom_uniform(
            queue,
            &bloom.uniform_buf,
            offset,
            bloom.exposure_scale,
            bloom.surface_is_srgb,
            bloom.white_balance_gains,
        );
        let kawase_bind = make_bloom_bind_group(
            device,
            &bloom.bind_layout,
            &bloom.sampler,
            read,
            &bloom.dummy_view,
            &bloom.uniform_buf,
        );
        {
            let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: Some("portal-bloom-kawase"),
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: write,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&bloom.kawase_pipeline);
            pass.set_bind_group(0, &kawase_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        std::mem::swap(&mut read, &mut write);
    }
    let bloom_result = read;

    write_bloom_uniform(
        queue,
        &bloom.uniform_buf,
        1.0,
        bloom.exposure_scale,
        bloom.surface_is_srgb,
        bloom.white_balance_gains,
    );
    let composite_bind = make_bloom_bind_group(
        device,
        &bloom.bind_layout,
        &bloom.sampler,
        &bloom.hdr_view,
        bloom_result,
        &bloom.uniform_buf,
    );
    {
        let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("portal-bloom-composite"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: surface_view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color {
                        r: clear_color[0],
                        g: clear_color[1],
                        b: clear_color[2],
                        a: clear_color[3],
                    }),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&bloom.composite_pipeline);
        pass.set_bind_group(0, &composite_bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
