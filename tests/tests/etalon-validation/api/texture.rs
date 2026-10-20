//! Tests of [`etalon::Texture`] and related.

use std::sync::Arc;

use etalon_test::{fail, valid};

/// Ensures that submitting a command buffer referencing an already destroyed texture
/// results in an error.
#[test]
#[should_panic = "Texture with 'dst' label has been destroyed"]
fn destroyed_texture() {
    let (device, queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let texture_src = device.create_texture(&etalon::TextureDescriptor {
        label: Some("src"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_dst = device.create_texture(&etalon::TextureDescriptor {
        label: Some("dst"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_texture(
        etalon::TexelCopyTextureInfo {
            texture: &texture_src,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        etalon::TexelCopyTextureInfo {
            texture: &texture_dst,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        size,
    );

    texture_dst.destroy();

    queue.submit([encoder.finish()]);
}

/// Ensures that creating a texture view from a specific plane of a planar
/// texture works as expected.
#[test]
fn planar_texture_view_plane() {
    let required_features = etalon::Features::TEXTURE_FORMAT_NV12
        | etalon::Features::TEXTURE_FORMAT_P010
        | etalon::Features::TEXTURE_FORMAT_16BIT_NORM;
    let device_desc = etalon::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = etalon::Device::noop(&device_desc);
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    for (tex_format, view_format, view_aspect) in [
        (
            etalon::TextureFormat::NV12,
            etalon::TextureFormat::R8Unorm,
            etalon::TextureAspect::Plane0,
        ),
        (
            etalon::TextureFormat::NV12,
            etalon::TextureFormat::Rg8Unorm,
            etalon::TextureAspect::Plane1,
        ),
        (
            etalon::TextureFormat::P010,
            etalon::TextureFormat::R16Unorm,
            etalon::TextureAspect::Plane0,
        ),
        (
            etalon::TextureFormat::P010,
            etalon::TextureFormat::Rg16Unorm,
            etalon::TextureAspect::Plane1,
        ),
    ] {
        let tex = device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: tex_format,
            usage: etalon::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        valid(&device, || {
            let _ = tex.create_view(&etalon::TextureViewDescriptor {
                format: Some(view_format),
                aspect: view_aspect,
                ..Default::default()
            });
        });
    }
}

/// Ensures that attempting to create a texture view from a specific plane of a
/// non-planar texture fails validation.
#[test]
fn non_planar_texture_view_plane() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        dimension: etalon::TextureDimension::D2,
        size,
        format: etalon::TextureFormat::R8Unorm,
        usage: etalon::TextureUsages::TEXTURE_BINDING,
        mip_level_count: 1,
        sample_count: 1,
        view_formats: &[],
    });
    fail(
        &device,
        || {
            let _ = tex.create_view(&etalon::TextureViewDescriptor {
                aspect: etalon::TextureAspect::Plane0,
                ..Default::default()
            });
        },
        Some("Aspect Plane0 is not a valid aspect of the source texture format R8Unorm"),
    );
}

/// Ensures that attempting to create a texture view from an invalid plane of a
/// planar texture fails validation.
#[test]
fn planar_texture_view_plane_out_of_bounds() {
    let required_features = etalon::Features::TEXTURE_FORMAT_NV12
        | etalon::Features::TEXTURE_FORMAT_P010
        | etalon::Features::TEXTURE_FORMAT_16BIT_NORM;
    let device_desc = etalon::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = etalon::Device::noop(&device_desc);
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    for (tex_format, view_format, view_aspect) in [
        (
            etalon::TextureFormat::NV12,
            etalon::TextureFormat::R8Unorm,
            etalon::TextureAspect::Plane2,
        ),
        (
            etalon::TextureFormat::P010,
            etalon::TextureFormat::R16Unorm,
            etalon::TextureAspect::Plane2,
        ),
    ] {
        let tex = device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: tex_format,
            usage: etalon::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        fail(
            &device,
            || {
                let _ = tex.create_view(&etalon::TextureViewDescriptor {
                    format: Some(view_format),
                    aspect: view_aspect,
                    ..Default::default()
                });
            },
            Some(&format!(
                "Aspect {view_aspect:?} is not a valid aspect of the source texture format {tex_format:?}"
            )),
        );
    }
}

/// Ensures that attempting to create a texture view from a specific plane of a
/// planar texture with an invalid format fails validation.
#[test]
fn planar_texture_bad_view_format() {
    let required_features = etalon::Features::TEXTURE_FORMAT_NV12
        | etalon::Features::TEXTURE_FORMAT_P010
        | etalon::Features::TEXTURE_FORMAT_16BIT_NORM;
    let device_desc = etalon::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = etalon::Device::noop(&device_desc);
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    for (tex_format, view_format) in [
        (etalon::TextureFormat::NV12, etalon::TextureFormat::Rg8Unorm),
        (etalon::TextureFormat::P010, etalon::TextureFormat::Rg16Unorm),
    ] {
        fail(
            &device,
            || {
                let _ = device.create_texture(&etalon::TextureDescriptor {
                    label: None,
                    dimension: etalon::TextureDimension::D2,
                    size,
                    format: tex_format,
                    usage: etalon::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[view_format],
                });
            },
            Some(&format!(
                "The view format {view_format:?} is not compatible with texture \
                 format {tex_format:?}, only changing srgb-ness is allowed."
            )),
        );
    }
}

/// Ensures that attempting to create a planar texture with an invalid size
/// fails validation.
#[test]
fn planar_texture_bad_size() {
    let required_features =
        etalon::Features::TEXTURE_FORMAT_NV12 | etalon::Features::TEXTURE_FORMAT_P010;
    let device_desc = etalon::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = etalon::Device::noop(&device_desc);
    let size = etalon::Extent3d {
        width: 255,
        height: 255,
        depth_or_array_layers: 1,
    };
    for format in [etalon::TextureFormat::NV12, etalon::TextureFormat::P010] {
        fail(
            &device,
            || {
                let _ = device.create_texture(&etalon::TextureDescriptor {
                    label: None,
                    dimension: etalon::TextureDimension::D2,
                    size,
                    format,
                    usage: etalon::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[],
                });
            },
            Some(&format!(
                "width {} is not a multiple of {format:?}'s width multiple requirement",
                size.width
            )),
        );
    }
}

/// Ensures that creating a planar textures that support `RENDER_ATTACHMENT` usage
/// is possible.
#[test]
fn planar_texture_render_attachment() {
    let required_features = etalon::Features::TEXTURE_FORMAT_NV12;
    let device_desc = etalon::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = etalon::Device::noop(&device_desc);
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    for (tex_format, view_format, view_aspect) in [
        (
            etalon::TextureFormat::NV12,
            etalon::TextureFormat::R8Unorm,
            etalon::TextureAspect::Plane0,
        ),
        (
            etalon::TextureFormat::NV12,
            etalon::TextureFormat::Rg8Unorm,
            etalon::TextureAspect::Plane1,
        ),
    ] {
        valid(&device, || {
            let texture = device.create_texture(&etalon::TextureDescriptor {
                label: None,
                dimension: etalon::TextureDimension::D2,
                size,
                format: tex_format,
                usage: etalon::TextureUsages::RENDER_ATTACHMENT,
                mip_level_count: 1,
                sample_count: 1,
                view_formats: &[],
            });

            let _ = texture.create_view(&etalon::TextureViewDescriptor {
                format: Some(view_format),
                aspect: view_aspect,
                ..Default::default()
            });
        });
    }
}

/// Ensures that creating a planar textures with `RENDER_ATTACHMENT`
/// for non renderable planar formats fails validation.
#[test]
fn planar_texture_render_attachment_unsupported() {
    let required_features =
        etalon::Features::TEXTURE_FORMAT_P010 | etalon::Features::TEXTURE_FORMAT_16BIT_NORM;
    let device_desc = etalon::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = etalon::Device::noop(&device_desc);
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    fail(
        &device,
        || {
            let _ = device.create_texture(&etalon::TextureDescriptor {
                label: None,
                dimension: etalon::TextureDimension::D2,
                size,
                format: etalon::TextureFormat::P010,
                usage: etalon::TextureUsages::RENDER_ATTACHMENT,
                mip_level_count: 1,
                sample_count: 1,
                view_formats: &[],
            });
        },
        Some("Texture usages TextureUsages(RENDER_ATTACHMENT) are not allowed on a texture of type P010"),
    );
}

/// Creates a texture and a buffer, and encodes a copy from the texture to the
/// buffer.
fn encode_copy_texture_to_buffer(
    device: &etalon::Device,
    format: etalon::TextureFormat,
    aspect: etalon::TextureAspect,
    bytes_per_texel: u32,
) -> etalon::CommandEncoder {
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format,
        usage: etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: (size.width * size.height * bytes_per_texel) as u64,
        usage: etalon::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

    encoder.copy_texture_to_buffer(
        etalon::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect,
        },
        etalon::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * bytes_per_texel),
                rows_per_image: None,
            },
        },
        size,
    );

    encoder
}

/// Ensures that attempting to copy a texture with a forbidden source format to
/// a buffer fails validation.
#[test]
fn copy_texture_to_buffer_forbidden_format() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    let format = etalon::TextureFormat::Depth24Plus;

    let encoder = encode_copy_texture_to_buffer(&device, format, etalon::TextureAspect::All, 4);

    fail(
        &device,
        || {
            encoder.finish();
        },
        Some(&format!(
            "Copying from textures with format {format:?} is forbidden"
        )),
    );
}

/// Ensures that attempting ta copy a texture with a forbidden source
/// format/aspect combination to a buffer fails validation.
#[test]
fn copy_texture_to_buffer_forbidden_format_aspect() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    let format = etalon::TextureFormat::Depth24PlusStencil8;
    let aspect = etalon::TextureAspect::DepthOnly;

    let encoder = encode_copy_texture_to_buffer(&device, format, aspect, 4);

    fail(
        &device,
        || {
            encoder.finish();
        },
        Some(&format!(
            "Copying from textures with format {format:?} and aspect {aspect:?} is forbidden"
        )),
    );
}

/// Creates a texture and a buffer, and encodes a copy from the buffer to the
/// texture.
fn encode_copy_buffer_to_texture(
    device: &etalon::Device,
    format: etalon::TextureFormat,
    aspect: etalon::TextureAspect,
    bytes_per_texel: u32,
) -> etalon::CommandEncoder {
    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format,
        usage: etalon::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: (size.width * size.height * bytes_per_texel) as u64,
        usage: etalon::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_texture(
        etalon::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * bytes_per_texel),
                rows_per_image: None,
            },
        },
        etalon::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect,
        },
        size,
    );

    encoder
}

/// Ensures that attempting to copy a buffer to a texture with a forbidden
/// destination format fails validation.
#[test]
fn copy_buffer_to_texture_forbidden_format() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    for format in [
        etalon::TextureFormat::Depth24Plus,
        etalon::TextureFormat::Depth32Float,
    ] {
        let encoder = encode_copy_buffer_to_texture(&device, format, etalon::TextureAspect::All, 4);

        fail(
            &device,
            || {
                encoder.finish();
            },
            Some(&format!(
                "Copying to textures with format {format:?} is forbidden"
            )),
        );
    }
}

/// Ensures that attempting to copy a buffer to a texture with a forbidden
/// destination format/aspect combination fails validation.
#[test]
fn copy_buffer_to_texture_forbidden_format_aspect() {
    let required_features = etalon::Features::DEPTH32FLOAT_STENCIL8;
    let device_desc = etalon::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = etalon::Device::noop(&device_desc);

    let aspect = etalon::TextureAspect::DepthOnly;

    for (format, bytes_per_texel) in [
        (etalon::TextureFormat::Depth24PlusStencil8, 4),
        (etalon::TextureFormat::Depth32FloatStencil8, 8),
    ] {
        let encoder = encode_copy_buffer_to_texture(&device, format, aspect, bytes_per_texel);

        fail(
            &device,
            || {
                encoder.finish();
            },
            Some(&format!(
                "Copying to textures with format {format:?} and aspect {aspect:?} is forbidden"
            )),
        );
    }
}

/// Ensures that attempting to create a texture with [`etalon::TextureUsages::TRANSIENT_ATTACHMENT`]
/// and its unsupported usages fails validation.
#[test]
fn transient_invalid_usage() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let invalid_usages = etalon::TextureUsages::all()
        - etalon::TextureUsages::RENDER_ATTACHMENT
        - etalon::TextureUsages::TRANSIENT_ATTACHMENT;

    for usage in invalid_usages {
        let usage = etalon::TextureUsages::RENDER_ATTACHMENT
            | etalon::TextureUsages::TRANSIENT_ATTACHMENT
            | usage;
        let invalid_texture_descriptor = etalon::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        };
        fail(
            &device,
            || device.create_texture(&invalid_texture_descriptor),
            Some(&format!("Transient texture usage must be equal to `TRANSIENT_ATTACHMENT | RENDER_ATTACHMENT`, but got `{usage:?}`")),
        );
    }

    let invalid_texture_descriptor = etalon::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::TRANSIENT_ATTACHMENT,
        view_formats: &[],
    };
    fail(
        &device,
        || device.create_texture(&invalid_texture_descriptor),
        Some("Transient texture usage must be equal to `TRANSIENT_ATTACHMENT | RENDER_ATTACHMENT`, but got `TextureUsages(TRANSIENT_ATTACHMENT)`"),
    );
}

/// Ensures that attempting to use a texture of [`etalon::TextureUsages::TRANSIENT_ATTACHMENT`]
/// with [`etalon::StoreOp::Store`] fails validation.
#[test]
fn transient_invalid_storeop() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    let size = etalon::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let transient_texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TRANSIENT_ATTACHMENT,
        view_formats: &[],
    });

    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

            let invalid_render_pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: &transient_texture.create_view(&etalon::TextureViewDescriptor::default()),
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::TRANSPARENT),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            drop(invalid_render_pass);

            encoder.finish()
        },
      Some("Color attachment with `TRANSIENT_ATTACHMENT` usage can only be used with `LoadOp::Clear` or `LoadOp::DontCare` (if it is available) and  `StoreOp::Discard`. Operations `(Clear(Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }), Store)` were provided")
    );
}

/// <https://bugzilla.mozilla.org/show_bug.cgi?id=2053860>
#[test]
fn no_overflow_in_texture_selector() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());
    // ignore validation errors to not panic before reaching `TextureView::invalid`
    device.on_uncaptured_error(Arc::new(|_| {}));
    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: 59,
            height: 401,
            depth_or_array_layers: 1948,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::R8Unorm,
        usage: etalon::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let _view = texture.create_view(&etalon::TextureViewDescriptor {
        base_array_layer: 3385121660,
        array_layer_count: Some(3815184380),
        ..Default::default()
    });
}
