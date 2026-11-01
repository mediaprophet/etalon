use serde_json::{json, Value};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use crate::browser::failure;

pub(crate) fn buffer(
    device: &etalon::Device,
    size: u64,
    usage: etalon::BufferUsages,
) -> etalon::Buffer {
    device.create_buffer(&etalon::BufferDescriptor {
        label: Some("bounded browser test buffer"),
        size,
        usage,
        mapped_at_creation: false,
    })
}

pub(crate) async fn read(buffer: &etalon::Buffer) -> Result<Vec<u8>, Value> {
    let slice = buffer.slice(..);
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        slice.map_async(etalon::MapMode::Read, move |mapped| match mapped {
            Ok(()) => {
                let _ = resolve.call0(&JsValue::UNDEFINED);
            }
            Err(error) => {
                let _ = reject.call1(&JsValue::UNDEFINED, &JsValue::from_str(&error.to_string()));
            }
        });
    });
    JsFuture::from(promise).await.map_err(|error| {
        failure(
            "map-failed",
            &error
                .as_string()
                .unwrap_or_else(|| "Buffer mapping failed".into()),
        )
    })?;
    let bytes = slice
        .get_mapped_range()
        .map(|view| view.to_vec())
        .map_err(|error| failure("mapped-range-failed", &error.to_string()));
    buffer.unmap();
    bytes
}

pub(crate) async fn buffer_case(
    device: &etalon::Device,
    queue: &etalon::Queue,
    zero: bool,
) -> Result<Value, Value> {
    let usage = etalon::BufferUsages::COPY_SRC | etalon::BufferUsages::COPY_DST;
    let src = buffer(device, 1024, usage);
    let dst = buffer(
        device,
        1024,
        etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
    );
    let expected: Vec<u8> = (0..1024)
        .map(|i| if zero { 0 } else { ((i * 37 + 11) % 256) as u8 })
        .collect();
    if !zero {
        queue.write_buffer(&src, 0, &expected);
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 1024);
    queue.submit([encoder.finish()]);
    let mapped = read(&dst).await;
    src.destroy();
    dst.destroy();
    let actual = mapped?;
    if actual != expected {
        return Err(failure(
            "buffer-mismatch",
            "Readback differs from expected bytes",
        ));
    }
    Ok(json!({"compared_bytes": expected.len(), "zero_initialized": zero}))
}
