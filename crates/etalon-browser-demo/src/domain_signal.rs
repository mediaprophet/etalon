use crate::browser::failure;
use core::{future::Future, task::Poll};
use etalon_sense_gpu::{ByteOrder, DeviceId, EtalonIqPowerProvider, GpuAllocationClass, GpuError};
use serde_json::{json, Value};
use std::time::Duration;

const CAPACITY: usize = 65;
const SENTINEL: f64 = 12345.0;

fn samples(count: usize, order: ByteOrder, scale: f32) -> (Vec<u8>, Vec<f64>) {
    let mut bytes = Vec::with_capacity(count * 4);
    let mut expected = Vec::with_capacity(count);
    for row in 0..count {
        let i = (row as i16 % 17 - 8) * 3;
        let q = (row as i16 % 13 - 6) * 5;
        for value in [i, q] {
            bytes.extend_from_slice(&match order {
                ByteOrder::LittleEndian => value.to_le_bytes(),
                ByteOrder::BigEndian => value.to_be_bytes(),
            });
        }
        let i = f32::from(i) * scale;
        let q = f32::from(q) * scale;
        expected.push(f64::from(i * i + q * q));
    }
    (bytes, expected)
}

async fn check(
    provider: &mut EtalonIqPowerProvider<'_>,
    count: usize,
    order: ByteOrder,
    scale: f32,
) -> Result<Value, Value> {
    let (input, expected) = samples(count, order, scale);
    let mut workspace = [777.0f32; CAPACITY * 2 + 2];
    let mut output = [SENTINEL; CAPACITY + 2];
    let written = provider
        .compute_into_async(
            &input,
            order,
            scale,
            &mut workspace[1..count * 2 + 1],
            &mut output[1..count + 1],
        )
        .await
        .map_err(|e| failure("signal-gpu-execution", &format!("{e:?}")))?;
    if written != count
        || output[1..count + 1] != expected
        || output[0] != SENTINEL
        || output[count + 1..].iter().any(|&value| value != SENTINEL)
        || workspace[0] != 777.0
        || workspace[count * 2 + 1..]
            .iter()
            .any(|&value| value != 777.0)
    {
        return Err(failure(
            "signal-gpu-oracle",
            "GPU IQ powers or caller output/workspace bounds differ from the f32 oracle",
        ));
    }
    Ok(
        json!({"pairs": count, "byte_order": format!("{order:?}"), "scale": scale,
        "output_guards_checked": true, "mismatches": 0}),
    )
}

pub(crate) async fn run(device: &etalon::Device, queue: &etalon::Queue) -> Result<Value, Value> {
    let identity = DeviceId::new(0x4951_504f_5745_5201)
        .map_err(|_| failure("signal-device-id", "Invalid fixture registry ID"))?;
    let timeout = Duration::from_secs(5);
    let budget = CAPACITY as u64 * 12 + 16;
    if !matches!(
        EtalonIqPowerProvider::new_with_allocation_class(
            device,
            queue,
            identity,
            CAPACITY,
            GpuAllocationClass::ColdBoundedBuffers,
            budget - 1,
            timeout
        ),
        Err(GpuError::BufferBudgetExceeded)
    ) {
        return Err(failure(
            "signal-gpu-budget",
            "Insufficient persistent-buffer budget was not rejected",
        ));
    }
    let mut provider = EtalonIqPowerProvider::new_with_allocation_class(
        device,
        queue,
        identity,
        CAPACITY,
        GpuAllocationClass::ColdBoundedBuffers,
        budget,
        timeout,
    )
    .map_err(|e| failure("signal-gpu-setup", &format!("{e:?}")))?;
    let mut checks = Vec::new();
    for count in [1, 63, 64, 65] {
        for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
            for scale in [0.0, 0.5, -0.25] {
                checks.push(check(&mut provider, count, order, scale).await?);
            }
        }
    }
    let mut workspace = [777.0f32; CAPACITY * 2];
    let mut output = [SENTINEL; CAPACITY];
    let rejected = provider
        .compute_into_async(
            &[0u8; 3],
            ByteOrder::LittleEndian,
            1.0,
            &mut workspace[..2],
            &mut output[..1],
        )
        .await;
    if rejected != Err(GpuError::LengthMismatch) || output != [SENTINEL; CAPACITY] {
        return Err(failure(
            "signal-gpu-preflight",
            "Invalid input was not rejected before changing caller output",
        ));
    }
    let (input, _) = samples(CAPACITY, ByteOrder::LittleEndian, 0.5);
    let mut pending = Box::pin(provider.compute_into_async(
        &input,
        ByteOrder::LittleEndian,
        0.5,
        &mut workspace,
        &mut output,
    ));
    let first = core::future::poll_fn(|cx| Poll::Ready(pending.as_mut().poll(cx))).await;
    if !first.is_pending() {
        return Err(failure(
            "signal-gpu-cancel",
            "Browser operation completed before pending cancellation could be exercised",
        ));
    }
    drop(pending);
    if output != [SENTINEL; CAPACITY] {
        return Err(failure(
            "signal-gpu-cancel-output",
            "Cancellation changed the staged output",
        ));
    }
    let after_cancel = check(&mut provider, CAPACITY, ByteOrder::LittleEndian, 0.5).await?;
    Ok(
        json!({"execution": "Etalon portable IQ-power WGSL GPU kernel; no CPU fallback",
        "allocation_class": "ColdBoundedBuffers", "persistent_buffer_budget_bytes": budget,
        "precision": "f32 GPU power converted to caller f64 output", "workgroup_size": 64,
        "cases": checks, "budget_rejection_checked": true, "preflight_output_unchanged": true,
        "pending_map_cancelled": true, "same_provider_remapped": true,
        "after_cancellation": after_cancel, "mismatches": 0,
        "identity_note": "Opaque fixture registry ID is not hardware attestation"}),
    )
}
