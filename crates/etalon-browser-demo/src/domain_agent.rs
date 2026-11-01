use core::{future::Future, task::Poll};
use std::sync::Arc;

use etalon_agent::{
    AccessRights, AllocationId, BufferRegion, CapabilityState, ComputeLane, HalComputeLane,
    HalOracleError, QueueId, SubmissionToken,
};
use serde_json::{json, Value};

use crate::{browser::failure, readback};

fn region(slot: u32, len: u64) -> BufferRegion {
    BufferRegion {
        allocation: AllocationId {
            device_id: 1,
            slot,
            generation: 1,
        },
        byte_offset: 0,
        byte_len: len,
        alignment: 4,
        access: AccessRights::ReadOnly,
    }
}

fn copy_command(
    device: &etalon::Device,
    source: &etalon::Buffer,
    stage: &etalon::Buffer,
    len: u64,
) -> etalon::CommandBuffer {
    let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
        label: Some("agent queue-bound lease qualification"),
    });
    encoder.copy_buffer_to_buffer(source, 0, stage, 0, len);
    encoder.finish()
}

pub(crate) async fn run(device: &etalon::Device, queue: &etalon::Queue) -> Result<Value, Value> {
    // HalComputeLane shares its native/browser API; browser handles stay on this thread.
    #[allow(clippy::arc_with_non_send_sync)]
    let device_handle = Arc::new(device.clone());
    let queue_handle = Arc::new(queue.clone());
    let mut lane = HalComputeLane::new(
        Arc::clone(&device_handle),
        Arc::clone(&queue_handle),
        QueueId(73),
    );
    let foreign_lane = HalComputeLane::new(device_handle, queue_handle, QueueId(73));
    let expected: Vec<u8> = (0..256).map(|i| ((i * 37 + 11) % 256) as u8).collect();
    let source = readback::buffer(
        device,
        256,
        etalon::BufferUsages::COPY_SRC | etalon::BufferUsages::COPY_DST,
    );
    let stage = readback::buffer(
        device,
        256,
        etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&source, 0, &expected);
    let command = copy_command(device, &source, &stage, 256);
    let (token, lease) = lane
        .lease_submission(command, source, region(0, 256))
        .map_err(|(_, error)| failure("agent-lease-submit", error))?;
    if lease.token() != token || lease.region() != region(0, 256) {
        return Err(failure(
            "agent-lease-description",
            "Lease token or buffer region changed",
        ));
    }
    if lane.state() == CapabilityState::Verified || lane.verified_submission().is_some() {
        return Err(failure(
            "agent-unverified-submission",
            "Submission was marked verified before a CPU oracle",
        ));
    }
    if lane.verify_submission(token, &expected, &expected)
        != Err(HalOracleError::IncompleteSubmission)
    {
        return Err(failure(
            "agent-incomplete-oracle",
            "Unfinished GPU submission accepted an oracle",
        ));
    }
    let lease = match lease.try_reclaim(&foreign_lane) {
        Err(lease) => lease,
        Ok(buffer) => {
            buffer.destroy();
            return Err(failure(
                "agent-foreign-lease",
                "A different lane with the same QueueId reclaimed a lease",
            ));
        }
    };
    let lease = match lease.try_reclaim(&lane) {
        Err(lease) => lease,
        Ok(buffer) => {
            buffer.destroy();
            return Err(failure(
                "agent-early-lease",
                "A pending resource was reclaimed before queue completion",
            ));
        }
    };
    let mut pending_wait = Box::pin(lane.wait_for_async(token));
    let first_poll = core::future::poll_fn(|cx| Poll::Ready(pending_wait.as_mut().poll(cx))).await;
    if !first_poll.is_pending() {
        return Err(failure(
            "agent-wait-cancel",
            "Wait completed before cancellation could be exercised",
        ));
    }
    drop(pending_wait);
    if !lane.wait_for_async(token).await {
        return Err(failure(
            "agent-queue-completion",
            "Issued WebGPU submission did not complete",
        ));
    }
    let actual = readback::read(&stage).await?;
    if actual != expected {
        return Err(failure(
            "agent-copy-oracle",
            "Queue copy differs from its CPU byte oracle",
        ));
    }
    lane.verify_submission(token, &expected, &actual)
        .map_err(|error| failure("agent-oracle-verification", &error.to_string()))?;
    if lane.state() != CapabilityState::Verified || lane.verified_submission() != Some(token) {
        return Err(failure(
            "agent-verified-state",
            "Matching GPU readback did not establish verified state",
        ));
    }
    let mut mismatch = actual.clone();
    mismatch[0] ^= 1;
    if lane.verify_submission(token, &expected, &mismatch)
        != Err(HalOracleError::OutputMismatch { index: 0 })
        || lane.state() == CapabilityState::Verified
        || lane.verified_submission().is_some()
    {
        return Err(failure(
            "agent-mismatch-state",
            "Incorrect readback did not revoke verified state",
        ));
    }
    if lane
        .wait_for_async(SubmissionToken::new(QueueId(74), token.fence_value))
        .await
        || lane
            .wait_for_async(SubmissionToken::new(token.queue, token.fence_value + 1))
            .await
    {
        return Err(failure(
            "agent-invalid-wait",
            "Foreign or unissued token was accepted",
        ));
    }
    lane.verify_submission(token, &expected, &actual)
        .map_err(|error| failure("agent-oracle-reverification", &error.to_string()))?;
    let source = lease.try_reclaim(&lane).map_err(|_| {
        failure(
            "agent-completed-lease",
            "Completed queue retained its resource",
        )
    })?;
    source.destroy();
    stage.destroy();

    let second_expected: Vec<u8> = (0..64).map(|i| (255 - i * 3) as u8).collect();
    let second_source = readback::buffer(
        device,
        64,
        etalon::BufferUsages::COPY_SRC | etalon::BufferUsages::COPY_DST,
    );
    let second_stage = readback::buffer(
        device,
        64,
        etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&second_source, 0, &second_expected);
    let command = copy_command(device, &second_source, &second_stage, 64);
    let (second_token, second_lease) = lane
        .lease_submission(command, second_source, region(1, 64))
        .map_err(|(_, error)| failure("agent-drop-lease-submit", error))?;
    drop(second_lease);
    if !lane.wait_for_async(second_token).await {
        return Err(failure(
            "agent-dropped-lease-completion",
            "Dropped-lease submission did not complete",
        ));
    }
    let second_actual = readback::read(&second_stage).await?;
    second_stage.destroy();
    if second_actual != second_expected {
        return Err(failure(
            "agent-dropped-lease-oracle",
            "Dropping a pending buffer lease changed the GPU copy",
        ));
    }
    if lane.verify_submission(token, &expected, &actual) != Err(HalOracleError::StaleSubmission) {
        return Err(failure(
            "agent-stale-oracle",
            "An old oracle verified a newer submission",
        ));
    }
    lane.verify_submission(second_token, &second_expected, &second_actual)
        .map_err(|error| failure("agent-final-oracle", &error.to_string()))?;
    Ok(
        json!({"execution": "Etalon WebGPU queue submission, actual Buffer leases and GPU copy readback",
        "submissions": 2, "compared_bytes": expected.len() + second_expected.len(),
        "actual_buffer_lease": true, "pending_resource_reclaim_rejected": true,
        "same_id_foreign_lane_rejected": true, "pending_async_wait_cancelled": true,
        "oracle_required_for_verified": true, "mismatch_revokes_verified": true,
        "stale_oracle_rejected": true, "dropped_pending_lease_copy_verified": true,
        "final_state": format!("{:?}", lane.state()), "mismatches": 0}),
    )
}
