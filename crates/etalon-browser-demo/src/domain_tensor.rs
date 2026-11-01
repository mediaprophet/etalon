use bytemuck::Zeroable;
use core::{future::Future, ops::Deref, task::Poll};
use etalon_tensor::{
    ForwardArena, ForwardMode, FusedAttentionParams, FusedAttentionPipeline, TensorArenaError,
    TokenReadbackError, TopkParams, TopkReductionPipeline,
};
use serde_json::{json, Value};

use crate::{browser::failure, readback};

struct Buffer(etalon::Buffer);
impl Deref for Buffer {
    type Target = etalon::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        self.0.unmap();
        self.0.destroy();
    }
}
fn buffer(device: &etalon::Device, size: u64, usage: etalon::BufferUsages) -> Buffer {
    Buffer(readback::buffer(device, size, usage))
}
fn upload(
    device: &etalon::Device,
    queue: &etalon::Queue,
    bytes: &[u8],
    usage: etalon::BufferUsages,
) -> Buffer {
    let buffer = buffer(
        device,
        bytes.len() as u64,
        usage | etalon::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}
fn words(bytes: &[u8]) -> Result<Vec<u32>, Value> {
    if !bytes.len().is_multiple_of(4) {
        return Err(failure(
            "tensor-readback-length",
            "Readback length is not a whole number of words",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn entry(binding: u32, buffer: &etalon::Buffer) -> etalon::BindGroupEntry<'_> {
    etalon::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

async fn check_topk(
    pipeline: &TopkReductionPipeline,
    device: &etalon::Device,
    queue: &etalon::Queue,
    logits: &[f32],
    k: u32,
    block: u32,
) -> Result<Value, Value> {
    let blocks = (logits.len() as u32).div_ceil(block);
    let candidates = (blocks * k) as usize;
    let params = TopkParams {
        n: logits.len() as u32,
        k,
        block_size: block,
        cand_base: 1,
    };
    let logits_buf = upload(
        device,
        queue,
        bytemuck::cast_slice(logits),
        etalon::BufferUsages::STORAGE,
    );
    let params_buf = upload(
        device,
        queue,
        bytemuck::bytes_of(&params),
        etalon::BufferUsages::UNIFORM,
    );
    let initial_vals = vec![12345.0f32; candidates + 2];
    let initial_idx = vec![0xaabb_ccddu32; candidates + 2];
    let output_usage = etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_SRC;
    let vals = upload(
        device,
        queue,
        bytemuck::cast_slice(&initial_vals),
        output_usage,
    );
    let indices = upload(
        device,
        queue,
        bytemuck::cast_slice(&initial_idx),
        output_usage,
    );
    let size = (candidates as u64 + 2) * 4;
    let staging = buffer(
        device,
        size * 2,
        etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
    );
    let group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: Some("actual tensor top-k oracle"),
        layout: &pipeline.layout,
        entries: &[
            entry(0, &logits_buf),
            entry(1, &params_buf),
            entry(2, &vals),
            entry(3, &indices),
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pipeline.dispatch(&mut pass, &group, blocks);
    }
    encoder.copy_buffer_to_buffer(&vals, 0, &staging, 0, size);
    encoder.copy_buffer_to_buffer(&indices, 0, &staging, size, size);
    queue.submit([encoder.finish()]);
    let raw = readback::read(&staging).await?;
    if raw.len() != size as usize * 2 {
        return Err(failure(
            "tensor-topk-length",
            "Unexpected candidate byte count",
        ));
    }
    let actual_vals = words(&raw[..size as usize])?;
    let actual_indices = words(&raw[size as usize..])?;
    let last = candidates + 1;
    if actual_vals[0] != initial_vals[0].to_bits()
        || actual_vals[last] != initial_vals[last].to_bits()
        || actual_indices[0] != initial_idx[0]
        || actual_indices[last] != initial_idx[last]
    {
        return Err(failure(
            "tensor-topk-bounds",
            "Candidate base or output guard was overwritten",
        ));
    }
    let mut invalid_candidates = 0;
    for (block_index, values) in logits.chunks(block as usize).enumerate() {
        let mut expected: Vec<(u32, f32)> = values
            .iter()
            .enumerate()
            .filter(|(_, value)| !value.is_nan())
            .map(|(i, &v)| ((block_index * block as usize + i) as u32, v))
            .collect();
        expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        for rank in 0..k as usize {
            let (index, value) = expected.get(rank).copied().unwrap_or((u32::MAX, f32::MIN));
            invalid_candidates += usize::from(index == u32::MAX);
            let output = 1 + block_index * k as usize + rank;
            let actual = f32::from_bits(actual_vals[output]);
            if actual_indices[output] != index || actual != value {
                return Err(failure("tensor-topk-oracle", &format!("Block {block_index}, rank {rank}: expected ({index}, {value}), got ({}, {actual})", actual_indices[output])));
            }
        }
    }
    Ok(
        json!({"elements": logits.len(), "k": k, "block_size": block, "workgroups": blocks,
        "candidates": candidates, "invalid_candidates": invalid_candidates, "output_guards_checked": true, "mismatches": 0}),
    )
}

pub(crate) async fn topk(device: &etalon::Device, queue: &etalon::Queue) -> Result<Value, Value> {
    let pipeline = TopkReductionPipeline::try_new(device)
        .map_err(|e| failure("tensor-topk-limits", &e.to_string()))?;
    let mut checks = Vec::new();
    checks.push(
        check_topk(
            &pipeline,
            device,
            queue,
            &[1.0, 4.0, 4.0, 2.0, -3.0, 8.0],
            2,
            4,
        )
        .await?,
    );
    checks.push(check_topk(&pipeline, device, queue, &[f32::NEG_INFINITY; 3], 4, 4).await?);
    checks.push(check_topk(&pipeline, device, queue, &[f32::NAN; 3], 2, 4).await?);
    checks.push(
        check_topk(
            &pipeline,
            device,
            queue,
            &[f32::NAN, f32::NEG_INFINITY, f32::MIN, -7.0, 9.0],
            4,
            4,
        )
        .await?,
    );
    let tail: Vec<f32> = (0..65).map(|i| (i % 7) as f32 - 3.0).collect();
    checks.push(check_topk(&pipeline, device, queue, &tail, 2, 4).await?);
    Ok(
        json!({"execution": "TopkReductionPipeline GPU WGSL", "workgroup_size": 64,
        "cpu_oracle": "independent descending stable candidate sort", "cases": checks,
        "ties_lowest_index": true, "tail_blocks_checked": true, "negative_infinity_checked": true,
        "all_nan_invalid_lanes_checked": true, "mismatches": 0}),
    )
}

pub(crate) async fn attention(
    device: &etalon::Device,
    queue: &etalon::Queue,
) -> Result<Value, Value> {
    let pipeline = FusedAttentionPipeline::try_new(device)
        .map_err(|e| failure("tensor-attention-limits", &e.to_string()))?;
    let params = FusedAttentionParams {
        n_embd: 2,
        n_head: 1,
        n_kv_head: 1,
        head_dim: 2,
        q_heads_per_kv: 1,
        max_context: 1,
        slot_kv_elems: 2,
        rope_theta_base: 10000.0,
        rope_scale: 1.0,
        num_tokens_in_batch: 1,
        mask_word_count: 1,
        out_stride_elems: 2,
        proj_row_stride: 2,
        ..FusedAttentionParams::zeroed()
    };
    let hidden = upload(
        device,
        queue,
        bytemuck::cast_slice(&[0.0f32, 0.0]),
        etalon::BufferUsages::STORAGE,
    );
    let weights = upload(
        device,
        queue,
        bytemuck::cast_slice(&[0u32]),
        etalon::BufferUsages::STORAGE,
    );
    let uniform = upload(
        device,
        queue,
        bytemuck::bytes_of(&params),
        etalon::BufferUsages::UNIFORM,
    );
    let kv = upload(
        device,
        queue,
        bytemuck::cast_slice(&[0.0f32, 0.0, 3.0, -2.0]),
        etalon::BufferUsages::STORAGE,
    );
    let out = upload(
        device,
        queue,
        bytemuck::cast_slice(&[f32::NAN; 2]),
        etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_SRC,
    );
    let mask = upload(
        device,
        queue,
        bytemuck::cast_slice(&[u32::MAX]),
        etalon::BufferUsages::STORAGE,
    );
    let staging = buffer(
        device,
        8,
        etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
    );
    let group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: Some("actual fused attention oracle"),
        layout: &pipeline.layout,
        entries: &[
            entry(0, &hidden),
            entry(1, &weights),
            entry(2, &uniform),
            entry(3, &kv),
            entry(4, &out),
            entry(5, &mask),
        ],
    });
    let mut checks = Vec::new();
    for masked in [false, true] {
        let params = FusedAttentionParams {
            mask_active: u32::from(masked),
            ..params
        };
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(
            &mask,
            0,
            bytemuck::bytes_of(&if masked { 0u32 } else { u32::MAX }),
        );
        queue.write_buffer(&out, 0, bytemuck::cast_slice(&[f32::NAN; 2]));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pipeline.dispatch(&mut pass, &group, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, 8);
        queue.submit([encoder.finish()]);
        let raw = readback::read(&staging).await?;
        let actual: Vec<f32> = words(&raw)?.into_iter().map(f32::from_bits).collect();
        let expected = if masked { [0.0, 0.0] } else { [3.0, -2.0] };
        if actual.len() != 2
            || actual
                .iter()
                .zip(expected)
                .any(|(&a, e)| !a.is_finite() || (a - e).abs() > 1e-5)
        {
            return Err(failure(
                "tensor-attention-oracle",
                &format!("Masked={masked}: expected {expected:?}, got {actual:?}"),
            ));
        }
        checks.push(
            json!({"all_masked": masked, "expected": expected, "actual": actual, "finite": true}),
        );
    }
    Ok(
        json!({"execution": "FusedAttentionPipeline GPU WGSL", "uniform_bytes": 96,
        "workgroup_size": 64, "heads": 1, "head_dim": 2, "tokens": 1,
        "cpu_oracle": "one-token softmax is 1; all-masked softmax is 0", "cases": checks, "mismatches": 0}),
    )
}

struct Arena(ForwardArena);
impl Drop for Arena {
    fn drop(&mut self) {
        self.0.readback_stage_buf.unmap();
        for buffer in [
            &self.0.hidden_state_buf,
            &self.0.residual_buf,
            &self.0.norm_buf,
            &self.0.readback_stage_buf,
        ] {
            buffer.destroy();
        }
    }
}
fn copy_token(
    arena: &ForwardArena,
    source: &etalon::Buffer,
    device: &etalon::Device,
    queue: &etalon::Queue,
    token: u32,
) -> Result<(), Value> {
    queue.write_buffer(source, 0, &token.to_le_bytes());
    let mut encoder = device.create_command_encoder(&Default::default());
    arena
        .try_record_readback_copy(&mut encoder, source, ForwardMode::DecodeSingle { pos: 0 })
        .map_err(|e| failure("tensor-forward-copy", &e.to_string()))?;
    queue.submit([encoder.finish()]);
    Ok(())
}

pub(crate) async fn forward(
    device: &etalon::Device,
    queue: &etalon::Queue,
) -> Result<Value, Value> {
    if !matches!(
        ForwardArena::try_new_with_budget(device, 1, 1, 1, 831),
        Err(TensorArenaError::BudgetExceeded)
    ) {
        return Err(failure(
            "tensor-forward-budget",
            "Insufficient aggregate padded-buffer budget was not rejected",
        ));
    }
    if !matches!(
        ForwardArena::try_new_with_budget(device, u32::MAX, u32::MAX, 1, u64::MAX),
        Err(TensorArenaError::SizeOverflow)
    ) {
        return Err(failure(
            "tensor-forward-overflow",
            "Allocation arithmetic overflow was not rejected",
        ));
    }
    let arena = Arena(
        ForwardArena::try_new_with_budget(device, 1, 1, 1, 832)
            .map_err(|e| failure("tensor-forward-allocation", &e.to_string()))?,
    );
    if !matches!(
        arena.0.try_read_token_blocking(device),
        Err(TokenReadbackError::BlockingUnsupported)
    ) {
        return Err(failure(
            "tensor-forward-blocking",
            "Blocking WebGPU readback did not return BlockingUnsupported",
        ));
    }
    let source = buffer(
        device,
        4,
        etalon::BufferUsages::COPY_SRC | etalon::BufferUsages::COPY_DST,
    );
    let tokens = [0u32, 1, u32::MAX, 42];
    for token in tokens {
        copy_token(&arena.0, &source, device, queue, token)?;
        let actual = arena
            .0
            .try_read_token_async(device)
            .await
            .map_err(|e| failure("tensor-forward-readback", &e.to_string()))?;
        if actual != token {
            return Err(failure(
                "tensor-forward-oracle",
                &format!("Expected token {token}, got {actual}"),
            ));
        }
    }
    let token = 0x1357_9bdf;
    copy_token(&arena.0, &source, device, queue, token)?;
    let mut pending = Box::pin(arena.0.try_read_token_async(device));
    let first_poll = core::future::poll_fn(|cx| Poll::Ready(pending.as_mut().poll(cx))).await;
    if !first_poll.is_pending() {
        return Err(failure(
            "tensor-forward-cancel",
            "Browser map completed before pending cancellation could be exercised",
        ));
    }
    if !matches!(
        arena.0.try_read_token_async(device).await,
        Err(TokenReadbackError::Busy)
    ) {
        return Err(failure(
            "tensor-forward-busy",
            "Concurrent readback did not return Busy",
        ));
    }
    drop(pending);
    let after_cancel = arena.0.read_token_async(device).await;
    if after_cancel != Some(token) {
        return Err(failure(
            "tensor-forward-cancel",
            "Staging could not be mapped again after cancellation",
        ));
    }
    copy_token(&arena.0, &source, device, queue, 77)?;
    if arena
        .0
        .try_read_token_async(device)
        .await
        .map_err(|e| failure("tensor-forward-readback", &e.to_string()))?
        != 77
    {
        return Err(failure(
            "tensor-forward-reuse",
            "Post-cancellation token copy/readback failed",
        ));
    }
    Ok(
        json!({"execution": "ForwardArena GPU copies and async token mapping", "arena_budget_bytes": 832,
        "budget_rejection_checked": true, "overflow_rejection_checked": true, "tokens": tokens,
        "blocking_wasm_rejected": true, "busy_error_checked": true, "pending_map_cancelled": true,
        "same_buffer_remapped_after_cancel": true, "post_cancel_copy_checked": true, "mismatches": 0}),
    )
}
