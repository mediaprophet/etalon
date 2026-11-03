//! Qwen4Exp four-stream Hyper-Connection vector operations.
//!
//! Projection weights are streamed elsewhere.  This module applies their
//! per-stream gates to hidden-state buffers supplied by the caller.

use super::QwenNumericError;

/// Normalize each Hyper-Connection stream independently with RMSNorm.
/// The GGUF converter pre-folds the zero-centered `(1 + weight)` scale into
/// the stored gamma, so the checkpoint bytes multiply directly — callers
/// must not add a further bias (llama.cpp `build_hc_mix`).
pub fn group_rms_norm_into(
    streams: &[f32],
    stream_count: usize,
    hidden: usize,
    weight: &[f32],
    epsilon: f32,
    out: &mut [f32],
) -> Result<(), QwenNumericError> {
    let needed = stream_count
        .checked_mul(hidden)
        .ok_or(QwenNumericError::BufferTooSmall)?;
    if streams.len() < needed || weight.len() < needed || out.len() < needed {
        return Err(QwenNumericError::BufferTooSmall);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(QwenNumericError::InvalidEpsilon);
    }
    for stream in 0..stream_count {
        let offset = stream * hidden;
        let mut squared_sum = 0.0f32;
        for index in 0..hidden {
            let value = streams[offset + index];
            squared_sum += value * value;
        }
        let inv_rms = 1.0 / (squared_sum / hidden.max(1) as f32 + epsilon).sqrt();
        for index in 0..hidden {
            out[offset + index] = streams[offset + index] * inv_rms * weight[offset + index];
        }
    }
    Ok(())
}

/// In-place variant for a transient stream buffer whose unnormalized values
/// are not needed by the caller after the operation.  Same pre-folded gamma
/// semantics as `group_rms_norm_into`.
pub fn group_rms_norm_in_place(
    values: &mut [f32],
    stream_count: usize,
    hidden: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<(), QwenNumericError> {
    let needed = stream_count
        .checked_mul(hidden)
        .ok_or(QwenNumericError::BufferTooSmall)?;
    if values.len() < needed || weight.len() < needed {
        return Err(QwenNumericError::BufferTooSmall);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(QwenNumericError::InvalidEpsilon);
    }
    for stream in 0..stream_count {
        let offset = stream * hidden;
        let mut squared_sum = 0.0f32;
        for index in 0..hidden {
            let value = values[offset + index];
            squared_sum += value * value;
        }
        let inv_rms = 1.0 / (squared_sum / hidden.max(1) as f32 + epsilon).sqrt();
        for index in 0..hidden {
            values[offset + index] *= inv_rms * weight[offset + index];
        }
    }
    Ok(())
}

/// Form the token-mixer or MoE input from the gated average of Hyper streams.
/// `mix_gate` is the sigmoid output of the streamed HC up-projection.
pub fn mix_streams_into(
    normalized: &[f32],
    mix_gate: &[f32],
    stream_count: usize,
    hidden: usize,
    out: &mut [f32],
) -> Result<(), QwenNumericError> {
    let needed = stream_count
        .checked_mul(hidden)
        .ok_or(QwenNumericError::BufferTooSmall)?;
    if normalized.len() < needed || mix_gate.len() < needed || out.len() < hidden {
        return Err(QwenNumericError::BufferTooSmall);
    }
    for index in 0..hidden {
        let mut sum = 0.0f32;
        for stream in 0..stream_count {
            let offset = stream * hidden + index;
            sum += normalized[offset] * mix_gate[offset];
        }
        out[index] = sum / stream_count.max(1) as f32;
    }
    Ok(())
}

/// Add a block update into each Hyper stream.  `inject_gate` is the sigmoid
/// output of the streamed HC inject projection; Qwen scales it by two.
pub fn inject_stream_update(
    streams: &mut [f32],
    update: &[f32],
    inject_gate: &[f32],
    stream_count: usize,
    hidden: usize,
) -> Result<(), QwenNumericError> {
    let needed = stream_count
        .checked_mul(hidden)
        .ok_or(QwenNumericError::BufferTooSmall)?;
    if streams.len() < needed || inject_gate.len() < needed || update.len() < hidden {
        return Err(QwenNumericError::BufferTooSmall);
    }
    for stream in 0..stream_count {
        let offset = stream * hidden;
        for index in 0..hidden {
            streams[offset + index] += update[index] * (2.0 * inject_gate[offset + index]);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "hyper_connection_tests.rs"]
mod tests;
