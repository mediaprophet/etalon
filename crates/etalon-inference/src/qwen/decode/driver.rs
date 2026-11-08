use super::*;
/// Drive a full decode: every prompt token updates the recurrent state, then
/// `max_tokens` argmax steps generate the output.  Stops early on a stop token
/// supplied by the caller (GGUF EOS / chat-end set).
pub fn decode_tokens<H: super::super::QwenHyperparams>(
    runtime: &mut Qwen4ExpRuntime<'_, H>,
    session: &mut Qwen4ExpSession,
    scratch: &mut Qwen4ExpDecodeScratch,
    prompt: &[u32],
    max_tokens: usize,
    is_stop: &dyn Fn(u32) -> bool,
    receipt: &mut Qwen4ExpDecodeReceipt,
    trace: &mut Vec<Qwen4ExpTraceRecord>,
) -> Result<(), Qwen4ExpDecodeError> {
    let trunk_start = runtime.trunk.io_stats();
    let ple_start = runtime.ple.io_stats();
    let start = Instant::now();
    // Compatibility receipt construction is cold and deliberately allocating.
    let mut output = vec![0u32; max_tokens];
    let count = decode_tokens_into(
        runtime,
        session,
        scratch,
        prompt,
        max_tokens,
        is_stop,
        &mut output,
        trace,
        &mut |_, _| true,
    )?;
    receipt.prompt_tokens = prompt.len();
    receipt.generated_tokens.clear();
    receipt.generated_tokens.extend_from_slice(&output[..count]);
    receipt.elapsed = start.elapsed();
    let trunk_end = runtime.trunk.io_stats();
    let ple_end = runtime.ple.io_stats();
    receipt.trunk_reads = trunk_end.reads - trunk_start.reads;
    receipt.trunk_rows = trunk_end.rows - trunk_start.rows;
    receipt.trunk_bytes = trunk_end.bytes - trunk_start.bytes;
    receipt.ple_reads = ple_end.reads - ple_start.reads;
    receipt.ple_rows = ple_end.rows - ple_start.rows;
    receipt.ple_bytes = ple_end.bytes - ple_start.bytes;
    receipt.state_bytes = session.state_bytes();
    receipt.scratch_bytes = scratch.scratch_bytes();
    Ok(())
}

/// Sink-driven generation. The sink owns acceptance and output custody.
/// On any error the numerical owner must be discarded, never resumed.
/// Tracing is caller-selected; use `()` for the allocation-free hot path.
pub fn decode_tokens_with_sink<H: super::super::QwenHyperparams>(
    runtime: &mut Qwen4ExpRuntime<'_, H>,
    session: &mut Qwen4ExpSession,
    scratch: &mut Qwen4ExpDecodeScratch,
    prompt: &[u32],
    max_tokens: usize,
    is_stop: &dyn Fn(u32) -> bool,
    trace: &mut impl Qwen4ExpTraceSink,
    accept: &mut impl FnMut(StreamedArgmax, usize) -> bool,
) -> Result<usize, Qwen4ExpDecodeError> {
    prompt
        .len()
        .checked_add(max_tokens.saturating_sub(1))
        .ok_or(Qwen4ExpDecodeError::SizeOverflow)?;
    let (&last, prefix) = prompt
        .split_last()
        .ok_or(Qwen4ExpDecodeError::MissingTensor)?;
    if max_tokens == 0 {
        return Ok(0);
    }
    for &token in prefix {
        decode_step(runtime, session, scratch, token, trace, false)?;
    }
    let mut winner = decode_step(runtime, session, scratch, last, trace, false)?;
    let mut written = 0;
    while written < max_tokens {
        if is_stop(winner.token_id) {
            break;
        }
        let position = prompt.len() - 1 + written;
        if !accept(winner, position) {
            return Err(Qwen4ExpDecodeError::PublicationDenied);
        }
        written += 1;
        // The final candidate need not advance any unobserved recurrent state.
        if written < max_tokens {
            winner = decode_step(runtime, session, scratch, winner.token_id, trace, false)?;
        }
    }
    Ok(written)
}

/// Caller-buffered adapter. The sink is responsible for custody while writing.
pub fn decode_tokens_into<H: super::super::QwenHyperparams>(
    runtime: &mut Qwen4ExpRuntime<'_, H>,
    session: &mut Qwen4ExpSession,
    scratch: &mut Qwen4ExpDecodeScratch,
    prompt: &[u32],
    max_tokens: usize,
    is_stop: &dyn Fn(u32) -> bool,
    out: &mut [u32],
    trace: &mut impl Qwen4ExpTraceSink,
    accept: &mut impl FnMut(StreamedArgmax, usize) -> bool,
) -> Result<usize, Qwen4ExpDecodeError> {
    if out.len() < max_tokens {
        return Err(Qwen4ExpDecodeError::OutputTooSmall);
    }
    let first = prompt.len().saturating_sub(1);
    decode_tokens_with_sink(
        runtime,
        session,
        scratch,
        prompt,
        max_tokens,
        is_stop,
        trace,
        &mut |winner, position| {
            if !accept(winner, position) {
                return false;
            }
            out[position - first] = winner.token_id;
            true
        },
    )
}
