# Standalone native P64 measurements

The standalone runner was executed on Windows x86_64 with Rust 1.96.1 in release
mode against the native P64 SmolLM2-360M fixture. Four generated tokens, context
32, prompt IDs `1,42,17`, and one warm repeat produced identical token IDs and
maximum-logit bits across reversible and direct execution, including both cold
and warm runs. The caller budgets were 1 GiB for buffers and 64 MiB for artifacts;
the retained evidence occupied 9,580 bytes and temporary directories were removed.
This is a bounded numerical parity check, not quality, GPU or throughput
qualification. An earlier unoptimized run hit its 300-second deadline during
the second configuration and retained a failed receipt.

From the Etalon workspace, with an existing native P64 model and existing scratch
parent directory:

```powershell
cargo run -p etalon-inference-lab --features standalone-cpu --example standalone_p64_lab -- --mode standalone-developer --profile dense-p64-cpu --workflow ablate --model C:/models/model.p64 --prompt-ids 1,42,17 --context 10240 --decode-tokens 10000 --warm-repeats 1 --trials 2 --buffer-bytes 8589934592 --artifact-bytes 67108864 --scratch-parent C:/lab --retain C:/lab/retained-run-001 --timeout-seconds 86400
```

The byte budget is caller selected. This example needs a 64-bit process and enough
physical memory for the actual model and admitted buffers. Admission does not imply
that the host can provide that memory. Use a smaller decode count/context when
exploring constrained devices. Decode is bounded to 10,000 tokens per sequence;
that is an operational source limit, not evidence of a completed 10,000-token run.
The prompt IDs must come from the model's actual tokenizer. This tool never guesses
a tokenizer or silently interprets text as IDs.

Set `--workflow benchmark` for one configuration, `search` for the existing Sobol/EI
search engine, `campaign` for the existing campaign/belief/report pipeline, or
`ablate` for the reversible/direct pair. Search/campaign use an explicitly selected
raw decode-throughput objective. Existing callers keep the quality-weighted default.
The configuration space currently contains exactly two points, so search can finish
before the requested trial count. Campaign adds its baseline outside its trial
budget, consistent with the existing campaign API. Trials are capped at 64 and warm
repeats at 16 in the CLI. Each trial runs once from fresh zeroed KV, then repeats
with the same loaded weights and zeroed KV; it does not reuse a cached prefix.

`reversible_decode=true` executes the actual whole-workspace undo/proposal/commit
mechanism. `false` executes the canonical direct dense forward/argmax function.
Both use the same native P64 bindings, codecs and scalar CPU kernel. Other axes,
including GPU fusion, cooperative GEMV and quantized KV flags, are rejected. This
is an ablation of developer rollback overhead, not a permission toggle for governed
application inference. The CLI also compares the generated ID sequences across
the ablation pair; this is not a semantic-quality assessment.

The supported family is the existing P64 dense Llama-compatible binder: separate
Q/K/V, ordinary gated FFN, supported GGML tensor codecs, no MoE router, hybrid SSM,
Q/K normalization or fused QKV. The binder rejects unsupported metadata/tensors.
No GGUF conversion or fallback occurs at this runtime boundary. The executor is
native scalar CPU; no GPU execution is claimed. Native P64 MoE, FTW and Qwen
sequence profiles remain unimplemented in this CLI. Existing provider-based lab
interfaces, ablation APIs and backend implementations remain available to hosts.

The reusable entry point is `standalone::StandaloneCpuExecutor`, implementing
`runner::ExperimentExecutor`. It composes with ordinary `ExperimentConfig`, search
and campaign APIs. `StandaloneOptions` accepts an externally owned atomic
cancellation flag and deadline. Checks occur before every prompt/decode token;
a running individual kernel, file read or model preparation is not interrupted.
`with_quality` and `with_telemetry` preserve independent host-supplied providers.
Quality providers receive the actual token-ID CSV, so a semantic evaluator must
perform model-appropriate decoding itself. CLI quality, temperature, power, energy,
peak RSS and repair measurements remain unavailable. Legacy numeric quality/thermal
fields remain compatibility placeholders; `evidence.raw_quality`, `energy_j`, and
optional telemetry samples determine availability. Greedy argmax consumes no RNG
seed and does not claim otherwise. Missing seed/quality/energy qualification keeps
these exploratory results outside the qualified Pareto frontier.

Each trial records model SHA-256, prompt SHA-256, a selected numerical/runtime
source-set SHA-256, OS/architecture, requested/actual token budget and a CPU
execution receipt. Per-repeat artifacts retain raw token IDs, exact max-logit bits
and individual step nanoseconds, with separate prefill/decode/first-token timings.
Load reporting separates the initial model read/hash from per-trial parse/CRC/
binding/workspace preparation. These are wall-clock timings, not device timestamps.
The source-set digest identifies the listed compiled-in sources, not the complete
binary, compiler, lockfile or machine. Preserve those independently for release
qualification. No kernel-allocation counter is asserted without measurement.

All output goes through a streaming byte-limited artifact sink in one unique
`tempfile::TempDir` below the explicit scratch parent. Partial writes are removed;
the directory is removed by RAII on ordinary success, error and unwind. There is
no broad stale-directory cleanup. `--retain` promotes into a new caller-selected
destination on the same filesystem. An existing destination is refused. Disk
budget exhaustion fails the trial and cannot produce a successful result receipt.
Without retention, raw artifacts are removed at exit. The buffer budget covers
the model allocation, index/bindings, execution/undo buffers, prompt/raw samples,
ID string and fixed overhead; it does not claim to bound process RSS, OS caches,
arbitrary host quality/telemetry providers or the existing orchestration engine.
The CLI's fixed trial/token/repeat caps bound orchestration result growth; hosts
using campaign/search directly must choose their own orchestration limits.

Remaining family integration interfaces are `ordinary::moe::execute_moe` with its
real `LayerProvider`, and Qwen's `qwen::decode::{decode_tokens_with_sink,
decode_tokens_into}` with its runtime index, PLE/trunk reader and model lease.
Those are not substitutes for a validated native P64 end-to-end family driver.
The current Qwen route still needs the retained GGUF trunk/raw PLE/HMC adapter.
FTW output projection support alone does not establish a complete FTW sequence
runner. A follow-up must bind those complete canonical engines to native P64
without inventing missing inference math or routing unsupported models to dense.
