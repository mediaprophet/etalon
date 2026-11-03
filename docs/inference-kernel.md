# Numerical inference kernel

`etalon-inference-kernel` contains dependency-free numerical contracts extracted
from Qualia's existing shared kernel. It provides dense f32 matrix-vector apply,
GGML Q4_K reconstruction and activation lookup, caller-owned workspace views,
paged GQA scalar attention and the existing segmented/tiled CUDA source templates.
It does not select a device, execute CUDA, authorize inference or publish tokens.

The package preserves the original standard-library math requirement, empty
default feature set and source license declaration. It has no external dependencies
and compiles for native and wasm32-unknown-unknown targets. Q4_K payloads here are
144 bytes per 256 values; the 160-byte Q4_K_SOA and DirectML formats are distinct.

The numerical source and its tests are moved without algorithm changes. Narrow
Clippy style allowances on module declarations preserve that frozen implementation
and API. They do not suppress correctness lints. The original 20 tests remain.

## Existing caller contracts and limitations

Use validated descriptors and correctly sized caller buffers. Public structs can
be constructed directly, so an arbitrary OperatorView is not proof of validation.
The low-level Q4_K lookup assumes input/output capacities supplied by its wrapper;
its lookup preparation currently observes the entire supplied input slice. Avoid
extra input tails when calling that low-level API. These historical behaviors are
retained for compatibility, pending separately tested improvements.

Paged attention uses the caller's immutable arena and page table and writes output
one head at a time. MissingPage can leave partial output; it is not a transactional
output API. PagedKvConfig has public fields and some derived size helpers retain
historical saturation/unchecked arithmetic. Callers must provide a valid bounded
configuration; helpers do not constitute complete validation of arbitrary values.

Embedded CUDA text preserves the original source and launch contract. Compilation
of this Rust crate does not prove a CUDA launch, numerical GPU parity or throughput.
Real device integration and Qualia forwarding adoption have separate qualification
gates. Developer/governed session control is outside this numerical package.

## Checks

```sh
cargo test --locked -p etalon-inference-kernel
cargo test --locked -p etalon-inference-kernel --release
cargo clippy --locked -p etalon-inference-kernel --all-targets -- -D warnings
cargo check --locked -p etalon-inference-kernel --target wasm32-unknown-unknown
```

Independent migration fixtures compare the frozen old implementation with this
library on the same target, preserving operation order and error/output behavior.
Cross-target floating-point equivalence and runtime backend execution need their
own evidence. Temporary original-source references are qualification artifacts,
not a second maintained numerical implementation.

On 3 November 2026, all 25 tests (20 preserved unit tests and five independent
contract tests) passed in debug and optimized builds, 50 executions with no
failures or ignored tests. The same five independent fixtures passed against the
frozen original library. They cover exact dense/Q4_K/attention fingerprints,
workspace and output tails, refusal behavior, nonfinite inputs, CUDA source bytes
and positive-controlled zero allocation. Strict Clippy and WASM compilation pass.
The initial strict Clippy attempt reported only inherited style findings; the
scoped annotations described above were independently reviewed.

## Dense CPU transformer execution

`dense_cpu` executes embedding, RMS normalization, interleaved RoPE, grouped-query
causal attention, KV updates, gated FFN, residuals and output projection. A
`TensorKernels` adapter supplies row decoding and matrix-vector operations;
the library owns the forward computation. `DensePlan` borrows tensor bindings,
checks dimensions and computes workspace requirements. `ingest_token` executes
prefill without output projection; `run_token` also produces logits and selects
the lowest-index finite maximum.

The caller supplies all thirteen execution buffers and the position. Admission
charges their full supplied capacities against an explicit byte budget, then
passes only used slices to kernels. The budget excludes model weights, bindings
and adapter-owned storage. Allocation behavior and codec correctness of an
arbitrary adapter require their own qualification.

Missing-tensor checks, numerical operation order and partial-failure behavior
follow the original dense CPU path. A failed step can leave KV and scratch
modified. `reset_kv` clears only used KV storage; it does not restore a prior
step, reset a sampler or authorize publication. Tokenization, architecture
admission, environment policy and governance stay outside this module. Native
MoE and recurrent execution are separate paths and have no dense fallback here.

### Dense qualification

On 3 November 2026, all 36 package tests passed in debug and release (72
executions), including five new unit tests and six independent dense tests.
Strict Clippy, formatting and WASM compilation passed. The independent tests
compare 42 states containing 83 f32 values each against actual original Qualia
forward code at `07018bb8dbb123f105a91609532b9b564e54a2af`, with bit-identical
debug and optimized traces. The two-layer F32 fixture covers prefill, decode,
reset, logits, every workspace, partial KV failures, invalid shapes, nonfinite
values and output rows differing from configured vocabulary. Additional checks
cover all thirteen buffer shortages, spare capacity, callback failures and
positive-controlled zero allocation of the library plus the fixture adapter.

A separate local baseline exercised the original thirteen native codec
row-decode/GEMV routes, truncated rows and direct native Q8 processing, and
confirmed dtype 158 remains unsupported in that CPU path. Those codec results
are distinct from the full F32 engine traces: they do not qualify a new
quantized-model adapter or actual WASM SIMD execution. Pretrained full-prompt
execution, GPU execution, Qualia engine adoption and physical rollback remain
separate gates. No throughput improvement is claimed.

Run the package commands above and the retained helper checks:

```sh
python crates/etalon-inference-kernel/tests/support/regenerate_dense_cpu_golden.py verify
python -B crates/etalon-inference-kernel/tests/support/test_regenerate_dense_cpu_golden.py
```

The six helper tests cover provenance/output validation, exclusive promotion
and cleanup on failure or interruption. To regenerate the reference into a new
file, use:

```sh
python crates/etalon-inference-kernel/tests/support/regenerate_dense_cpu_golden.py generate --qualia-root <Qualia-Git-checkout> --dependencies <Qualia>/target/debug/deps --output <new-file> --optimized
```

The helper reads SHA-checked Git objects at the frozen revision, independently
of the checkout's current files. It requires cached Rust 1.96.1 libraries for
serde, serde_json, bytemuck, half and memmap2; it performs no Cargo build,
network access or model download. Only cold construction bypasses the original
1000-token vocabulary floor for the tiny fixture. Actual forward/math/codecs
remain original. Source extraction has a two-MiB budget; the 32-MiB compiled
scratch check limits retention after compilation, not compiler peak memory.
Temporary scratch is cleaned and an existing destination is never overwritten.

## Routed experts, NVFP4 and recurrent primitives

`moe::routing`, `moe::nvfp4` and `recurrent::gated_deltanet` preserve the existing
Qualia numerical primitives. They use caller-owned output and state buffers and
do not require a database, policy identity or model file. They are separate from
model architecture preparation, complete MoE layers, streamed Qwen execution,
expert residency and normal Webizen-controlled token acceptance.

The routing module preserves top-k insertion/tie order, the selected-logit
softmax or nonnegative linear normalization, and weighted expert accumulation.
`MoeRoutingConfig::qwen3_6_35b` retains its existing configured 256-expert/top-eight
profile; the generic selector's stack capacity is 16. This profile name does not
certify a particular checkpoint. The NVFP4 module retains its nibble ordering,
FP8 conversion and block/global scale arithmetic, including signed zero and the
existing special-value mapping. It does not select a GPU or use native FP4
tensor cores.

The recurrent module preserves channel-major four-tap convolution history and
the existing matrix recurrence and output accumulation order. Despite the older
function documentation allowing per-head gates, the implementation uses only
the first alpha and beta values, or 1.0 for empty slices. This extraction keeps
that behavior; it does not silently change trained-model math.

These low-level compatibility functions retain existing input limitations:
nonempty routing with `k = 0` panics, expert indices beyond `u16` wrap, and some
dimension products are unchecked. Expert combination can zero or partially
update its output before returning a later dimension error; zero-weight experts
are skipped. Callers must validate their prepared shapes and routing capacity.
Checked entrypoints and correctness changes require separate validation rather
than changing these semantics during ownership migration.

Qualification of these primitives does not establish full Qwen, FTW, hybrid or
streamed-model compatibility, complete numerical rollback, or GPU/NPU execution.
Those families retain separate model, state, backend and caller gates.

The package passes 59 tests in both debug and release, including ten retained
primitive tests and thirteen independent hand-vector/allocation tests. Strict
all-target Clippy, package formatting and WASM library compilation pass. The
unchanged indexed arithmetic and public parameter lists retain narrowly scoped
style-lint allowances; no arithmetic was rewritten to satisfy Clippy.

A separate same-target original-versus-Cargo-library harness passes sixteen
tests in each profile: ten original tests and six differential groups. It checks
every FP8 byte and packed nibble pair with six global-scale cases, routing
capacities/ties/nonfinite values and refusals, matrix/row output tails, and
sixteen-step convolution and recurrence with full caller-state comparison.
Float values are compared by bits, including signed zero and NaNs. The private
optimized harness initially omitted the workspace's thin-LTO setting and failed
to link; matching that setting fixed the harness without changing the library.
No throughput improvement or unexecuted model/backend support is claimed.

## Qwen vector and Hyper-Connection arithmetic

`qwen::numerics` and `qwen::hyper_connection` provide the existing seven
caller-buffered vector, RMS normalization, stream-mixing and update-injection
functions. Their shared `QwenNumericError`, buffer/epsilon refusal order,
checked stream geometry, zero-shape behavior, pre-folded gamma and factor-two
injection arithmetic are preserved. Projection weights, model metadata and
streamed layer composition remain outside these helpers.

The in-place and out-of-place group normalization functions intentionally retain
their different multiplication associations. For `[0.3, 4.0]`, gamma
`[0.1, 1.0]` and epsilon `1e-6`, their first-element bits are respectively
`0x3c2d4ab8` and `0x3c2d4ab7`, matching the original functions. They must not be
collapsed into one implementation solely because the formulas look equivalent.

With these helpers, all 76 package tests pass in debug and release. This includes
nine additional independent numeric challenge groups and five allocation
controls. A separate original-versus-actual-Cargo-library harness passes eight
tests per profile, including three retained original tests, and was independently
rerun. Strict all-target Clippy, formatting and WASM library compilation pass.
The original arithmetic bodies are unchanged after line-ending normalization.
These results cover the helpers, not model parsing, complete streamed Qwen
execution, GPU/NPU dispatch, state rollback or Webizen acceptance.

## Qwen PLE row selection and history

`qwen::ple` owns trained n-gram row selection, fixed-capacity token history and
in-place row sorting/deduplication. `PleNgramConfig` borrows the original fixed
arrays; `PleNgramConfigSource` lets a host retain its owned metadata and wire
format. Each operation projects the configuration once and uses that same view
for row selection and EOS handling. Arbitrary host projection implementations
must be qualified separately for allocation and failure behavior.

The extraction preserves completeness and capacity refusal order, signed
wrapping products, XOR and Euclidean remainder. A late invalid row may leave a
written output prefix, but history advances only after successful selection.
EOS resets the count after row selection and retains the existing stale slots.
The original sort/dedup output tail behavior is unchanged.

QM-02D qualification passes all 79 package tests in debug and release, strict
all-target/all-feature Clippy, formatting and dependency-free WASM compilation.
A separate frozen-source comparison passes 15 tests per profile: three original
tests, three retained candidate tests, seven author differential groups and two
independent challenges. These include 2,048 changing-configuration/history
transitions, refusal recovery, projection unwind, signed arithmetic, full output
tails, EOS boundaries and positive-controlled allocation measurement. The older
private draft directory labelled `qm02c` is retained provenance, not the packet
identifier.

This is numerical ownership extraction. Qualia metadata/serde adapters, PLE
table I/O, embedding gathers, residual/convolution execution, full Qwen/MoE
sequences and governed state commit remain separate migration gates.
