# Standalone dense CPU inference development

The optional `etalon-inference/gguf-cpu` feature supplies borrowed GGUF bindings
and CPU callbacks for the extracted dense numerical engine. It has no QualiaDB
dependency. The `dense_gguf_developer` example requires an explicit
`--mode standalone-developer` selection; missing or different modes are errors.

```text
cargo run --release -p etalon-inference --features gguf-cpu --example dense_gguf_developer -- --mode standalone-developer MODEL.gguf TOKEN_IDS_CSV CONTEXT STEPS BYTE_BUDGET
```

Input is a nonempty comma-separated list of already-tokenized unsigned token IDs.
The runner consumes the full prompt, generates the requested bounded number of
greedy tokens, and prints token IDs, logit bits and full numerical-state SHA-256
digests after each prefill/decode step and KV reset. It does not tokenize text or
claim chat-template support. `CONTEXT` and `STEPS` are each bounded by 4096; the
prompt plus generated positions must fit the selected context. Prompt CSV input
is bounded to 65,536 bytes. Model reading refuses file growth and does not create
a second model-sized allocation.

`BYTE_BUDGET` bounds the model, prompt, binding, numerical workspace and undo
buffer capacities, retained argument strings and 40,960-byte numerical kernel
stack. It is not a process RSS limit: executable pages, allocator bookkeeping,
standard I/O, OS argument parsing and small parser/hash stack frames are separate.
The runner writes results to standard output and creates no model variants or
temporary artifacts. A caller retaining output owns its artifact budget.

The initial supported numerical profile is dense Llama GGUF with adjacent-pair
RoPE, fixed RMS epsilon `1e-5`, full even head rotary width and standard nine
weight roles per layer. Missing metadata, duplicate relevant keys/tensors,
unsupported numerical metadata or tensor roles, mismatched shapes, truncated
payloads, and misaligned Q6_K payloads are errors. Unknown `llama.*` settings are
refused. Context must not exceed the declared training context when present.
Output weights may use the embedding fallback; output normalization is optional,
matching the existing numerical path. Rope defaults are base `100000` and scale
`1`. This additive profile does not retire other existing model architectures.

Preparation writes into a caller-owned layer slice and charges its full byte
length. `BindingLimits::max_tensor_visits` bounds repeated descriptor scans.
Every binding is validated before any caller layer is written. Model bytes are
borrowed immutably, and private packed handles prevent fabrication of offsets.
The native callback uses the original scalar row-decode and multiply/sum order.
It does not replace or qualify the existing WASM Q8 SIMD implementation.

Numerical transactions snapshot the workspace before tentative execution and
restore it on abort, drop or execution error. Explicit commit retains the
tentative changes. This mode deliberately does not obtain Sentinel
decisions. Normal applications must separately authenticate their acting subject
and obtain current Webizen decisions before accepting or publishing tokens.
Turning on profiling or diagnostics in a governed session does not select this
standalone mode. An unavailable controller must not trigger a mode fallback.

Qualification includes 44 package tests in each of debug and optimized profiles
(the six independent groups were also run separately), strict all-target Clippy,
optional-feature WASM library compilation and a default dependency-tree check.
The default inference library still depends only on the numerical kernel;
`sha2` is an example/test dependency, and model codecs are feature-selected.

The independent boundary tests exercise all thirteen retained row types against
original native GEMV bit expectations, unsupported metadata, duplicate keys,
invalid shapes, truncated/misaligned payloads, complete caller-storage budgets,
output tails and positive-controlled zero-allocation success/refusal paths.
Six CLI negative cases refuse missing/wrong mode or invalid prompt/context/budget.

The retained SmolLM2-360M-Instruct-Q8_0 GGUF (SHA-256
`c004ac34cce45e03e4452bca472535bc371fcb0d07ae560bba2c57ce4a284b7b`)
consumed all 19 supplied prompt IDs and generated 16 tokens at context 128.
All 35 complete workspace digests, each covering 2,683,840 float bit patterns,
match the original implementation trace: 18 prefill states, 16 decode states
and KV reset. Every generated token and maximum-logit bit pattern also matches.
The runner charged 407,935,658 bytes under the specified 1 GiB buffer budget.
Different argument-path lengths change the charged argument bytes. This is
numerical equivalence evidence, not a measured speed improvement or full-model
semantic conformance claim.

Normal runtime integration and hardware-specific GPU/NPU execution are separate
qualification scopes. A successful WASM compile does not qualify browser SIMD
execution. This packet makes no change to normal application acceptance policy.
