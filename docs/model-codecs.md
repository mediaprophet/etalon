# Portable model row codecs

`etalon-model` exposes GGML row decoding and Q4 conversion through its optional
`ggml` feature. The default container/metadata library remains dependency-free
and `no_std`. Enabling `ggml` requires `std` on native and WASM targets so the
original floating-point operations, including `powi` and `round`, remain intact.
Its optional dependencies are the same half 2.7.1 and bytemuck 1.25.2 versions
used by the source consumer. No GPU provider or Qualia dependency is introduced.

The row decoder accepts F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K,
IQ4_NL, IQ4_XS, FP8 E4M3 and the existing Q4_K_SOA extension. Type identifiers,
block strides, scale/min encodings and arithmetic order are preserved. Stock
Q4_K uses 144 bytes per 256 weights; Q4_K_SOA uses 160. They are distinct layouts.
Conversion APIs preserve the original 144-to-160 rewrite and f32 quantization.

All outputs are caller-buffered. Model indexing, tensor slicing and ownership,
tokenization, semantic policy, and matrix-vector execution are separate adapters.
Type 158 has no row decoder in this library; its existing container accounting
and separate execution path are not replaced by another dtype.

The extraction preserves existing low-level preconditions and refusal behavior.
In particular, Q6_K's bytemuck view requires two-byte alignment and can panic for
a misaligned byte slice. Arithmetic over arbitrary near-`usize::MAX` dimensions
is not newly validated by this extraction. Callers must validate model-derived
sizes and bounds before entering these kernels. Truncated/refused conversions
may retain the original partial-output behavior; they are not transactions.

The original FP8 helper maps the two exponent-15/mantissa-7 encodings to positive
zero. This behavior is retained and checked for all 256 byte encodings; the move
does not claim a redesigned IEEE special-value policy. Numerical NaNs, infinities,
signed zero, output tails and conversion bytes belong to compatibility evidence.

The numerical library does not grant permission to accept or publish tokens.
Normal Qualia inference must obtain a real Webizen/Sentinel decision in its token
acceptance loop. Independent execution is for explicitly selected developer,
debug or lab sessions; enabling this feature does not select that execution mode.

## Qualification

Eleven applicable original pure-codec tests remain alongside the container tests.
Original descriptor, ternary and model-dependent tests remain in Qualia until a
separately qualified caller migration. Independent fixtures compare the same
decoder contracts with immutable original source, including non-aligned Q6 input
behavior and positive-controlled allocation measurements.

The default package's 62 tests pass. With `ggml`, 73 package tests plus six
independent contract groups pass in debug and optimized builds. Those six groups
also pass against actual immutable original code: 234 row cases cover thirteen
formats, nine lengths and successful/truncated inputs, with every output bit
checked. All 256 FP8 byte encodings, conversion bytes, error precedence, tails,
nonfinite values and the Q6 alignment behavior are included. Positive-controlled
hot allocation measurements report zero. Strict Clippy and both default and
codec-enabled WASM compilation pass. Original and snapshot source digests differ
only in CRLF/LF representation; their complete normalized content matches.

```text
cargo +1.96.1 test --locked -p etalon-model
cargo +1.96.1 test --locked -p etalon-model --features ggml
cargo +1.96.1 test --locked -p etalon-model --features ggml --release
cargo +1.96.1 clippy --locked -p etalon-model --features ggml --all-targets -- -D warnings
cargo +1.96.1 check --locked -p etalon-model --no-default-features --target wasm32-unknown-unknown
cargo +1.96.1 check --locked -p etalon-model --no-default-features --features ggml --target wasm32-unknown-unknown
```

WASM checks establish compilation, not browser execution or SIMD acceleration.
No model throughput improvement, device execution, governed token acceptance or
consumer cutover follows merely from these codec tests.
