Mixed-model independent oracle
==============================

Reference: Transformers commit f2754311602414ed561cdf8ed9d852390f514404,
modeling_qwen3_5_moe.py, saved source SHA256
fdf9e801902e2f25873c0d0d182ebbf3a03988e4386398072c04ed248a572aed.
This transcribes the original gated-delta recurrence, split-half text attention,
decoder residual/norm ordering, routed and gated shared SwiGLU experts, and final
RMS normalization. Explicit synthetic output projection extends the final hidden
vector to four logits. Inputs represent caller-provided embedding outputs.

Regenerate using `py -3 crates/etalon-inference/tests/support/generate_qwen_model_reference.py crates/etalon-inference/tests/support`.
The generator imports Python stdlib only and executes no candidate code. All
weights/inputs round to F32; subsequent equations use float64. State starts zero.
Explicit destination and a 100,000-byte output cap bound fixture generation.

Frozen before Rust execution: absolute error <= 2e-6 + 2e-5*abs(expected),
allowing short F32 reductions/transcendental rounding against float64. No
tolerance change from the independently qualified single-layer fixtures.

SHA256 of LF UTF-8 artifacts:

- Generator: 4cc27bd52a561b9f033bb37ea314cb8f29003bf62621d89e75f9ef585bb72a50
- Rust: 490c27b88299b6f80acce2a1ec5e8534fafeb3f770a04f8b22bb995560c22aa0
- JSON: 4be418262e062447fdefbefafc9132c60a47b1949caed79a02a7137c103ca00e

Two tokens, hidden3, linear HK1/HV2/K2/V2/kernel2 followed by full attention
Q2/KV1/head6/rotary4, two routed experts selected top2 and a nonzero gated shared
expert at each layer. Every expert has width2. Both norms/residuals and final
projection are nontrivial. Per-layer after-attention and after-FFN hidden,
final normalized hidden, logits, history, recurrent state and full KV capacity
are frozen. Reset/replay follows the same sequential token path used for prefill
and decode; no separate batched-prefill API is claimed. Synthetic fixtures do
not qualify real checkpoint packing, physical GPU execution, full-model quality,
or embedding/governance policy integration.
