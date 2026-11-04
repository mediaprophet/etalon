Independent full-attention fixture provenance
=============================================

Primary reference: Hugging Face Transformers commit
`f2754311602414ed561cdf8ed9d852390f514404`, blob
`21b33fc8216db19b42e9423834cee73ce344c890`, file
`src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`.
Saved source SHA256:
`fdf9e801902e2f25873c0d0d182ebbf3a03988e4386398072c04ed248a572aed`.
The saved original is under the canonical worktree's
`target/inference-migration-qualification/qwen-pinned-reference-20261104`.

The generator transcribes the scalar equations independently using Python
stdlib float64, with inputs rounded to IEEE F32. It imports no candidate Rust.
Regenerate from the worktree root with:

```powershell
py -3 crates/etalon-inference/tests/support/generate_qwen_attention_reference.py crates/etalon-inference/tests/support
```

The script requires an existing explicit destination and caps output at 64 KiB.
Generated values were frozen before candidate Rust execution. The comparison
tolerance was fixed before execution: `abs(actual-expected) <= 2e-6 +
2e-5*abs(expected)`. This admits short F32 reductions and transcendental
rounding against float64; it does not claim bitwise HF framework equivalence.

SHA256 receipts (UTF-8 LF bytes):

| Artifact | SHA256 |
|---|---|
| generator | `27ee652dfa6457e278c2c98c7b0f467c8c79db696cd2de4c7f6b6b71708dd097` |
| qwen_attention_fixture.rs | `fb26d16b3157535842f19584502098c950def64c09dc3f9dfaad38c79672ef53` |
| qwen_attention_reference.json | `ff8129d209d56421796840ed715afa8d7ac4a6e1d829ac25a210ddc8ed340949` |

Three tokens use two query heads sharing one KV head, dimension six, with
four rotary lanes. The first four lanes rotate as `(0,2)` and `(1,3)`;
the last two pass through. An independently generated adjacent-pair alternative
differs by more than 0.1 at position one. Nonzero raw zero-centered Q/K norm
weights, per-query-head `[query6, gate6]` projection rows, sigmoid gate,
nontrivial output projection and residual are exercised. Explicit folded norm
weights test the separate multiplier convention. This synthetic fixture does
not qualify checkpoint packing, multimodal RoPE, GPU performance or a full model.
