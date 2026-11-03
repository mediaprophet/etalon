# Tentative dense numerical execution

`etalon-inference::runtime::dense_transaction::propose_dense` executes one dense
CPU proposal while retaining an undo image in caller-owned storage. It returns
a non-cloneable `PendingDense`, which keeps exclusive access to all thirteen
numerical buffers until its owner resolves the proposal.

The owner may inspect the proposed token, score and read-only buffers. Inspection
is not acceptance. Consuming `commit` retains numerical state and returns the
step; consuming `abort`, ordinary drop, callback failure or Rust unwinding restores
every complete buffer, including spare tails, NaN payloads and signed zeros.
Ownership prevents re-entry or duplicate resolution through the safe API.

Admission checks the token position, all thirteen required capacities, checked
aggregate sizes, undo capacity and total byte budget before any snapshot write
or model callback. The budget charges the full supplied workspace slice lengths
and the full supplied undo slice length. It does not account for the immutable
model, callback-private state, host state or allocations outside those slices.
Only the required snapshot prefix is written; spare undo tail values are preserved.
Undo storage retains its captured values after resolution for its owner to manage.

This first implementation copies the full workspace to undo and copies it back
on abort. That bounded copy cost is intentional; no throughput improvement is
claimed. Optimizing it to selected KV positions requires separate state-equivalence
evidence. It does not implement GPU, SSM, recurrent or physical paged-KV restoration.

Callbacks must provide stateless/read-only model access, or their state must be
enrolled separately in the host transaction. This primitive cannot restore a
callback's private mutations. It also does not restore host token position,
session generation, sampler history/RNG, grammar, stop/thinking state, cache
publication or application output. Those belong to the integrated acceptance
transaction. Rust process abort or deliberately forgetting a guard does not run
Drop; this is ordinary Rust resource ownership, not hostile-caller authorization.

## Webizen and developer modes

Normal Qualia execution must keep the pending proposal inside a governed adapter,
obtain the actual Webizen/Sentinel decision bound to that proposal, then commit
numerical and host state before publishing accepted output exactly once. Normal
callers must not receive an ungated commit operation. Missing, expired, rejected
or lost control cannot mean acceptance. A numerical snapshot is not a policy grant.

Explicit developer/debug/lab sessions may use the numerical API independently,
without Qualia permissions or Sentinel overhead. Enabling debug instrumentation
on a governed session does not select that developer mode. Numerical parity and
restoration tests do not qualify actual Webizen token acceptance.

## Qualification

All 36 package tests pass in debug and release, including six independent
transaction groups. They exercise all thirteen short workspace refusals, actual
two-layer execution with 21 callback failures and 21 callback unwinds, numerical
algorithm failures, full-capacity budgeting, undo tails and exact clean-prefix
retry. Positive-controlled measurements report zero hot allocations. Five negative
Rust ownership probes verify consumed commit/abort and observation/state borrowing;
a positive control verifies the probe harness. Strict all-target Clippy and WASM
compilation pass. WASM compilation is not browser execution evidence.

```text
cargo +1.96.1 test --locked -p etalon-inference
cargo +1.96.1 test --locked -p etalon-inference --release
cargo +1.96.1 clippy --locked -p etalon-inference --all-targets -- -D warnings
cargo +1.96.1 check --locked -p etalon-inference --target wasm32-unknown-unknown
```
