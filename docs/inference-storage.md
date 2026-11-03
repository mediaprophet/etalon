# Inference storage mechanics

`etalon-inference::runtime` contains the existing joint memory budget planner and
fixed-capacity KV page/prefix bookkeeping extracted from Qualia. It depends only
on `etalon-inference-kernel`. This initial package does not execute a model,
create a device, authorize a request, choose application memory thresholds, or
restore physical KV/recurrent/sampler contents.

`budget` preserves ModelMemoryProfile, BlockGeometry, MemoryPoolBudget,
RequestReservation and BudgetError. `kv::paged` preserves the page free-list and
reference counts, copy-on-write actions, sequence page tables and identity GPU
table construction. `kv::prefix` preserves opaque prefix keys, fixed-capacity
registries, page-retaining prefix storage and iterative token-prefix radix lookup.

PrefixIdentity remains a repr(C) array of four u64 words. The words are opaque
caller identities; equality is not authentication, authorization or proof of a
current model generation. Graph-based identity derivation and graph query meaning
remain in Qualia. The scheduler and its derived launch/transfer counters are not
part of this package. They do not provide measured hardware evidence.

## Ownership and caller requirements

Constructors size their vectors from caller-provided counts. These cold allocations
are retained; this packet does not add a universal byte ceiling or fallible allocator.
Prepared hot operations reuse existing capacity. Callers must validate construction
sizes against their memory budget before allocating. Public mutable tables and
page-set fields retain their historical valid-input preconditions.

A CopyOnWrite::Copy result describes a required physical copy. Updating metadata
does not perform a GPU/CPU data copy or prove that a denied inference branch has
been restored. The caller performs the copy and coordinates backend completion.
Prefix ownership retains pool references while attached requests survive registry
eviction. Registry replacement order, token counts and release/error semantics
remain unchanged.

The source move changes import paths and module routing only. Narrow inherited
style-lint allowances keep the original algorithms intact, including explicit
rollback counters. Known public-input/size limitations remain subject to separate
reviewed improvements. Actual Qualia forwarding is a separate cutover gate.

## Qualification

```sh
cargo test --locked -p etalon-inference
cargo test --locked -p etalon-inference --release
cargo clippy --locked -p etalon-inference --all-targets -- -D warnings
cargo check --locked -p etalon-inference --target wasm32-unknown-unknown
```

On 3 November 2026, all 26 tests passed in debug and optimized builds (52
executions): 18 preserved tests and eight independent tests. The same eight
fixtures passed against the frozen original Qualia sources. Strict Clippy and
WASM compilation pass. Initial Clippy findings were inherited style issues;
the scoped annotations preserve the original algorithm and API.

Independent coverage includes exact budget errors, allocation order/refcounts,
copy-on-write/exhaustion, install/fork rollback, registry replacement and key
equality, active-request ownership through eviction, radix matching/final-token
withholding/LRU/repointing, tails and positive-controlled hot zero allocation.
The fixtures retain historical partial mutation for invalid public state and
the maximum-epoch radix eviction limitation. Runtime adoption remains separate.
