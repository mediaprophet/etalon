# Retained model byte ownership

`etalon-model-io` is a separate std library for retaining immutable model bytes
and identifying an ordered bundle of those bytes. Portable parsing remains in
`etalon-model`, whose default dependency-free/no_std and unsafe-forbidden
configuration is unchanged. The I/O library has no dependency on Qualia,
Webizen, a model catalogue or a GPU provider.

`lease::ModelLease::open` accepts a path, an independently selected full-file
SHA-256, explicit limits and a cancellation flag. On Windows it opens a retained
read handle without write/delete sharing, validates local disk NTFS from that
opened handle, admits the byte length, maps the same object and hashes every
byte. An existing incompatible writer or writable mapping causes refusal.
The mapping is dropped before the exclusion handle. `bytes()` borrows the
lease; neither the file handle nor an independently owned mmap can escape.
An `Arc<ModelLease>` retains the complete lifetime guarantee.

The guarantee concerns the retained file object, not stable pathnames: another
hardlink name may be removed or replaced without changing the leased bytes.
Consumers must retain and identify the actual lease, rather than reopening a
path or accepting a matching filename. Privileged raw-volume writes, malicious
kernel components and hardware faults are outside this ordinary file-sharing
guarantee.

## Explicit limits and unsupported backing

`LeaseLimits` separately caps source length, mapped address range, and their
checked sum. This is admission accounting, not a measured physical working-set
or commit limit. Full hashing touches the mapped pages. Cold path/OS bookkeeping
is separate; there is no payload-sized scratch copy. Path encoding is bounded.
Cancellation is checked before opening/mapping and between hashing chunks; it
cannot interrupt a blocking OS operation or page fault synchronously.

The opened-handle support check rejects remote or redirected devices, unknown
filesystems and ambiguous query results. Other platforms return
`UnsupportedPlatform`. There is no implicit snapshot, remote fallback or
platform emulation. This additive API does not remove any existing consumer's
remote/shared model support; those consumers need an independently qualified
snapshot or equivalent before adopting this guarantee.

## Explicit verified snapshots

`snapshot::Snapshot::copy_from` provides an opt-in copy route for a caller-owned
`Read` source. The caller supplies a trusted expected full SHA-256 and byte
length, an existing trusted local parent, limits, and a cancellation flag.
The implementation creates a unique owned temporary directory, checks the
actual output handle's local NTFS backing before payload writes, and copies
through a fixed 64 KiB buffer. It rejects short or long sources, mismatched
digests, exceeded operation budgets and cancellation. After closing the writer,
it opens the retained `ModelLease` and verifies the complete bytes again.

The snapshot retains both the lease and temporary directory. Its bytes borrow
the snapshot; the public API exposes no raw mapping, file handle, path or
independently owned lease. The lease is released before cleanup. Explicit
`close` returns `CleanupPending` when removal fails, and that owner can retry
after the interfering condition ends. Constructor errors can also carry a
cleanup obligation through `take_cleanup`. Drop and unwind use best-effort
cleanup: OS refusal or process abort cannot guarantee removal. The caller must
maintain a trusted, cooperatively managed parent namespace throughout cleanup.

For a payload of length `L`, admission accounts for `L + 1` source bytes
(including the final EOF probe), `L` copied payload bytes, `L` mapped bytes and
the 64 KiB copy buffer. Separate read/write call limits include interrupted and
partial operations. These are byte and attempt budgets, not RSS, filesystem
metadata, reader allocations or syscall latency limits. Cancellation is checked
between operations; it cannot interrupt an arbitrary blocking reader.

A verified snapshot identifies the bytes that were copied. It does not establish
remote writer exclusion, a coherent remote source version, trusted provenance
or permission. No network protocol is implemented or tested by accepting
`Read`. Direct lease failure never silently selects this route, and existing
Qualia loaders have not yet adopted it.

## Complete bundle identity and host responsibilities

`bundle_digest` binds ordered `ModelPart` entries, including role, format
version, ordinal, byte length and full content digest. Base model, shards,
tokenizer, configuration, helper metadata, LoRA and draft model remain distinct
roles. Explicit part/source byte budgets and canonical ordering are checked.

The caller must enumerate every file its prepared model actually consumes and
retain every lease. This function does not discover omitted configuration or
tokenizer files, authorize the expected hashes, or bind transformed parameters
and uploaded device state automatically. A matching caller-supplied digest is
content equality, not provenance or permission. Normal Webizen admission,
current actor associations and accepted output remain host responsibilities.

The library contains no numerical engine, token acceptance, rollback mechanism
or model activation. Existing Qualia loaders and provider pins are unchanged by
its addition.

## Package checks

```sh
cargo test --locked -p etalon-model-io
cargo test --locked -p etalon-model-io --release
cargo clippy --locked -p etalon-model-io --all-targets -- -D warnings
cargo check --locked -p etalon-model-io --lib --target wasm32-unknown-unknown
```

Windows tests exercise real writer and writable-mapping exclusion, digest and
budget refusal, cancellation, retained aliases and Arc lifetimes, bundle
identity, and cleanup on failure/unwind. Separate compile probes check private
mapping/handle access and borrowed-byte lifetime. A non-Windows library compile
does not establish an executable backing guarantee on that platform.

The snapshot extension passes 49 tests in each debug/release profile: 26 unit
tests, ten existing external lease tests and thirteen independent snapshot
tests. The latter include real OS deletion refusal during explicit close and
constructor failure, retained cleanup/retry, interrupted and partial reads,
byte/call budgets, cancellation, unwind and retained Arc ownership. Positive
allocation controls cover borrowed access after construction. Actual-library
compile probes reject escaping borrowed bytes, accessing private lease/directory
fields, and consuming the owner through a shared Arc. Strict all-target Clippy,
formatting and WASM library compilation pass; WASM backing execution is not
claimed. One initial verifier fixture incorrectly expected a particular
Windows error kind; the corrected fixture compares the actual OS refusal while
preserving the refusal/retry assertions. Production code was unchanged.
