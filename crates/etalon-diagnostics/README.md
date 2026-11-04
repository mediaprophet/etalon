# Native diagnostics

This cold tooling crate owns the cross-device GEMV probe and bounded native worker
mechanisms. Its default feature set has no GPU dependency. `gpu` enables the
adapter/backend-specific GEMV and upload probe. Qualia owns discovery preference,
ranking, signed passports and worker executable/environment selection.

The historical workload is preserved: one GEMV warmup batch, five measured batches
of sixteen dispatches, a 16 MiB upload with one warmup and three measured uploads,
and the rayon CPU reference with one warmup and five measured calls. GPU numbers
are submission/completion wall-clock measurements, including host overhead;
they are not device timestamp measurements. Timestamp conversion is provided
separately and rejects invalid periods and reversed or overflowing tick intervals.

Shape admission checks arithmetic before allocation and caps the probe's modeled
host, staging and device-buffer storage at 256 MiB. This is a cold allocation
class; driver-internal allocations are not a zero-heap guarantee. Device buffer
limits are checked independently. IPC response production and consumption are
capped at 1 MiB plus the eight-byte frame header. CBOR nesting is limited to 32.
Each device response lives in one randomly named, RAII-owned `TempDir`; no retained
artifacts or broad cleanup operation is provided.

Worker stdout/stderr have separate caller-selected byte budgets. One thread owns
the child and both pipes. Windows readiness uses `PeekNamedPipe`; Unix readiness
uses `poll`. Each polling turn reads a bounded number of chunks. No output reader
threads are created. Cancellation and the deadline also cover output drain, so
inherited open write handles cannot block the caller indefinitely. The immediate
worker is killed and reaped on failure, with a one-second cleanup grace per attempt
and a bounded RAII retry. OS kill/reap failures are returned or reported during
unwind. Descendant process trees are not terminated; this API does not provide an
OS process sandbox. Benchmark workers must not launch background descendants.

This extraction has not yet passed EM-08 qualification. Required deferred checks
include malformed and oversized IPC, output overflow, crash, hang, cancellation,
inherited-pipe drain, driver failures, and temporary-directory cleanup on success,
error and unwind. Native CPU-only, GPU, lab and actual Qualia worker-entry builds,
existing fixtures and independently scheduled hardware measurements remain gates.
