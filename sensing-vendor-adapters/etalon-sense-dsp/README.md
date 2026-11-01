# Explicit DSP routing

This portable no-std library routes existing Etalon live-processing operations
to an exact host-registered DSP target. It uses the existing operation revision,
encoding/layout, workspace, lifecycle and allocation admission contracts.
No alternate target or CPU fallback is attempted. A target can register several
distinct operation IDs/revisions; duplicate target-operation pairs are refused.
At most 16 caller-owned
bindings are examined; there is no registry allocation or runtime discovery.

Each SDK provider implements `LiveProcessingProvider`, declares `Dsp`, and
must actually execute on the bound native target. Register it with host-defined
vendor/model/device IDs through `DspBinding`. IDs and provider declarations are
not hardware attestation. The binding freezes the full descriptor and rejects
restarts or capability changes until the caller creates a fresh binding. A
forwarding adapter exposes only that frozen descriptor to the core dispatcher,
checks the exact grant and actual provider before forwarding, then checks
again after completion. The SDK must still atomically validate its grant and
hold its native lifecycle lease throughout dispatch; repeated checks alone
cannot establish that atomicity.

Providers must accurately declare their allocation class. An ORT/vendor path
with unbounded internal allocation must use `ColdUnbounded` and cannot satisfy
zero-heap or bounded admission. Synchronous SDK calls must return and must not
unwind; cancellation, host policy, native isolation and physical qualification
remain provider/deployment responsibilities.

The separate native Hexagon provider is an example concrete implementation for
an actual DSP target. Other SDK providers can register different operations,
precisions, layouts and devices without changing this router. Host execution of
ordinary CPU DSP routines must not be registered as physical DSP execution.

CI host tests use synthetic providers to exercise routing and refusal. They
do not qualify a native DSP, runtime or remote execution.
