# Sensing library integration and validation

Etalon sensing is available from this repository through Git or a local checkout.
It does not require the former sensing worktrees, a QualiaDB database or an
installed native SDK for portable signal processing. Packages are not published
to crates.io. Use the [public guide](https://mediaprophet.github.io/etalon/getting-started.html)
and pin the full source revision associated with successful CI.

## Public entry points

- `etalon-sense` composes portable contracts, runtime coordination and signal
  primitives, with explicit feature selection for providers.
- `etalon-sense-core` represents source, encoding, sample layout, uncertainty,
  calibration, clocks, policy stages and observation envelopes without acquisition.
- `etalon-signal` implements caller-buffered decoding, I/Q power, spectral
  transforms, windows and filters. Numeric output retains its declared meaning;
  an uncalibrated value does not acquire a physical unit automatically.
- CPU, GPU, SDR, NFC and camera libraries can be selected independently. Windows
  Wi-Fi, Bluetooth, cellular, audio, PC/SC, sensors and camera adapters require
  their explicit platform features and host authorization. Camera acquisition
  retains its documented unqualified gates.
- `etalon-sense-gpu::prepared::PreparedIqPower` connects the sensing provider
  contract to a private Vulkan prepared execution owner. It validates layouts,
  checks host policy before and after preparation/execution, clears failed output
  and invalidates an owner after execution failure. It remains `ColdUnbounded`.
- The isolated `sensing-vendor-adapters` workspace supplies DSP routing, Hexagon
  kernel/service, FastRPC host RPC, RTL-SDR receive, OpenVINO NPU and QNN HTP
  model adapters plus explicit deployment admission. Select these packages
  directly; the default facade does not pull native runtimes into applications.

QualiaDB binds these observation envelopes to its field records, Q42 uncertainty,
Tensor10D projection, inference, Sanctuary/guardianship authorities and custody
storage through its own sensing bridge. Etalon does not provide a database or
choose a universal permission ontology. Generic and environmental observations
can use host-defined authorities without inventing a human subject.

## Validation gates

The [CI workflow](https://github.com/mediaprophet/etalon/actions/workflows/ci.yml)
checks the platform matrix, documentation, formatting, spelling and dependency
policy. Native test jobs run the full `cargo xtask test` suite. Separate Windows
and Linux vendor jobs check, lint and test the isolated workspace with its lockfile
and all features; a separate job checks its Rust dependency licenses.

The Linux native job additionally runs the ignored prepared allocator test alone
and then the ignored prepared sensing functional test. Its software Vulkan
results do not certify a physical DSP, NPU, radio or GPU. The allocator instrument
counts Rust allocator activity in its test process, not driver/OS allocations or
arbitrary application policy callbacks. See [prepared compute](prepared-compute.md)
for lifecycle, timeout, quarantine and qualification limits.

The provenance check verifies all 30 retained SDK reference files against their
recorded byte lengths and SHA-256 values. They are immutable reference inputs,
not native binaries or a DSP firmware build. Reference sources retain their own
licenses. The root CC policy is unchanged; in particular, dynamic loading of GPL
librtlsdr is not a blanket license clearance for a combined distributed product.

The [Docs workflow](https://github.com/mediaprophet/etalon/actions/workflows/docs.yml)
builds API references and a bounded Pages artifact, validates local links and
stamps the exact source revision. Only the `dev` branch deploys the development
site. Check the workflow for the exact pinned revision used by your application.

## Deployment evidence still required from a host

Real hardware requires its matching SDK, runtime, driver, firmware/signing chain,
compatible target and authenticated artifact declaration. No proprietary runtime,
firmware or driver binary is bundled by the portable libraries or Pages artifact.
Rust compilation and mock tests do not establish remote DSP execution, actual
accelerator selection, device identity, calibrated uncertainty, sample timing,
allocation bounds or a complete native dependency closure. Runtime admission
also does not grant permission to capture, store, read or export observations.

Keep those deployment facts explicit alongside the software revision and test
results. The current adapters fail closed where a required capability or authority
is unavailable; they do not silently substitute a CPU result for a claimed native
accelerator execution.
