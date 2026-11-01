# Etalon sensing contracts

Portable, dependency-free Rust contracts for caller-owned sensing buffers,
observation envelopes, uncertainty, calibration, clock mapping and policy
composition. This library uses `no_std` and forbids unsafe code.

The contracts do not acquire devices, grant authority or attest a source.
Applications supply physical meaning, trusted clocks, calibration references and
policy authorities. Native Rust type layout is not a stable serialized ABI.

Use this crate directly or through the `etalon-sense` facade. Packages are
available through Git or a local sibling checkout; they are not published to
crates.io. Pin a tested commit for reproducible consumers.

- [Library guide](https://mediaprophet.github.io/etalon/getting-started.html)
- [Contract API](https://mediaprophet.github.io/etalon/doc/etalon_sense_core/)
- [Validation results](https://github.com/mediaprophet/etalon/actions/workflows/ci.yml)

Software validation and hardware/scientific qualification are separate.
Unknown source, quality, time and uncertainty stay explicit.
