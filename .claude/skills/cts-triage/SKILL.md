---
name: cts-triage
description: Explain retired CTS tooling and locate current integration coverage
---

# Retired CTS tooling

The embedded Deno CTS runner and `cargo xtask cts` have been removed.
Do not invoke the old commands or claim the QualiaDB/VibeScript harness runs CTS.

Read `docs/conformance-coverage.md` for the coverage loss, historical revision
and selectors, retained Rust suites, and the consuming QualiaDB harness commands.
Use the WebGPU/WGSL specifications to establish correct behavior when investigating
implementation defects. A separate CTS bridge must actually exercise this Etalon
implementation before upstream conformance coverage can be claimed again.
