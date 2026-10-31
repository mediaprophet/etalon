# Standalone sensing contracts: first implementation slice

This directory stages dependency-free, allocation-free Rust contracts above the
Etalon execution APIs. It does not acquire sensors, grant authority or execute
accelerators. No stable serialized ABI is declared by native Rust type layout.

This package is registered through the existing `crates/*` workspace glob in
the isolated sensing worktree. The modernization chat approved isolated
registration; the primary checkout, lockfile, accelerator APIs, compiler/HAL and
qualification infrastructure remain under its coordination.

The implementation uses `no_std`, no dependencies and forbids unsafe code.
Builds and tests are deferred until the ongoing cleanup is finished. The
sources have not been compiled or validated yet.

The implementation plan remains at
`C:/github/qualiaDB/docs/plans/unified-sensing-implementation-2026-10-30.md`.
Existing plan documents are preserved unchanged in the primary QualiaDB tree.
