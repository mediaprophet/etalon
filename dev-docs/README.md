# Etalon Engineering Documentation & Architectural Blueprints

This directory contains the authoritative implementation plans, migration inventories, and governance architectures for the **Etalon** sovereign GPU engine.

---

## Document Index

| ID     | Title & Focus                                                                                                                                                                               | Status                  | File Link                                                                                                  |
| :----- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------- | :--------------------------------------------------------------------------------------------------------- |
| **00** | **Strategic Foundations & Hardware Agent Vision**<br>Core roadmap for silicon stewardship, pointer-native memory, cooperative matrices, and multi-queue architecture.                       | _Draft Baseline_        | [`initial-doc.md`](initial-doc.md)                                                                         |
| **01** | **Complete Codebase Rebranding (`wgpu` → `etalon`)**<br>Topological crate rename, directory transformation, namespace purification, and backwards-compatibility shim.                       | **Ready for Execution** | [`01-wgpu-to-etalon-refactor-plan.md`](01-wgpu-to-etalon-refactor-plan.md)                                 |
| **02** | **QualiaDB GPU Subsystems Migration Inventory & Roadmap**<br>Exhaustive inventory of QualiaDB's 70+ GPU modules, clean database-vs-silicon boundaries, and phased crate consolidation.      | **Ready for Execution** | [`02-qualiadb-migration-inventory-and-roadmap.md`](02-qualiadb-migration-inventory-and-roadmap.md)         |
| **03** | **Licensing, Fiduciary Rights & Silicon Governance**<br>Dual-stage licensing model (CC BY-NC-ND 4.0 baseline transitioning to MediaProphet Commons), and hardware-enforced licensing gates. | **Approved Baseline**   | [`03-licensing-and-rights-architecture.md`](03-licensing-and-rights-architecture.md)                       |
| **04** | **QualiaDB GPU Full Review & Migration Scope**<br>Comprehensive inventory of QualiaDB GPU, shader, inference, and rendering modules.                                                        | **Approved Baseline**   | [`04-qualiadb-gpu-full-review-and-migration-scope.md`](04-qualiadb-gpu-full-review-and-migration-scope.md) |

---

## Consultant Architectural Reviews

| Reviewer                 | Focus & Key Contributions                                                                                                                                                                                                        | File Link                                                                                                                                                            |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Codex / OpenAI**       | Dependency/version unification, LSD radix sort stability bug proof, Metal capability truthfulness, lease-based memory safety, and vertical-slice migration gates.                                                                | [`consultant-reviews/2026-10-20-codex-cross-platform-migration-strategy-review.md`](consultant-reviews/2026-10-20-codex-cross-platform-migration-strategy-review.md) |
| **Gemini / Antigravity** | Dual-plane `ComputeLane` + `etalon-hal` interop, `16x16x32` INT8 Cooperative Matrix & `dot4I8Packed` `M=1` decode roofline, 48-byte `NQuin` warp-transposed `uvec4` loader, 420-BindGroup collapse, and 8 Fork-Dividend unlocks. | [`consultant-reviews/2026-10-20-gemini4-sota-silicon-architecture-review.md`](consultant-reviews/2026-10-20-gemini4-sota-silicon-architecture-review.md)             |

---

## Consolidated Implementation Runbooks (For Execution Agents)

Start with **`implementation/00-MASTER-IMPLEMENTATION-PLAYBOOK.md`** and execute the runbooks in sequence:

| Runbook         | Scope                                                                                                                                                                                                                       | File Link                                                                                                                                    |
| :-------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **Playbook 00** | **Master Implementation Playbook**<br>Immovable rules, Codex + Gemini architectural synthesis matrix, and verification commands.                                                                                            | [`implementation/00-MASTER-IMPLEMENTATION-PLAYBOOK.md`](implementation/00-MASTER-IMPLEMENTATION-PLAYBOOK.md)                                 |
| **Runbook 01**  | **Critical Bug Fixes & Truthfulness Gates**<br>Step-by-step fixes for GPU LSD radix sort instability, false-positive Metal availability, WASM spin-poll deadlock, DXC temp-file race & cache hashing, and `M=1` WMMA waste. | [`implementation/01-CRITICAL-BUG-FIXES-AND-TRUTHFULNESS-GATES.md`](implementation/01-CRITICAL-BUG-FIXES-AND-TRUTHFULNESS-GATES.md)           |
| **Runbook 02**  | **Phase 1: `etalon-hal`, `naga`, and Dependency Unlocks**<br>`compat/wgpu` `30.0.0` bridge, `CooperativeSize::ThirtyTwo` (`I8`/`U8`/`BF16`), and safe BDA / external host memory feature flags.                             | [`implementation/02-PHASE1-ETALON-HAL-AND-NAGA-SILICON-UNLOCKS.md`](implementation/02-PHASE1-ETALON-HAL-AND-NAGA-SILICON-UNLOCKS.md)         |
| **Runbook 03**  | **Phase 2: `etalon-agent`, `etalon-wgsl-forge`, and `etalon-nquin`**<br>Safe resource lease hierarchy, `ComputeLane`, pure IR compiler, 48B `NQuinWire` warp scan, and 3-pass stable GPU radix sort.                        | [`implementation/03-PHASE2-ETALON-AGENT-FORGE-AND-NQUIN-MIGRATION.md`](implementation/03-PHASE2-ETALON-AGENT-FORGE-AND-NQUIN-MIGRATION.md)   |
| **Runbook 04**  | **Phase 3: `etalon-tensor`, `etalon-render`, and `etalon-wasm`**<br>Unified decode/prefill/verify arena, in-shader GBNF/sieve argmax, POD scene contracts, and dual-transport WASM worker steward.                          | [`implementation/04-PHASE3-ETALON-TENSOR-RENDER-AND-WASM-MIGRATION.md`](implementation/04-PHASE3-ETALON-TENSOR-RENDER-AND-WASM-MIGRATION.md) |
| **Audit 05**    | **Implementation Audit & Remaining Gap Report**<br>Forensic code-vs-spec audit of `etalon` and `QualiaDB` commits, documenting completed items and 9 concrete bugs/stubs/regressions with drop-in fixes.                    | [`implementation/05-IMPLEMENTATION-AUDIT-AND-REMAINING-GAP-REPORT.md`](implementation/05-IMPLEMENTATION-AUDIT-AND-REMAINING-GAP-REPORT.md)   |
| **Tracker**     | **Execution Tracker**<br>Wave-by-wave TODO resolution and Phase 1–3 implementation checklist.                                                                                                                               | [`implementation/EXECUTION-TRACKER.md`](implementation/EXECUTION-TRACKER.md)                                                                 |

---

## Workspace Architecture Summary

```text
etalon/
├── Zone 1: Upstream-Synced Core (Refactored to Etalon)
│   ├── etalon-types/          # 64-bit BDA descriptors, device limits, features
│   ├── etalon-sync/           # Lock rank validation, synchronization primitives
│   ├── etalon-hal/            # Silicon Steward: Vulkan, Metal 3, DX12, GLES, native CUDA
│   ├── etalon-naga-bridge/    # Naga AST to pipeline layout translation
│   ├── etalon-core/           # Multi-queue state machine, SnatchLock removal
│   ├── etalon/                # Public Rust API for graphics and AI compute
│   └── naga/                  # Universal shader translator with Etalon extensions
│
├── Zone 2: Foundational Crates & The WGSL Forge (Migrated from QualiaDB)
│   ├── crates/etalon-agent/   # Direct Silicon Steward over HAL (BDA, mmap zero-copy)
│   ├── crates/etalon-forge/   # Multi-target WGSL/Slang JIT compiler and persistent cache
│   ├── crates/etalon-tensor/  # Local LLM autoregressive decoder & cooperative GEMM
│   ├── crates/etalon-nquin/   # 48-byte Super-Quin GPU graph traversal & radix joins
│   ├── crates/etalon-render/  # PortalGpu 10D manifold volumetric rendering engine
│   └── crates/etalon-math/    # Geometric algebra, wave mechanics & tensor geometry
│
├── Zone 3: Fiduciary Domain Extensions
│   ├── extensions/ext-health/        # Clinical/biosignal kernels, zero-retention VRAM
│   └── extensions/ext-civic-safety/  # Anti-surveillance, k-anonymity & sensor redaction
│
└── compat/
    └── wgpu/                  # Lightweight compatibility shim for legacy callers
```
