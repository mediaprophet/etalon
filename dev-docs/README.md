# Etalon Engineering Documentation & Architectural Blueprints

This directory contains the authoritative implementation plans, migration inventories, and governance architectures for the **Etalon** sovereign GPU engine.

---

## Document Index

| ID | Title & Focus | Status | File Link |
| :--- | :--- | :--- | :--- |
| **00** | **Strategic Foundations & Hardware Agent Vision**<br>Core roadmap for silicon stewardship, pointer-native memory, cooperative matrices, and multi-queue architecture. | *Draft Baseline* | [`initial-doc.md`](initial-doc.md) |
| **01** | **Complete Codebase Rebranding (`wgpu` → `etalon`)**<br>Topological crate rename, directory transformation, namespace purification, and backwards-compatibility shim. | **Ready for Execution** | [`01-wgpu-to-etalon-refactor-plan.md`](01-wgpu-to-etalon-refactor-plan.md) |
| **02** | **QualiaDB GPU Subsystems Migration Inventory & Roadmap**<br>Exhaustive inventory of QualiaDB's 70+ GPU modules, clean database-vs-silicon boundaries, and phased crate consolidation. | **Ready for Execution** | [`02-qualiadb-migration-inventory-and-roadmap.md`](02-qualiadb-migration-inventory-and-roadmap.md) |
| **03** | **Licensing, Fiduciary Rights & Silicon Governance**<br>Dual-stage licensing model (CC BY-NC-ND 4.0 baseline transitioning to MediaProphet Commons), and hardware-enforced licensing gates. | **Approved Baseline** | [`03-licensing-and-rights-architecture.md`](03-licensing-and-rights-architecture.md) |

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
