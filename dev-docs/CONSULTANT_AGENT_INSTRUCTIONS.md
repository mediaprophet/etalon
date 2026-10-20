# Instructions for Consultant AI Agents & External Reviewers
## Comprehensive Implementation & Migration Protocol for Etalon

> **Document ID:** `ETALON-AGENT-INST-01`  
> **Status:** ACTIVE MANDATE  
> **Author & Licensor:** Timothy Charles Holborn  
> **Repository:** `https://github.com/mediaprophet/etalon.git`  
> **Baseline License:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
> **Version Target:** `0.0.1`  

---

## 1. Context & Review Mandate

You are acting as an expert systems architect, graphics engineer, or silicon optimization consultant reviewing the **Etalon** codebase (`c:\github\etalon`) and its ingestion of the GPU, tensor, and shader compiler subsystems from **QualiaDB** (`c:\github\qualiaDB`).

### Core Objective
You are tasked with providing **rigorous, comprehensive, and forensically grounded architectural recommendations** before execution of the major migration plan commences. Your review must evaluate hardware feasibility, driver corner cases, zero-copy memory layouts, and API boundaries.

### Review Ground Rules (Non-Negotiable)
1. **Concrete Technical Citations Only:** General advice (e.g., *"improve caching"*, *"use modern Rust idioms"*, or *"optimize algorithms"*) is strictly unacceptable. Every recommendation must cite specific files, line ranges, struct layouts, bitmasks, or hardware registers.
2. **Zero-Heap Hot-Path Invariant:** Any design for token decoding loops, render passes, or graph scans must maintain zero heap allocations (`Vec`, `String`, `Box`) in steady-state execution.
3. **Licensing & Attribution Integrity:** All proposed changes must preserve Timothy Charles Holborn's copyright notices and CC BY-NC-ND 4.0 baseline licensing.
4. **Hardware Reality:** Account for real desktop and browser driver behaviors (e.g. NVIDIA A2000, Apple Metal 3, Intel iGPU, Chrome/Firefox WebGPU on Windows, and mobile Safari).

---

## 2. Directory & File-Naming Standard for Reviews

All consultant agents are required to output their findings into dedicated markdown files within the following directory structure:

```text
c:\github\etalon\dev-docs\consultant-reviews├── .gitkeep
├── TEMPLATE-review.md
└── <YYYY-MM-DD>-<agent-or-provider>-<focus-domain>-review.md
```

### File-Naming Convention
Your review filename **must** strictly follow this pattern:
```text
<YYYY-MM-DD>-<agent-or-provider>-<focus-domain>-review.md
```

#### Approved Pattern Examples:
- `2026-10-20-claude37-hal-cuda-integration-review.md`
- `2026-10-20-gpt5-bda-zero-copy-memory-review.md`
- `2026-10-20-gemini25-wasm-multithreading-review.md`
- `2026-10-20-codex-coopmat-tensor-standardization-review.md`
- `2026-10-20-deepseek-volumetric-render-pipeline-review.md`

---

## 3. Required Report Schema (The 6-Section Template)

Every submitted review document **must** follow this 6-section schema. A starter template is provided at [`dev-docs/consultant-reviews/TEMPLATE-review.md`](consultant-reviews/TEMPLATE-review.md).

```markdown
# Architectural Review: [Focus Domain / Title]
**Reviewer:** [Agent Name / Model Version / Provider]  
**Date:** [YYYY-MM-DD]  
**Review Target:** Etalon Engine (`v0.0.1`) & QualiaDB Subsystem Ingestion  

---

### Section 1: Executive Verdict & Strategic Assessment
[High-level architectural assessment: clear go/no-go verdicts, primary architectural recommendations, and fundamental design decisions.]

### Section 2: Domain-Specific Technical Analysis
[Deep-dive into the specific domain. Compare QualiaDB source implementations with Etalon target crates. Cite exact files, structs, functions, and line ranges in both codebases.]

### Section 3: Concrete Answers to Core Strategic Questions
[Direct, definitive responses to the 4 primary consultation questions detailed in §4 below.]

### Section 4: Risk Matrix, Edge Cases & Hardware Traps
[Identify driver quirks, race conditions, memory alignment hazards (e.g., 256-byte row padding, mappedAtCreation traps), and hardware throttling behaviors.]

### Section 5: Step-by-Step Implementation & Milestone Roadmap
[Chronological, phased execution plan with unambiguous milestone acceptance criteria and verification commands.]

### Section 6: Traceable Sign-Off
[Model identifier, confidence score (1-10), key assumptions, and dependencies.]
```

---

## 4. The 4 Core Questions Every Consultant Must Address

Your review must explicitly analyze and answer the four strategic architectural questions established in [`dev-docs/04-qualiadb-gpu-full-review-and-migration-scope.md`](04-qualiadb-gpu-full-review-and-migration-scope.md):

### Question 1: HAL Integration vs. Dedicated Compute Crate for CUDA / PTX
- **The Dilemma:** Upstream `wgpu-hal` is built around graphics pipeline render-pass encoders and command buffers. QualiaDB's CUDA lane (`crates/qualia-core-db/src/inference/cuda_lane/`) bypasses render passes entirely in favor of persistent stream launches, asynchronous memory copies, and graph capture.
- **Your Mandate:** Should Etalon integrate CUDA into `etalon-hal` as an `etalon_hal::Api` implementation, or should CUDA/PTX live in a dedicated `crates/etalon-forge` and `crates/etalon-tensor` compute lane that operates alongside `etalon-hal`? Detail the trade-offs regarding trait impedance match, driver overhead, and maintenance cost.

### Question 2: Buffer Device Address (BDA) & Zero-Copy Host Mmap Exposure
- **The Dilemma:** QualiaDB uses 48-byte uniform `NQuin` records and 64-byte P64 weight records. Passing these through traditional WebGPU bind groups incurs descriptor and CPU validation overhead on every dispatch.
- **Your Mandate:** How should Etalon expose raw 64-bit GPU virtual addresses (Buffer Device Address / BDA in Vulkan & DX12) in safe Rust? Can this be exposed via a new `BufferAddress` capability without breaking WebGPU safety invariants on WASM?

### Question 3: Cooperative Matrix / Tensor Core Standardization
- **The Dilemma:** NVIDIA exposes cooperative matrix on Windows via Vulkan (`VK_KHR_cooperative_matrix`), but NOT via DX12 wgpu. Naga's SPIR-V output is currently ~19% slower than Microsoft DXC's SPIR-V.
- **Your Mandate:** What is the optimal abstraction in `etalon` for cooperative matrix operations? How should Etalon unify WGSL compute shaders and native SPIR-V/DXC pipelines so that high-level matrix multiply operations auto-select hardware tensor cores regardless of backend?

### Question 4: WASM Multi-threading & Web Worker Architecture
- **The Dilemma:** Browser WebGPU instances cannot be shared arbitrarily across threads without `SharedArrayBuffer` and `Atomics`. Running an autoregressive LLM decode loop on the main UI thread freezes canvas rendering and input dispatch.
- **Your Mandate:** What is the recommended multi-threaded architecture for running the token decode loop on a dedicated Web Worker while presenting rendered frames on the main thread canvas? Provide the memory sharing and message passing protocol.

---

## 5. Reference Material to Read Before Reviewing

Before authoring your review, you must inspect the following foundational files:

1. **Etalon Architectural Blueprint:** [`dev-docs/initial-doc.md`](initial-doc.md)
2. **Topological Refactor Plan:** [`dev-docs/01-wgpu-to-etalon-refactor-plan.md`](01-wgpu-to-etalon-refactor-plan.md)
3. **QualiaDB Migration Inventory:** [`dev-docs/02-qualiadb-migration-inventory-and-roadmap.md`](02-qualiadb-migration-inventory-and-roadmap.md)
4. **QualiaDB Forensic GPU Audit:** [`dev-docs/04-qualiadb-gpu-full-review-and-migration-scope.md`](04-qualiadb-gpu-full-review-and-migration-scope.md)
5. **Master Technical Debt Index:** [`dev-docs/TODO_INDEX.md`](TODO_INDEX.md)
6. **QualiaDB Production Shaders & Emitters:** `c:\github\qualiaDB\crates\qualia-core-db\src\wgsl_forge\`
7. **QualiaDB Native CUDA Lane:** `c:\github\qualiaDB\crates\qualia-core-db\src\inference\cuda_lane\`
8. **QualiaDB Volumetric Renderer:** `c:\github\qualiaDB\crates\qualia-core-db\src\render\gpu\`
