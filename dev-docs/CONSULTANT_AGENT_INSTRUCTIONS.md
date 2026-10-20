# Mandate & Instructions for Consultant AI Agents & Principal System Architects
## The Standard of Excellence: World-Class Silicon Architecture & SOTA Migration for Etalon

> **Document ID:** `ETALON-AGENT-INST-01`  
> **Status:** RATIFIED SOVEREIGN MANDATE  
> **Author & Licensor:** Timothy Charles Holborn  
> **Repository:** `https://github.com/mediaprophet/etalon.git`  
> **Baseline License:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
> **Target Version:** `0.0.1`  

---

## 1. Executive Mandate: The Pursuit of Technical Supremacy

You are acting as an elite principal systems architect, high-performance computing (HPC) specialist, or silicon compiler engineer reviewing the **Etalon** engine (`c:\github\etalon`) and the ingestion of the high-throughput GPU, tensor, and shader compiler subsystems from **QualiaDB** (`c:\github\qualiaDB`).

### The Core Directive: No Compromises, SOTA Only
**Etalon is not an incremental refactoring project or a pedestrian wrapper.** It is engineered to be a peerless, state-of-the-art (SOTA) silicon runtime—a sovereign substrate executing at the physical limits of contemporary hardware architectures. 

Every recommendation, architectural proposal, and code specification you provide **must embody the most superior, advanced, and mathematically optimal methods known to modern computer engineering**. 

#### Unacceptable:
- ❌ Recommending naive, textbook, or "minimum viable" solutions.
- ❌ Recommending legacy abstractions that introduce CPU serialization, redundant memory barriers, or descriptor table thrashing.
- ❌ Hand-waving with generic platitudes (*"refactor for performance"*, *"implement better caching"*, *"use modern Rust patterns"*).
- ❌ Settling for second-best because it is easier to implement.

#### Mandatory:
- ✅ **Micro-architectural Mastery:** Solutions must directly optimize for hardware instruction pipelines, memory coalescing, warp/subgroup occupancy, L1/L2 cache locality, register pressure, asynchronous transfer engines, and PCIe/NVLink saturation.
- ✅ **Mathematical & Algorithmic Rigor:** Where compute kernels are discussed (GEMM, GEMV, FlashAttention, SDPA, SwiGLU, RMSNorm), you must specify the optimal tiled algorithmic formulation, numerical precision handling (BF16, FP16, INT8/INT4 quantization), and reduction tree structures.
- ✅ **Zero-Cost Abstractions:** High-level ergonomic interfaces in safe Rust must compile down to bare-metal hardware execution with zero runtime penalty and zero steady-state heap allocations.

---

## 2. Mandatory Deep Research Requirement

You are explicitly required to **conduct deep, rigorous research** into the latest hardware specifications, vendor microarchitecture whitepapers, and peer-reviewed academic literature before proposing architectural solutions:

1. **Active Specification Research:**
   - Investigate the latest **Vulkan 1.3 / 1.4 specifications** and extensions (e.g., `VK_KHR_buffer_device_address`, `VK_KHR_cooperative_matrix`, `VK_EXT_descriptor_indexing`, `VK_KHR_timeline_semaphore`).
   - Investigate the cutting-edge **W3C WebGPU specification drafts**, particularly ongoing standardization work around subgroup operations, transient attachments, and native cooperative matrix proposals.
   - Investigate modern **DirectX 12 Agility SDK** features (e.g., Shader Model 6.8+, Work Graphs, Enhanced Barriers) and **Metal 3** capabilities (e.g., Mesh Shaders, SIMD-scoped matrix operations, Indirect Command Buffers).
   - Investigate **CUDA 12.x driver APIs**, PTX ISA updates, and Tensor Memory Accelerator (TMA) concepts.
2. **Academic & Industry Cross-Pollination:**
   - Reference and cross-pollinate breakthrough techniques from state-of-the-art runtimes (e.g., NVIDIA CUTLASS, FlashAttention-2/3, OpenAI Triton, vLLM PagedAttention, and Google DeepMind systems).
   - If you encounter a design dilemma, **research the best known prior art**, cite the relevant academic papers or vendor specifications, and explain how Etalon can implement an equal or superior design.

---

## 3. Directory & File-Naming Standard for Submissions

All consultant agents are required to output their findings into dedicated markdown files within the following directory structure:

```text
c:\github\etalon\dev-docs\consultant-reviews├── .gitkeep
├── TEMPLATE-review.md
└── <YYYY-MM-DD>-<agent-or-provider>-<focus-domain>-review.md
```

### File-Naming Convention
Your review filename **must** strictly adhere to this exact pattern:
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

## 4. The 6-Section Superior Review Schema

Every submitted review document **must** follow this 6-section schema. A starter template is provided at [`dev-docs/consultant-reviews/TEMPLATE-review.md`](consultant-reviews/TEMPLATE-review.md).

```markdown
# Architectural Review: [Focus Domain / Title]
**Reviewer:** [Agent Name / Model Version / Provider / Human Specialist]  
**Date:** [YYYY-MM-DD]  
**Review Target:** Etalon Engine (`v0.0.1`) & QualiaDB Subsystem Ingestion  
**Research & Literature Cited:** [List specifications, whitepapers, or papers investigated]  

---

### Section 1: Executive Verdict & State-of-the-Art Strategic Vision
[Provide an uncompromising, authoritative systems assessment. Declare clear go/no-go verdicts on migration candidates, identify architectural dead-ends, and articulate the sovereign SOTA blueprint.]

### Section 2: Forensic Technical Analysis & Theoretical Foundations
[Deep-dive into the specific domain. Perform an exhaustive side-by-side comparison between QualiaDB source implementations and Etalon target crates. Cite exact files, structs, functions, line ranges, and memory layouts. Detail the theoretical mathematical or microarchitectural principles underpinning your analysis.]

### Section 3: Definitive Solutions to the Core Strategic Questions
[Provide the most superior, advanced, and thoroughly researched solutions to the 4 primary consultation questions detailed in §5 below.]

### Section 4: Risk Matrix, Edge Traps & Microarchitectural Failure Modes
[Forensically examine driver quirks, hardware race conditions, synchronization hazards, 256-byte alignment pitfalls, memory fragmentation under sustained pressure, and thermal throttling mitigations.]

### Section 5: Precision Execution Roadmap & Milestone Acceptance Criteria
[Chronological, phased execution plan. Define unambiguous, automated verification gates (e.g. specific cargo test commands, profiling counters, roofline analysis targets, and latency ceilings).]

### Section 6: Traceable Sign-Off & Attestation
[Model identifier, confidence score (1-10), technical assumptions, and verification signatures.]
```

---

## 5. The 4 Core Strategic Questions Every Consultant Must Solve

Your review must explicitly analyze, resolve, and provide state-of-the-art designs for the four strategic architectural dilemmas identified in [`dev-docs/04-qualiadb-gpu-full-review-and-migration-scope.md`](04-qualiadb-gpu-full-review-and-migration-scope.md):

### Question 1: HAL Integration vs. Dedicated Compute Crate for CUDA / PTX
- **The Dilemma:** Upstream `wgpu-hal` is centered around traditional graphics pipeline render passes, render encoders, and command buffers. QualiaDB's CUDA lane (`crates/qualia-core-db/src/inference/cuda_lane/`) completely bypasses render passes in favor of persistent stream launches, asynchronous memory copies, and graph capture.
- **Your Mandate:** Define the most advanced, zero-overhead architectural approach. Should Etalon integrate CUDA into `etalon-hal` implementing `etalon_hal::Api`, or should CUDA/PTX live in a dedicated `crates/etalon-forge` and `crates/etalon-tensor` compute lane that operates alongside `etalon-hal`? You must evaluate trait impedance match, driver dispatch latency, and compute-queue specialization.

### Question 2: Buffer Device Address (BDA) & Zero-Copy Host Mmap Exposure
- **The Dilemma:** QualiaDB processes 48-byte uniform `NQuin` records and 64-byte P64 weight records. Passing these through traditional WebGPU descriptor sets incurs descriptor re-binding and CPU validation overhead on every dispatch.
- **Your Mandate:** Research and architect the most superior mechanism to expose raw 64-bit GPU virtual addresses (Buffer Device Address / BDA in Vulkan & DX12) in safe Rust. How can Etalon provide zero-copy host mmap page access directly to GPU shader stages while preserving WebGPU memory safety invariants on WASM?

### Question 3: Cooperative Matrix / Tensor Core Standardization
- **The Dilemma:** NVIDIA exposes cooperative matrix on Windows via Vulkan (`VK_KHR_cooperative_matrix`), but NOT via standard DX12 wgpu. Naga's SPIR-V output is currently ~19% slower than Microsoft DXC's SPIR-V.
- **Your Mandate:** Research and define the industry's most advanced abstraction for cooperative matrix operations. How should Etalon unify WGSL compute shaders and native SPIR-V/DXC pipelines so that high-level matrix multiply operations auto-select hardware tensor cores regardless of backend without vendor-specific dialect fragmentation?

### Question 4: WASM Multi-threading & Web Worker Architecture
- **The Dilemma:** Browser WebGPU instances cannot be shared arbitrarily across threads without `SharedArrayBuffer` and `Atomics`. Running an autoregressive LLM decode loop on the main UI thread freezes canvas rendering and input dispatch.
- **Your Mandate:** Formulate the premier multi-threaded architecture for running the token decode loop on a dedicated Web Worker while presenting rendered frames on the main thread canvas. Detail the ring-buffer synchronization protocol, zero-copy buffer handoff, and latency-hiding mechanisms.

---

## 6. Mandatory Grounding & Reference Files

Before authoring your review, you must inspect the following foundational files:

1. **Etalon Architectural Blueprint:** [`dev-docs/initial-doc.md`](initial-doc.md)
2. **Topological Refactor Plan:** [`dev-docs/01-wgpu-to-etalon-refactor-plan.md`](01-wgpu-to-etalon-refactor-plan.md)
3. **QualiaDB Migration Inventory:** [`dev-docs/02-qualiadb-migration-inventory-and-roadmap.md`](02-qualiadb-migration-inventory-and-roadmap.md)
4. **QualiaDB Forensic GPU Audit:** [`dev-docs/04-qualiadb-gpu-full-review-and-migration-scope.md`](04-qualiadb-gpu-full-review-and-migration-scope.md)
5. **Master Technical Debt Index:** [`dev-docs/TODO_INDEX.md`](TODO_INDEX.md)
6. **QualiaDB Production Shaders & Emitters:** `c:\github\qualiaDB\crates\qualia-core-db\src\wgsl_forge\`
7. **QualiaDB Native CUDA Lane:** `c:\github\qualiaDB\crates\qualia-core-db\src\inference\cuda_lane\`
8. **QualiaDB Volumetric Renderer:** `c:\github\qualiaDB\crates\qualia-core-db\src\render\gpu\`
