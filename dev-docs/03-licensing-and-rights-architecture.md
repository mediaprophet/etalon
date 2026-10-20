# Etalon Architectural Plan: Licensing, Fiduciary Rights & Silicon Governance

> **Document ID:** `ETALON-PLAN-03`  
> **Status:** APPROVED FOR IMPLEMENTATION  
> **Author:** Timothy Charles Holborn & Antigravity  
> **Date:** October 2026  
> **Licensing Baseline:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

---

## 1. Context & Fiduciary Principles

Standard open-source licenses (MIT, Apache-2.0, BSD) were designed for an era when software was primarily executed by human developers and small tools. In the era of autonomous AI swarms, trillion-parameter model training, and massive corporate aggregation, passive text licenses fail: large commercial actors routinely ingest, strip provenance from, and monetize open-source systems without contributing back or fairly compensating the human creators.

In the **Etalon**, **QualiaDB**, and **Aether** ecosystem, intellectual property governance is not a passive legal footnote—it is an active, architecturally integrated fiduciary safeguard.

---

## 2. The Dual-Stage Licensing Architecture

The ecosystem operates under a deliberate, two-stage licensing progression:

```
┌────────────────────────────────────────────────────────┐
│               Stage 1: Active Baseline                 │
│      CC BY-NC-ND 4.0 (NonCommercial - NoDerivatives)    │
│  - Protects the unfinished ecosystem from extraction   │
│  - Establishes immutable moral rights & authorship     │
│  - Bars unauthorized commercial exploitation           │
└──────────────────────────┬─────────────────────────────┘
                           │ Ecosystem reaches functional maturity
                           ▼
┌────────────────────────────────────────────────────────┐
│              Stage 2: Target Rights Model              │
│       MediaProphet Permissive Commons License          │
│  - Permissive, open commons for individual humans      │
│  - Ontologically defined rights via WebID & DIDs       │
│  - Automated Web Payments ledgers for corporate agents │
└────────────────────────────────────────────────────────┘
```

### Stage 1: Current Active Baseline
All architectures, modifications, refactorings, extensions, and integrations created by **Timothy Charles Holborn** are governed by:

> **Copyright (c) 2026 Timothy Charles Holborn**  
> **[LinkedIn](https://www.linkedin.com/in/ubiquitous/) | [Email](mailto:timothy.holborn@gmail.com)**  
> **License:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
> Full legal text: [`LICENSE`](../LICENSE) and [Creative Commons Legal Code](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

This strict baseline protects the core technology during its formative development, guaranteeing that proprietary platforms cannot absorb the compute engine without authorization.

### Stage 2: Target Model — The MediaProphet Permissive Commons
Once the ecosystem is sufficiently functional, it will transition to the [MediaProphet License](https://github.com/mediaprophet/license). 

This model establishes an **ontologically defined rights framework**:
- **Verified Individual Persons**: Granted unencumbered, royalty-free, permissive commons access for personal, academic, humanitarian, and collaborative use.
- **Corporate & Commercial Entities**: Required to obtain commercial utilization licenses with fair payment ledgers cleared via W3C Web Payments and WebID identity attestation.

---

## 3. Hardware-Enforced Silicon Licensing Gates

Instead of relying solely on post-hoc legal remedies, Etalon integrates rights enforcement directly into the compiler (`etalon-forge`) and hardware execution agent (`etalon-agent`).

### Architectural Gateways:

1. **Shader Metadata & N3Logic Capability Manifests**:
   Every compute kernel compiled by `etalon-forge` embeds an ontological license header declaring its data classification and permitted execution classes:
   ```wgsl
   // @license_graph: "urn:mediaprophet:license:v1"
   // @agent_class_exempt: "urn:class:individual-human"
   // @toll_endpoint: "https://trustfactory.org/payments/etalon-compute"
   
   @compute @workgroup_size(64)
   fn main(@builtin(global_invocation_id) global_id: vec3<u32>) { ... }
   ```

2. **Compile-Time Agent Verification (`etalon-forge`)**:
   During AST lowering, `etalon-forge` verifies the executing agent's credentials against the embedded rights graph:
   ```rust
   pub enum AgentClass {
       IndividualHuman(HumanAgencyToken),
       CorporateEntity(CommercialFiduciaryToken),
   }

   pub struct MediaProphetLicenseVerifier;

   impl MediaProphetLicenseVerifier {
       pub fn verify_execution_rights(
           shader_module: &naga::Module,
           executing_agent: &AgentClass,
       ) -> Result<(), RightsViolationError> {
           match executing_agent {
               AgentClass::IndividualHuman(token) => {
                   Self::verify_personhood_attestation(token)?;
                   Ok(())
               }
               AgentClass::CorporateEntity(token) => {
                   if Self::verify_commercial_ledger_clearance(token)? {
                       Ok(())
                   } else {
                       Err(RightsViolationError::CommercialTollUnpaid)
                   }
               }
           }
       }
   }
   ```

3. **Hardware Dispatch Enforcement (`etalon-agent`)**:
   If an unauthorized commercial entity attempts to dispatch an `etalon-tensor` or `etalon-nquin` compute pipeline, the hardware agent rejects the command buffer submission at the HAL boundary, preventing the kernel from ever executing on physical silicon.

---

## 4. Standardized Source Code & Shader Headers

Every Rust source file, WGSL shader, and `.q42` bundle created in the Etalon workspace must carry this exact preamble:

```rust
// Copyright (c) 2026 Timothy Charles Holborn
// https://www.linkedin.com/in/ubiquitous/ | timothy.holborn@gmail.com
//
// This work is licensed under the Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
// Transitioning to: https://github.com/mediaprophet/license
//
// Permissions beyond the scope of this license, including permissive commons use for
// individual human agents and commercial utilization by corporate entities, are governed
// by ontologically defined rights instruments via WebID and Web Payments.
```

---

## 5. Upstream Provenance Preservation

Upstream source code originating from the `gfx-rs/wgpu` and `gfx-rs/naga` repositories retains its historical Apache-2.0 and MIT licensing, strictly preserved within:
[`upstream-licenses/`](../upstream-licenses/)

The upstream licenses apply strictly to unmodified historical source files and do not grant rights to derivative architectures, Qualia-specific integrations, or Etalon hardware agents developed herein.
