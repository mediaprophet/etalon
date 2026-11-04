# Numerical compatibility frontend

Etalon owns this runtime and the graph execution/compiler extensions. The historical
frontend preserves Qualia's serialized operation schema, three-field schedules, error categories,
manifest schema 2, and source/toolchain cache identity. Primitive builtin specs and
emission delegate to the canonical Forge rather than owning another kernel set.
Primitive buffer/scalar/intrinsic types and the P64 GPU POD are canonical reexports.
The canonical public Forge remains available, including its macro operators.

Qualia owns semantic graph encoding, P64 semantic projection, application device
selection/passport preferences, global shared-context initialization, governance,
and calibration integration. It installs typed device-policy callbacks. The standalone
Etalon path uses explicit ETALON_WGPU_BACKEND/EXPERIMENTAL_FEATURES preferences.

The migrated dispatch/runtime/execute and graph files retain inherited cohesive
algorithms and tests during the source-preserving migration. Their inherited size
is an ownership-review exception for EM02/03, not permission for unrelated growth.
The migration coordinator reviews decomposition of dispatch by precision/provider,
execution by pipeline/readback/acceleration structure, and graph builders by operation
family before adding substantial new behavior. Qualification is deferred to the
consolidated migration pass; source/compile success does not claim runtime parity.
