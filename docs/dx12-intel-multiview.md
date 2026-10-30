# Intel DX12 fragment view-index forwarding

For Intel vendor `0x8086`, device `0x5917`, driver `31.0.101.2141`, Etalon
enables Naga HLSL's `force_vertex_view_index` interface lowering. It reads
`SV_ViewID` in the vertex stage and forwards the value to the fragment stage
through a private flat scalar interface. Multiview and selective multiview
remain enabled. Other adapter/driver combinations retain their existing path.

The observed failing sequence creates single-view, two-view and sparse-view
pipelines using the same fragment-only view-index shader. After the single-view
pipeline, subsequent active layers can incorrectly receive view index zero.
Runtime captures show that the vertex and fragment DXIL payloads are identical
across these pipelines, and that the fragment DXIL contains the correct ViewID
operation and dependency metadata. This is a reproduced configuration-specific
failure, not a claim that Intel has published an equivalent diagnosis.

Controlled probes used the same device/test setup and exact CPU pixel oracle:

| Shader path, using its own single-view pipeline first | Later two/sparse views    |
| ----------------------------------------------------- | ------------------------- |
| Original fragment-only ViewID                         | Incorrect zero view index |
| Extra unused fragment position input                  | Same failure              |
| Vertex ViewID forwarded through a flat scalar         | Correct view indices      |

The forwarding probe keeps identical shader bytecode across view masks; it
does not depend on random labels or a different entry-point name per mask.
The production regression reuses one device for masks `1, 3, 9, 9, 3, 1`,
alternates absent/present shader labels and checks every returned layer.

The compiler owns this interface adaptation. Its planner handles direct and
structured builtin arguments and vertex results, reserves a private semantic,
and preserves the existing user-location interface. Core validation already
deducts a complete varying for `ViewIndex`, so the adaptation uses that reserved
space. It does not enlarge public interstage limits or bypass validation.
DX12 currently does not expose clip distances; adding that feature will require
rechecking the combined vertex-output register budget for this adaptation.

Forwarding can move view-dependent work earlier into vertex processing. The
workaround is therefore limited to the observed adapter/driver rather than
applied to all hardware. Pipelines that do not need fragment ViewIndex keep
their existing interface. Native HLSL and precompiled DXIL passthrough shaders
are not rewritten; their authors control the paired stage interface.

The [DirectX View Instancing specification](https://microsoft.github.io/DirectX-Specs/d3d/ViewInstancing.html)
permits fragment ViewID use without manual vertex forwarding. The workaround
avoids the failing implementation path while preserving that shader's results.
It does not install drivers, change OS settings or establish CTS coverage.
