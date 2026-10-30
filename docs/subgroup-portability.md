# Subgroup regression portability

The October 2026 review found two distinct failures on Intel UHD Graphics 620:
DX12 failed divergent reduction/broadcast cases 27/28; Vulkan failed case 22's
self-shuffle in the upper half of each 32-lane subgroup.

WGSL's [subgroups section](https://gpuweb.github.io/gpuweb/wgsl/#subgroups)
warns that compiler optimizations can change active sets in divergent control
flow. [Subgroup built-ins](https://gpuweb.github.io/gpuweb/wgsl/#subgroup-builtin-functions)
also have uniformity diagnostic requirements. Tests 27–29 previously assumed
particular divergent active sets. They now execute in uniform control flow and
check parity-masked reductions, scalar/vector broadcast and every masked-prefix
count from zero through the subgroup size. The former Mac failure exceptions
covered those removed assumptions and are removed with them. Hardware failures
in the replacement cases still fail normally.

For Vulkan case 22, the emitted SPIR-V used `OpGroupNonUniformShuffle` with the
current subgroup invocation ID as both value and index. This is a legal
self-gather, with no operand or scope mismatch. WGSL's
[shuffle definition](https://gpuweb.github.io/gpuweb/wgsl/#subgroupshuffle-builtin)
selects the invocation identified by the index. Selecting one's own ID returns
one's already evaluated value. Naga therefore folds this identity for direct
built-in inputs, including scalar/vector values and built-ins in input structs.
Other indices retain real shuffle instructions; input evaluation is preserved.
GPU cases retain the original self-shuffle and add nontrivial scalar/vector
self-gathers and a vector gather from another lane.

This is an implementation optimization and a portable test-oracle correction.
It does not establish that arbitrary Intel compiler behavior is correct, nor
that the retired CTS has been run. Naga instruction regressions and actual GPU
reruns are separate acceptance checks.
