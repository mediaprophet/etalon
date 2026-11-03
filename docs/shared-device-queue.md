# Shared device and queue ownership

`Device::owns_queue` compares the native core device with the queue's actual
parent device. It returns `Some(true)` for matching handles, `Some(false)` for
known mismatches, and `None` when ownership cannot be inspected. Names, adapter
identity and caller-assigned queue IDs are not ownership evidence.

`etalon_agent::SharedDeviceQueue::try_new` retains an immutable pair of shared
handles after that check. It refuses unknown ownership as well as mismatches.
Cloning the wrapper preserves the pair; accessors let compute and rendering use
the same device and queue. `HalComputeLane::from_shared_pair` consumes a checked
wrapper while preserving the existing lane constructor for compatibility.

The check establishes handle ownership only. It does not establish device
liveness, generations, allocation budgets, exclusive use, completion or safe
resource retirement. Those remain separate contracts.

WebGPU pairs currently return `None`; browser ownership support requires actual
browser validation. A recognized core/WebGPU cross-backend pair returns
`Some(false)`. Custom backends with no inspectable ownership also return `None`.

Independent qualification passed on an NVIDIA RTX A2000 12GB using DX12,
OpenGL and Vulkan. Each explicitly selected backend executed matching-pair
compute and render readbacks, cloned-handle acceptance and both cross-device
rejections. The test refuses a missing or CPU adapter rather than silently
substituting one. It is ignored by default and runs explicitly with
`ETALON_PAIR_TEST_BACKEND=dx12`, `gl` or `vulkan` and
`cargo test -p etalon-agent --test device_queue_contract -- --ignored`.

The agent crate's 29 unit tests, strict all-target Clippy and WASM compilation
also pass. The full `cargo xtask test` run passed 4,136 tests with 47 skipped;
that aggregate includes CPU, software-adapter and expected-failure cases and is
not a count of physical GPU checks. The three explicit pairing runs are separate
from that aggregate. Test-generated snapshot line endings were restored to their
exact pre-run bytes, and every frozen source file was rechecked.

This API does not yet replace a QualiaDB caller or change its GPU provider pin.
