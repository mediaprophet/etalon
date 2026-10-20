/*!
# Vulkan Requirements

`etalon`'s Vulkan backend requires the following from a physical device:

- `fragmentStoresAndAtomics` must be enabled (see
  <https://github.com/gpuweb/gpuweb/issues/639>).
- `independentBlending` must be enabled (see
  [etalon#2498](https://github.com/mediaprophet/etalon/pull/2498)).
- The `maxDrawIndexedIndexValue` limit must be 32-bit.
- `VK_KHR_storage_buffer_storage_class` must be supported.
*/
