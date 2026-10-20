/*!
## Feature flags
*/
#![doc = document_features::document_features!()]
/*!

### Feature Aliases

These features aren't actually features on the crate itself, but a convenient shorthand for
complicated cases.

- **`etalon_core`** --- Enabled when there is any non-webgpu backend enabled on the platform.
- **`naga`** --- Enabled when target `glsl` or `spirv` input is enabled, or when `etalon_core` is enabled.
*/

use crate::Adapter;
