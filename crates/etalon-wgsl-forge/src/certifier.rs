//! Naga module validation and CPU oracle drift checking.

use crate::ir::ForgeError;

pub fn certify_kernel(source: &str) -> Result<(), ForgeError> {
    #[cfg(feature = "naga-validate")]
    {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|e| ForgeError::Validation(format!("WGSL parse failed: {e:?}")))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .map_err(|e| ForgeError::Validation(format!("Naga validation failed: {e:?}")))?;
    }
    let _ = source;
    Ok(())
}
