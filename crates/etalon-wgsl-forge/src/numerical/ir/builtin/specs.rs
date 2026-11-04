use super::super::core::KernelSpec;
use super::BuiltinKernel;
impl BuiltinKernel {
    pub fn spec(self) -> KernelSpec {
        let canonical: crate::BuiltinKernel = self.into();
        serde_json::from_value(
            serde_json::to_value(canonical.spec()).expect("builtin serialization"),
        )
        .expect("compatible builtin schema")
    }
}
