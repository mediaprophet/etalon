mod package;
mod run;
mod types;
pub use package::*;
pub use run::*;
use std::path::Path;
pub use types::*;
pub trait LabMode: Copy + Clone + std::fmt::Debug + PartialEq + Eq + std::hash::Hash {
    const PORTABLE: Self;
    const FAST_VERIFY: Self;
    const CUDA_TC: Self;
    fn as_str(self) -> &'static str;
}
#[derive(Debug, Clone, Default)]
pub struct InstrumentReports {
    pub audit: String,
    pub roof: String,
    pub micro: String,
    pub best_label: String,
    pub micro_cuda_ok: bool,
    pub micro_max_ulp: u64,
    pub notes: Vec<String>,
    pub ffn_fusion_in_resident_decode: bool,
    pub passport_key: String,
}
pub trait AutoImproveHost<M: LabMode> {
    fn instruments(&mut self) -> Result<InstrumentReports, String>;
    fn measure(&mut self, model: &Path, tokens: u32, config: &LabConfig<M>) -> TrialResult<M>;
    fn external_comparison(&mut self, _url: &str, _model: &str, _tokens: u32) -> Option<f64> {
        None
    }
}

pub trait ConfigApplier: LabMode {
    fn apply_config(config: &LabConfig<Self>);
}
impl<M: ConfigApplier> LabConfig<M> {
    pub fn apply(&self) {
        M::apply_config(self);
    }
}
