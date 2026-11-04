use super::{PleNvmeReader, QwenHyperparams, TrunkNvmeReader};
use etalon_model::inference::{ModelHyperparams, TensorIndex};

pub struct Qwen4ExpRuntime<'a, H: QwenHyperparams = ModelHyperparams> {
    pub index: &'a TensorIndex<H>,
    pub ple: &'a mut PleNvmeReader,
    pub trunk: &'a mut TrunkNvmeReader,
}
