use super::{
    validate_qwen4exp_graph, PleNvmeError, PleNvmeReader, Qwen4ExpGraphError, Qwen4ExpRuntime,
    QwenHyperparams, TrunkNvmeError, TrunkNvmeReader,
};
use etalon_model::inference::{
    ModelHyperparams, ModelMetadata, ModelPreparationBudget, PrepareModelError, PreparedP64,
    TensorIndex,
};
use etalon_model::p64::IntegrityMode;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4ExpWeightResidency {
    Streamed,
    Mapped,
}

#[derive(Debug)]
pub enum Qwen4ExpP64Error {
    Model(PrepareModelError),
    Graph(Qwen4ExpGraphError),
    Trunk(TrunkNvmeError),
    Ple(PleNvmeError),
}
impl core::fmt::Display for Qwen4ExpP64Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Qwen P64 activation: {self:?}")
    }
}
impl std::error::Error for Qwen4ExpP64Error {}

pub struct Qwen4ExpP64Runtime<H: QwenHyperparams = ModelHyperparams> {
    pub index: TensorIndex<H>,
    pub ple: PleNvmeReader,
    pub trunk: TrunkNvmeReader,
}

impl<H: QwenHyperparams> Qwen4ExpP64Runtime<H> {
    pub fn open(
        source: &Path,
        integrity: IntegrityMode,
        metadata: Option<ModelMetadata>,
        model_budget: ModelPreparationBudget,
        parallel_scratch_bytes: usize,
        residency: Qwen4ExpWeightResidency,
    ) -> Result<Self, Qwen4ExpP64Error> {
        let mut trunk = TrunkNvmeReader::open_mapped_with_budget(source, 0, parallel_scratch_bytes)
            .map_err(Qwen4ExpP64Error::Trunk)?;
        let prepared = PreparedP64::parse(
            trunk.mapped_bytes().expect("mapped source"),
            integrity,
            model_budget,
            metadata,
        )
        .map_err(Qwen4ExpP64Error::Model)?;
        let index = prepared.into_index();
        let mut contracts = [Default::default(); 64];
        validate_qwen4exp_graph(&index, &mut contracts).map_err(Qwen4ExpP64Error::Graph)?;
        let ple = PleNvmeReader::open_from_index(source, &index).map_err(Qwen4ExpP64Error::Ple)?;
        if residency == Qwen4ExpWeightResidency::Streamed {
            trunk = TrunkNvmeReader::open(source, 0).map_err(Qwen4ExpP64Error::Trunk)?;
        }
        Ok(Self { index, ple, trunk })
    }

    pub fn engine(&mut self) -> Qwen4ExpRuntime<'_, H> {
        Qwen4ExpRuntime {
            index: &self.index,
            ple: &mut self.ple,
            trunk: &mut self.trunk,
        }
    }
}
