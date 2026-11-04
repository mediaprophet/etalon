/// One benchmark case: a model + prompt to drive through the real path.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Human-readable row label, e.g. "SmolLM2-360M Q8".
    pub label: String,
    /// Path to the native model artifact on disk.
    pub model_path: String,
    /// Descriptive quantization tag for the report (e.g. "Q8_0").
    pub quantization: String,
    /// Prompt to run.
    pub prompt: String,
    /// Fixed decode-token count for a bounded, comparable measurement (0 = production default).
    pub decode_tokens: u32,
    /// Warm repeats to average over (≥1). Cold is always a single fresh run.
    pub warm_repeats: u32,
}

impl BenchConfig {
    /// Creates `BenchConfig` with the given state.
    pub fn new(
        label: impl Into<String>,
        model_path: impl Into<String>,
        quantization: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            model_path: model_path.into(),
            quantization: quantization.into(),
            prompt: prompt.into(),
            decode_tokens: 64,
            warm_repeats: 3,
        }
    }
}
