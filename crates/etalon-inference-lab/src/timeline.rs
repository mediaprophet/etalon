#[derive(Debug, Clone)]
pub struct DecodeTimeline {
    pub model: String,
    pub tokens: u32,
    pub tok_s: Option<f64>,
    pub wall_ms: f64,
    pub resident_hits: u64,
    pub resident_fallbacks: u64,
    pub host_phase_ns: String,
    pub gpu_phase_ns: String,
    pub notes: String,
}

impl DecodeTimeline {
    /// Formats `report`.
    pub fn format_report(&self) -> String {
        format!(
            "Decode timeline\n  model:     {}\n  tokens:    {}\n  tok_s:     {}\n  wall_ms:   {:.2}\n  resident:  hits={} fallbacks={}\n  host_ns:   {}\n  gpu_ns:    {}\n  notes:     {}\n",
            self.model,
            self.tokens,
            self.tok_s
                .map(|t| format!("{t:.4}"))
                .unwrap_or_else(|| "fail".into()),
            self.wall_ms,
            self.resident_hits,
            self.resident_fallbacks,
            self.host_phase_ns,
            self.gpu_phase_ns,
            self.notes
        )
    }
}
