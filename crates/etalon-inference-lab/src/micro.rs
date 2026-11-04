#[derive(Debug, Clone)]
pub struct MicrobenchResult {
    pub n_in: usize,
    pub n_out: usize,
    pub cpu_ms: f64,
    pub cpu_gflops: f64,
    pub cuda_ms: Option<f64>,
    pub cuda_ok: bool,
    pub max_abs_err: f32,
    pub max_ulp: u64,
    pub notes: String,
}

impl MicrobenchResult {
    /// Formats `report`.
    pub fn format_report(&self) -> String {
        format!(
            "Q4_K SoA GEMV microbench\n  shape:     {} x {}\n  cpu_ms:    {:.4}  ({:.2} GFLOP/s)\n  cuda_ms:   {}\n  cuda_ok:   {}\n  max_abs:   {:.6e}\n  max_ulp:   {}\n  {}\n",
            self.n_in,
            self.n_out,
            self.cpu_ms,
            self.cpu_gflops,
            self.cuda_ms
                .map(|m| format!("{m:.4}"))
                .unwrap_or_else(|| "—".into()),
            self.cuda_ok,
            self.max_abs_err,
            self.max_ulp,
            self.notes
        )
    }
}
