#[derive(Debug, Clone)]
pub struct DeviceRoof {
    pub gemv_n: usize,
    pub best_label: String,
    pub best_backend: String,
    pub gemv_ms: f64,
    pub gemv_gflops: f64,
    pub upload_gbps: f64,
    pub balance_flop_per_byte: f64,
    pub notes: String,
}

impl DeviceRoof {
    /// Formats `report`.
    pub fn format_report(&self) -> String {
        format!(
            "Device roof calibration\n  gemv_n:        {}\n  best:          {} [{}]\n  gemv_ms:       {:.4}\n  gemv_gflops:   {:.2}\n  upload_gbps:   {:.2}\n  balance FLOP/B:{:.2}\n  {}\n",
            self.gemv_n,
            self.best_label,
            self.best_backend,
            self.gemv_ms,
            self.gemv_gflops,
            if self.upload_gbps.is_finite() {
                self.upload_gbps
            } else {
                -1.0
            },
            self.balance_flop_per_byte,
            self.notes
        )
    }
}
