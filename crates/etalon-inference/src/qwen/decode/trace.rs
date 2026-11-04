/// One observed tensor inside a token step, for lab-style localization of a
/// divergent stage.  `layer` is `u32::MAX` for the embedding/final stages.
#[derive(Clone, Debug)]
pub struct Qwen4ExpTraceRecord {
    pub step: usize,
    pub layer: u32,
    pub stage: &'static str,
    pub fingerprint: u64,
    pub rms: f32,
    pub abs_max: f32,
    /// Stage values, populated only when the caller passes `capture=true`;
    /// kept empty on the production decode path so it stays allocation-free.
    pub values: Vec<f32>,
}

/// Layer sentinel for records that do not belong to a numbered block.
pub const QWEN4EXP_TRACE_META_LAYER: u32 = u32::MAX;

pub trait Qwen4ExpTraceSink {
    fn record(&mut self, step: usize, layer: u32, stage: &'static str, data: &[f32], capture: bool);
}
impl Qwen4ExpTraceSink for () {
    fn record(&mut self, _: usize, _: u32, _: &'static str, _: &[f32], _: bool) {}
}
impl Qwen4ExpTraceSink for Vec<Qwen4ExpTraceRecord> {
    fn record(
        &mut self,
        step: usize,
        layer: u32,
        stage: &'static str,
        data: &[f32],
        capture: bool,
    ) {
        let mut fingerprint = 0xcbf2_9ce4_8422_2325u64;
        let mut sum_sq = 0.0f64;
        let mut abs_max = 0.0f32;
        for &value in data {
            fingerprint ^= value.to_bits() as u64;
            fingerprint = fingerprint.wrapping_mul(0x1000_0000_01b3);
            sum_sq += (value as f64) * (value as f64);
            abs_max = abs_max.max(value.abs());
        }
        self.push(Qwen4ExpTraceRecord {
            step,
            layer,
            stage,
            fingerprint,
            rms: (sum_sq / data.len().max(1) as f64).sqrt() as f32,
            abs_max,
            values: if capture { data.to_vec() } else { Vec::new() },
        });
    }
}
pub(super) fn push_trace(
    trace: &mut impl Qwen4ExpTraceSink,
    step: usize,
    layer: u32,
    stage: &'static str,
    data: &[f32],
    capture: bool,
) {
    trace.record(step, layer, stage, data, capture);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen4ExpTraceSummary {
    pub step: usize,
    pub layer: u32,
    pub stage: &'static str,
    pub fingerprint: u64,
    pub rms: f32,
    pub abs_max: f32,
    pub value_start: usize,
    pub value_count: usize,
}

pub struct Qwen4ExpTraceBuffer<'a> {
    pub records: &'a mut [Qwen4ExpTraceSummary],
    pub values: &'a mut [f32],
    pub records_written: usize,
    pub values_written: usize,
    pub refused: usize,
}
impl Qwen4ExpTraceSink for Qwen4ExpTraceBuffer<'_> {
    fn record(
        &mut self,
        step: usize,
        layer: u32,
        stage: &'static str,
        data: &[f32],
        capture: bool,
    ) {
        let count = if capture { data.len() } else { 0 };
        if self.records_written >= self.records.len()
            || self
                .values_written
                .checked_add(count)
                .filter(|n| *n <= self.values.len())
                .is_none()
        {
            self.refused = self.refused.saturating_add(1);
            return;
        }
        let mut fingerprint = 0xcbf2_9ce4_8422_2325u64;
        let mut sum_sq = 0.0f64;
        let mut abs_max = 0.0f32;
        for &value in data {
            fingerprint ^= value.to_bits() as u64;
            fingerprint = fingerprint.wrapping_mul(0x1000_0000_01b3);
            sum_sq += (value as f64) * (value as f64);
            abs_max = abs_max.max(value.abs());
        }
        let start = self.values_written;
        if capture {
            self.values[start..start + count].copy_from_slice(data);
        }
        self.records[self.records_written] = Qwen4ExpTraceSummary {
            step,
            layer,
            stage,
            fingerprint,
            rms: (sum_sq / data.len().max(1) as f64).sqrt() as f32,
            abs_max,
            value_start: start,
            value_count: count,
        };
        self.records_written += 1;
        self.values_written += count;
    }
}

/// Measurement receipt for a real multi-token decode run.  `trunk_*` and
/// `ple_*` counts come from the readers' own counters, not estimates.
#[derive(Clone, Debug, Default)]
pub struct Qwen4ExpDecodeReceipt {
    pub prompt_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub elapsed: std::time::Duration,
    pub trunk_reads: u64,
    pub trunk_rows: u64,
    pub trunk_bytes: u64,
    pub ple_reads: u64,
    pub ple_rows: u64,
    pub ple_bytes: u64,
    pub state_bytes: u64,
    pub scratch_bytes: u64,
}
