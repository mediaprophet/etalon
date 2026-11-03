use super::{PleNgramConfigSource, PleNgramError, MAX_PLE_NGRAM};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PleTokenHistory {
    pub(super) previous: [u32; MAX_PLE_NGRAM - 1],
    pub(super) count: usize,
}

impl PleTokenHistory {
    /// Compute row IDs for `token` and then retain it for the following step.
    ///
    /// Missing history, and history preceding EOS, is represented by the
    /// checkpoint's EOS token. This is the reference `shift_right` behaviour:
    /// a n-gram never crosses an EOS boundary.
    pub fn select_and_push<C: PleNgramConfigSource + ?Sized>(
        &mut self,
        config: &C,
        token: u32,
        table_rows: u64,
        out: &mut [u64],
    ) -> Result<usize, PleNgramError> {
        let config = config.ple_ngram_config();
        let count = super::rows::select_rows_view(config, token, self, table_rows, out)?;
        if token == config.eos_token_id {
            self.count = 0;
            return Ok(count);
        }
        let retain = self.previous.len();
        if self.count < retain {
            for index in (1..=self.count).rev() {
                self.previous[index] = self.previous[index - 1];
            }
            self.count += 1;
        } else {
            for index in (1..retain).rev() {
                self.previous[index] = self.previous[index - 1];
            }
        }
        self.previous[0] = token;
        Ok(count)
    }
}
