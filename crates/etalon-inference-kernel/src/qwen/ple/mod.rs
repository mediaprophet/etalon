//! Trained PLE n-gram row selection over caller-owned history and output.
//! No table I/O, layer admission, residual arithmetic or full model execution.
//! Kernel operations and the borrowed view allocate nothing. A custom projection
//! trait implementation is external code and must be qualified separately.
mod config;
mod history;
mod rows;
pub use config::{PleNgramConfig, PleNgramConfigSource};
pub use history::PleTokenHistory;
pub use rows::{select_rows, sort_dedup_rows, PleNgramError};
pub const MAX_PLE_NGRAM: usize = 8;
pub const MAX_PLE_HEADS: usize = 32;
#[cfg(test)]
mod tests;
