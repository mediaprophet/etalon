pub mod params;
mod provider;
pub mod types;
pub use provider::{PrefillProvider, PrefillResources};
use types::*;
pub struct PrefillEngine<'a, P> {
    pub host: &'a P,
}
mod execute;
mod planning;
mod tail;

pub mod encoding;
mod encoding_attention;
mod encoding_attention_offset;
mod encoding_element;
mod encoding_ffn;
mod encoding_gemm;
