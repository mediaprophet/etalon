//! Optional captured CUDA graph mega-pass execution.

#[derive(Debug)]
pub struct CapturedCudaMegaPass {
    pub graph_instantiated: bool,
}
