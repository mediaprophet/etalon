//! In-shader SieveStateMask and GPU block top-1 / top-k reduction.

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LogitMaskHeader {
    pub mask_words: [u32; 16],
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
}
