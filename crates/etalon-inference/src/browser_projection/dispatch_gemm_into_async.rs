use super::*;
use crate::ordinary::gpu_params::GemmGpuParams;
use etalon_model::inference::TensorInfo as GgufTensorInfo;
impl<P: BrowserProjectionProvider> BrowserProjectionEngine<'_, P> {
    pub async fn dispatch_gemm_into_async(
        &self,
        index: &P::Index,
        info: &GgufTensorInfo,
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool {
        let resources = self.host.resources();
        if n_in > input.len() || n_out > out.len() {
            self.host.log(&format!(
                "[gemm_into_async] GUARD n_in={n_in} n_out={n_out} input={} out={}",
                input.len(),
                out.len()
            ));
            return false;
        }
        let mmap = match self.host.model_bytes() {
            Some(m) => m,
            None => return false,
        };
        let raw = match etalon_model::ggml::fetch_tensor_bytes(mmap, P::data_start(index), info) {
            Ok(s) => s,
            Err(_) => return false,
        };
        self.dispatch_gemm_raw_into_async(info, raw, input, out, n_in, n_out)
            .await
    }
}
