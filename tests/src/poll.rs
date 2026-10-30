use crate::TestingContext;

impl TestingContext {
    /// Wait asynchronously on WebGPU, and use device polling on native backends.
    pub async fn async_poll(
        &self,
        poll_type: etalon::PollType,
    ) -> Result<etalon::PollStatus, etalon::PollError> {
        #[cfg(target_arch = "wasm32")]
        if self.adapter_info.backend == etalon::Backend::BrowserWebGpu {
            if let etalon::PollType::Wait {
                submission_index,
                timeout,
            } = poll_type
            {
                assert!(
                    submission_index.is_none(),
                    "WebGPU has no submission indices"
                );
                let queue = self.queue.as_webgpu().expect("WebGPU queue");
                let completed = browser_wait(
                    queue.as_ref(),
                    timeout.map_or(-1.0, |duration| duration.as_secs_f64() * 1000.0),
                )
                .await
                .expect("WebGPU queue completion promise rejected")
                .as_bool()
                .expect("queue wait result must be a boolean");
                return if completed {
                    Ok(etalon::PollStatus::WaitSucceeded)
                } else {
                    Err(etalon::PollError::Timeout)
                };
            }
        }
        self.device.poll(poll_type)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = "
export async function browser_wait(queue, timeoutMs) {
    const completed = queue.onSubmittedWorkDone().then(() => true);
    if (timeoutMs < 0) return await completed;
    const deadline = performance.now() + timeoutMs;
    let timer;
    const expired = new Promise(resolve => {
        const tick = () => {
            const remaining = deadline - performance.now();
            if (remaining <= 0) resolve(false);
            else timer = setTimeout(tick, Math.min(remaining, 2147483647));
        };
        tick();
    });
    try {
        return await Promise.race([completed, expired]);
    } finally {
        clearTimeout(timer);
    }
}
")]
extern "C" {
    #[wasm_bindgen(catch)]
    async fn browser_wait(
        queue: &wasm_bindgen::JsValue,
        timeout_ms: f64,
    ) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;
}
