use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};
use std::sync::{Arc, Mutex};

/// Failures from the cold GPU query convenience APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GpuQueryError {
    #[error("input size cannot be represented by the shader")]
    InputTooLarge,
    #[error("GPU limits do not admit this query")]
    UnsupportedLimits,
    #[error("output needs {required} rows but has {provided}")]
    OutputTooSmall { required: usize, provided: usize },
    #[error("GPU buffer mapping failed or was cancelled")]
    Mapping,
    #[error("GPU progress timed out or polling failed")]
    Poll,
    #[error("GPU output failed validation")]
    InvalidOutput,
}

#[derive(Default)]
struct Completion {
    result: Option<Result<(), etalon::BufferAsyncError>>,
    waker: Option<Waker>,
}

impl Completion {
    fn finish(&mut self, result: Result<(), etalon::BufferAsyncError>) -> Option<Waker> {
        self.result = Some(result);
        self.waker.take()
    }

    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), GpuQueryError>> {
        if let Some(result) = self.result.take() {
            Poll::Ready(result.map_err(|_| GpuQueryError::Mapping))
        } else {
            self.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

pub(crate) struct MapRead {
    pub buffer: etalon::Buffer,
    completion: Arc<Mutex<Completion>>,
}

impl MapRead {
    pub fn new(buffer: etalon::Buffer) -> Self {
        let completion = Arc::new(Mutex::new(Completion::default()));
        let callback = Arc::clone(&completion);
        buffer
            .slice(..)
            .map_async(etalon::MapMode::Read, move |result| {
                let waker = {
                    let mut state = callback.lock().unwrap();
                    state.finish(result)
                };
                if let Some(waker) = waker {
                    waker.wake();
                }
            });
        Self { buffer, completion }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_blocking(&mut self, device: &etalon::Device) -> Result<(), GpuQueryError> {
        device
            .poll(etalon::PollType::wait_or_timeout(
                core::time::Duration::from_secs(5),
            ))
            .map_err(|_| GpuQueryError::Poll)?;
        self.completion
            .lock()
            .unwrap()
            .result
            .take()
            .ok_or(GpuQueryError::Poll)?
            .map_err(|_| GpuQueryError::Mapping)
    }
}

impl Future for MapRead {
    type Output = Result<(), GpuQueryError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.completion.lock().unwrap().poll(cx)
    }
}

impl Drop for MapRead {
    fn drop(&mut self) {
        // Unmap also aborts pending mapping. All BufferViews are scoped inside the caller.
        let waker = self.completion.lock().unwrap().waker.take();
        drop(waker);
        self.buffer.unmap();
    }
}

pub(crate) fn validate_capacity(required: usize, provided: usize) -> Result<(), GpuQueryError> {
    if provided < required {
        Err(GpuQueryError::OutputTooSmall { required, provided })
    } else {
        Ok(())
    }
}

pub(crate) fn read_word(bytes: &[u8], index: usize) -> Result<u32, GpuQueryError> {
    let start = index.checked_mul(4).ok_or(GpuQueryError::InvalidOutput)?;
    let end = start.checked_add(4).ok_or(GpuQueryError::InvalidOutput)?;
    let raw: [u8; 4] = bytes
        .get(start..end)
        .ok_or(GpuQueryError::InvalidOutput)?
        .try_into()
        .map_err(|_| GpuQueryError::InvalidOutput)?;
    Ok(u32::from_le_bytes(raw))
}

pub(crate) fn check_buffer(
    limits: &etalon::Limits,
    size: u64,
    storage: bool,
) -> Result<(), GpuQueryError> {
    if size > limits.max_buffer_size || (storage && size > limits.max_storage_buffer_binding_size) {
        return Err(GpuQueryError::UnsupportedLimits);
    }
    Ok(())
}

pub(crate) fn check_dispatch(
    limits: &etalon::Limits,
    groups: u32,
    lanes: u32,
    shared: u32,
    storage_buffers: u32,
) -> Result<(), GpuQueryError> {
    if groups > limits.max_compute_workgroups_per_dimension
        || lanes > limits.max_compute_invocations_per_workgroup
        || lanes > limits.max_compute_workgroup_size_x
        || shared > limits.max_compute_workgroup_storage_size
        || storage_buffers > limits.max_storage_buffers_per_shader_stage
        || storage_buffers + 1 > limits.max_buffers_and_acceleration_structures_per_shader_stage
        || limits.max_uniform_buffer_binding_size < 16
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_bind_groups < 1
        || limits.max_bindings_per_bind_group < storage_buffers + 1
    {
        return Err(GpuQueryError::UnsupportedLimits);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_storage_and_uniform_limit_is_checked() {
        let mut limits = etalon::Limits::default();
        limits.max_buffers_and_acceleration_structures_per_shader_stage = 5;
        assert_eq!(
            check_dispatch(&limits, 1, 64, 1024, 5),
            Err(GpuQueryError::UnsupportedLimits)
        );
        limits.max_buffers_and_acceleration_structures_per_shader_stage = 6;
        assert_eq!(check_dispatch(&limits, 1, 64, 1024, 5), Ok(()));
    }

    #[test]
    fn capacity_and_unaligned_readback_are_checked() {
        assert_eq!(
            validate_capacity(3, 2),
            Err(GpuQueryError::OutputTooSmall {
                required: 3,
                provided: 2
            })
        );
        assert_eq!(validate_capacity(0, 0), Ok(()));
        let bytes = [0, 0x78, 0x56, 0x34, 0x12];
        assert_eq!(read_word(&bytes[1..], 0), Ok(0x12345678));
        assert_eq!(read_word(&bytes, 1), Err(GpuQueryError::InvalidOutput));
        assert_eq!(
            read_word(&bytes, usize::MAX),
            Err(GpuQueryError::InvalidOutput)
        );
    }
}

#[cfg(test)]
mod completion_tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        task::Wake,
    };

    #[derive(Default)]
    struct Counter(AtomicUsize);
    impl Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn completion_handles_callbacks_before_poll_and_after_waker_registration() {
        let counter = Arc::new(Counter::default());
        let waker = Waker::from(Arc::clone(&counter));
        let mut cx = Context::from_waker(&waker);
        let mut state = Completion::default();
        assert!(state.finish(Ok(())).is_none());
        assert_eq!(state.poll(&mut cx), Poll::Ready(Ok(())));
        assert!(state.poll(&mut cx).is_pending());
        state.finish(Err(etalon::BufferAsyncError)).unwrap().wake();
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        assert_eq!(
            state.poll(&mut cx),
            Poll::Ready(Err(GpuQueryError::Mapping))
        );
        assert!(state.waker.is_none());
    }
}
