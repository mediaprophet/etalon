use super::*;
#[test]
fn rejects_only_known_foreign_queue() {
    assert!(validate_queue_ownership(Some(false)).is_err());
    assert!(validate_queue_ownership(Some(true)).is_ok());
    assert!(validate_queue_ownership(None).is_ok());
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::{
        sync::Arc,
        task::{Context, Poll, Waker},
    };

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        struct WakeThread;
        impl std::task::Wake for WakeThread {
            fn wake(self: Arc<Self>) {}
        }
        let waker = Waker::from(Arc::new(WakeThread));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
            std::thread::yield_now();
        }
    }
    fn device() -> (etalon::Device, etalon::Queue) {
        let instance =
            etalon::Instance::new(etalon::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&etalon::RequestAdapterOptions::default()))
            .expect("native GPU adapter required");
        block_on(adapter.request_device(&etalon::DeviceDescriptor::default()))
            .expect("native GPU device required")
    }

    #[test]
    #[ignore = "requires native GPU"]
    fn known_foreign_queue_is_rejected_and_loss_invalidates_shared_lease() {
        let (first, first_queue) = device();
        let (_, second_queue) = device();
        assert_eq!(first.owns_queue(&second_queue), Some(false));
        assert!(DeviceLease::new(&first, &second_queue).is_err());
        let lease = DeviceLease::new(&first, &first_queue).unwrap();
        let shared = DeviceLease::new(&first, &first_queue).unwrap();
        assert_eq!(lease.identity(), shared.identity());
        assert!(lease.is_live());
        let _ = first_queue.submit([]);
        first.destroy();
        assert!(!lease.is_live());
        assert!(!shared.is_live());
    }
}
