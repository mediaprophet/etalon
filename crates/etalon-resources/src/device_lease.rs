use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Mutex, OnceLock, Weak};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// Process-local construction identity, not a hardware identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceIdentity(u64);

impl DeviceIdentity {
    pub fn generation(self) -> u64 { self.0 }
}

#[derive(Clone, Debug)]
pub struct DeviceLease {
    owner: Arc<DeviceOwner>,
}

#[derive(Debug)]
struct DeviceOwner {
    device: etalon::Device,
    queue: etalon::Queue,
    identity: DeviceIdentity,
    lost: Arc<AtomicBool>,
}

impl DeviceLease {
    pub fn new(device: &etalon::Device, queue: &etalon::Queue) -> Result<Self, String> {
        #[cfg(not(target_arch = "wasm32"))]
        let mut registry = {
            static OWNERS: OnceLock<Mutex<[Option<Weak<DeviceOwner>>; 64]>> = OnceLock::new();
            OWNERS.get_or_init(|| Mutex::new(std::array::from_fn(|_| None)))
                .lock().map_err(|_| "device lease registry poisoned")?
        };
        #[cfg(not(target_arch = "wasm32"))]
        for entry in registry.iter_mut() {
            if let Some(owner) = entry.as_ref().and_then(Weak::upgrade) {
                if owner.device == *device && owner.queue == *queue { return Ok(Self { owner }); }
            } else { *entry = None; }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let slot = registry.iter_mut().find(|entry| entry.is_none()).ok_or("device lease capacity exceeded")?;
        let lost = Arc::new(AtomicBool::new(false));
        let callback_lost = lost.clone();
        device.set_device_lost_callback(move |_, _| {
            callback_lost.store(true, Ordering::Release);
        });
        let owner = Arc::new(DeviceOwner {
            device: device.clone(),
            queue: queue.clone(),
            identity: DeviceIdentity(NEXT_OWNER.fetch_add(1, Ordering::Relaxed)),
            lost,
        });
        #[cfg(not(target_arch = "wasm32"))]
        { *slot = Some(Arc::downgrade(&owner)); }
        Ok(Self { owner })
    }

    pub fn identity(&self) -> DeviceIdentity { self.owner.identity }

    pub fn matches(&self, device: &etalon::Device, queue: &etalon::Queue) -> bool {
        self.owner.device == *device && self.owner.queue == *queue && device.owns_queue(queue) != Some(false)
    }

    pub fn is_live(&self) -> bool {
        if self.owner.lost.load(Ordering::Acquire) { return false; }
        #[cfg(not(target_arch = "wasm32"))]
        if self.owner.device.poll(etalon::PollType::Poll).is_err() { return false; }
        if self.owner.device.is_valid_state() == Some(false) { return false; }
        !self.owner.lost.load(Ordering::Acquire)
    }

    pub fn device(&self) -> &etalon::Device { &self.owner.device }
    pub fn queue(&self) -> &etalon::Queue { &self.owner.queue }
}
