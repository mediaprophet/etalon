use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}}};
use crate::device_caps::GpuAdapterCaps;

#[derive(Clone, Debug)]
pub struct PreparedDevice {
    pub device: etalon::Device,
    pub queue: etalon::Queue,
    pub caps: GpuAdapterCaps,
}

#[derive(Clone, Debug)]
pub struct ProviderSet {
    devices: Vec<PreparedDevice>,
    cuda_owners: Vec<u64>,
    invalid: Arc<AtomicBool>,
    routes: Arc<AtomicU64>,
}

thread_local! {
    static ACTIVE: RefCell<Option<ProviderSet>> = const { RefCell::new(None) };
}

impl ProviderSet {
    pub fn new(devices: Vec<PreparedDevice>, cuda_owners: Vec<u64>) -> Self {
        Self { devices, cuda_owners, invalid: Arc::new(AtomicBool::new(false)), routes: Arc::new(AtomicU64::new(0)) }
    }

    pub fn enter(&self) -> ProviderScope {
        let previous = ACTIVE.with(|active| active.replace(Some(self.clone())));
        ProviderScope { previous, _thread: PhantomData }
    }

    pub fn valid(&self) -> bool { !self.invalid.load(Ordering::Acquire) }
    pub fn executed_routes(&self) -> u64 { self.routes.load(Ordering::Acquire) }
}

#[derive(Debug)]
pub struct ProviderScope { previous: Option<ProviderSet>, _thread: PhantomData<Rc<()>> }
impl Drop for ProviderScope {
    fn drop(&mut self) { ACTIVE.with(|active| { active.replace(self.previous.take()); }); }
}

pub fn prepared_device(coopmat: bool) -> Option<Result<PreparedDevice, &'static str>> {
    ACTIVE.with(|active| active.borrow().as_ref().map(|set| {
        set.devices.iter().find(|device| !coopmat || device.device.features().contains(etalon::Features::EXPERIMENTAL_COOPERATIVE_MATRIX))
            .cloned().ok_or("requested device route was unavailable during preparation")
    }))
}

pub fn admits_device(device: &etalon::Device, queue: &etalon::Queue) -> bool {
    ACTIVE.with(|active| active.borrow().as_ref().is_none_or(|set| {
        let allowed = set.devices.iter().any(|owner| owner.device == *device && owner.queue == *queue);
        if !allowed { set.invalid.store(true, Ordering::Release); }
        allowed
    }))
}

pub fn admits_cuda(owner: u64) -> bool {
    ACTIVE.with(|active| active.borrow().as_ref().is_none_or(|set| {
        let allowed = set.cuda_owners.contains(&owner);
        if !allowed { set.invalid.store(true, Ordering::Release); }
        allowed
    }))
}

pub fn admits_new_owner() -> bool {
    ACTIVE.with(|active| {
        if let Some(set) = active.borrow().as_ref() {
            let _ = set;
            false
        } else { true }
    })
}

/// Bit 0 CPU, bit 1 WebGPU, bit 2 CUDA, bit 3 DirectML, bit 4 Accelerate.
pub fn record_route(route: u64) {
    ACTIVE.with(|active| { if let Some(set) = active.borrow().as_ref() { set.routes.fetch_or(route, Ordering::Relaxed); } });
}
