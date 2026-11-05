use std::sync::OnceLock;
static OBSERVER: OnceLock<fn(u64)> = OnceLock::new();
pub fn install(observer: fn(u64)) -> Result<(), fn(u64)> {
    OBSERVER.set(observer)
}
#[allow(dead_code)]
pub(super) fn record(route: u64) {
    if let Some(observer) = OBSERVER.get() {
        observer(route);
    }
}
