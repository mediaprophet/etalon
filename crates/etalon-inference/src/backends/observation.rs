use std::sync::OnceLock;
static OBSERVER: OnceLock<fn(u64)> = OnceLock::new();
pub fn install(observer: fn(u64)) -> Result<(), fn(u64)> {
    OBSERVER.set(observer)
}
pub fn record(route: u64) {
    if let Some(observer) = OBSERVER.get() {
        observer(route);
    }
}
