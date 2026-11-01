//! Composable sensing facade. Optional providers require explicit selection;
//! availability, authorization and scientific qualification remain host duties.
//! WASAPI, BLE, PC/SC and Windows camera providers are Windows-only; host I/O
//! is native-only. GPU processing and portable observation contracts remain cross-target.
#![no_std]
#![forbid(unsafe_code)]

pub use etalon_sense_core as core;
pub use etalon_sense_runtime as runtime;
pub use etalon_signal as signal;

#[cfg(all(feature = "ble", windows))]
pub use etalon_sense_ble as ble;
#[cfg(feature = "camera")]
pub use etalon_sense_camera as camera;
#[cfg(all(feature = "camera-windows", windows))]
pub use etalon_sense_camera_windows as camera_windows;
#[cfg(feature = "cellular")]
pub use etalon_sense_cellular as cellular;
#[cfg(feature = "cpu")]
pub use etalon_sense_cpu as cpu;
#[cfg(feature = "gpu")]
pub use etalon_sense_gpu as gpu;
#[cfg(all(feature = "host", not(target_family = "wasm")))]
pub use etalon_sense_host as host;
#[cfg(feature = "nfc")]
pub use etalon_sense_nfc as nfc;
#[cfg(all(feature = "pcsc", windows))]
pub use etalon_sense_pcsc as pcsc;
#[cfg(feature = "sdr")]
pub use etalon_sense_sdr as sdr;
#[cfg(all(feature = "wasapi", windows))]
pub use etalon_sense_wasapi as wasapi;
#[cfg(feature = "wifi")]
pub use etalon_sense_wifi as wifi;
#[cfg(feature = "windows-sensors")]
pub use etalon_sense_windows_sensors as windows_sensors;
