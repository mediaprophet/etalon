//! Composable sensing facade. Optional providers require explicit selection;
//! availability, authorization and scientific qualification remain host duties.
#![no_std]
#![forbid(unsafe_code)]

pub use etalon_sense_core as core;
pub use etalon_sense_runtime as runtime;
pub use etalon_signal as signal;

#[cfg(feature = "host")]
pub use etalon_sense_host as host;
#[cfg(feature = "wasapi")]
pub use etalon_sense_wasapi as wasapi;
#[cfg(feature = "wifi")]
pub use etalon_sense_wifi as wifi;
#[cfg(feature = "ble")]
pub use etalon_sense_ble as ble;
#[cfg(feature = "cellular")]
pub use etalon_sense_cellular as cellular;
#[cfg(feature = "sdr")]
pub use etalon_sense_sdr as sdr;
#[cfg(feature = "nfc")]
pub use etalon_sense_nfc as nfc;
#[cfg(all(feature = "pcsc", windows))]
pub use etalon_sense_pcsc as pcsc;
#[cfg(feature = "windows-sensors")]
pub use etalon_sense_windows_sensors as windows_sensors;
#[cfg(feature = "cpu")]
pub use etalon_sense_cpu as cpu;
#[cfg(feature = "camera")]
pub use etalon_sense_camera as camera;
#[cfg(all(feature = "camera-windows", windows))]
pub use etalon_sense_camera_windows as camera_windows;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub use etalon_sense_gpu as gpu;
