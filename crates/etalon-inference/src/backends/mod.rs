#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
#[allow(unsafe_code)]
pub mod cuda;
#[cfg(all(target_os = "windows", feature = "native-directml"))]
#[allow(unsafe_code)]
pub mod directml;
#[cfg(feature = "metal-lane")]
pub mod metal_lane;
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    feature = "native-accelerate"
))]
#[allow(unsafe_code)]
pub mod native_accelerate;
pub mod observation;
