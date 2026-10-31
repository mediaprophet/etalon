//! C ABI declarations from the pinned public Osmocom header, not its implementation.
use core::ffi::{c_char, c_int, c_void};

#[repr(C)]
pub(crate) struct Dev { _opaque: [u8; 0] }

pub(crate) type Open = unsafe extern "C" fn(*mut *mut Dev, u32) -> c_int;
pub(crate) type Close = unsafe extern "C" fn(*mut Dev) -> c_int;
pub(crate) type SetU32 = unsafe extern "C" fn(*mut Dev, u32) -> c_int;
pub(crate) type GetU32 = unsafe extern "C" fn(*mut Dev) -> u32;
pub(crate) type SetInt = unsafe extern "C" fn(*mut Dev, c_int) -> c_int;
pub(crate) type GetInt = unsafe extern "C" fn(*mut Dev) -> c_int;
pub(crate) type Reset = unsafe extern "C" fn(*mut Dev) -> c_int;
pub(crate) type Read = unsafe extern "C" fn(*mut Dev, *mut c_void, c_int, *mut c_int) -> c_int;
pub(crate) type UsbStrings = unsafe extern "C" fn(*mut Dev, *mut c_char, *mut c_char, *mut c_char) -> c_int;
pub(crate) type Gains = unsafe extern "C" fn(*mut Dev, *mut c_int) -> c_int;

#[derive(Clone, Copy)]
pub(crate) struct Api {
    pub open: Open, pub close: Close, pub read: Read, pub reset: Reset,
    pub usb_strings: UsbStrings,
    pub set_center: SetU32, pub get_center: GetU32,
    pub set_rate: SetU32, pub get_rate: GetU32,
    pub set_correction: SetInt, pub get_correction: GetInt,
    pub set_gain_mode: SetInt, pub set_gain: SetInt, pub get_gain: GetInt, pub get_gains: Gains,
    pub set_agc: SetInt, pub set_testmode: SetInt,
    pub set_direct: SetInt, pub get_direct: GetInt,
}

#[cfg(any(windows, unix))]
impl Api {
    pub unsafe fn load(library: &libloading::Library) -> Result<Self, crate::Error> {
        macro_rules! symbol {
            ($name:literal, $kind:ty) => {
                *unsafe { library.get::<$kind>(concat!($name, "\0").as_bytes()) }
                    .map_err(|_| crate::Error::MissingSymbol)?
            };
        }
        Ok(Self {
            open: symbol!("rtlsdr_open", Open), close: symbol!("rtlsdr_close", Close),
            read: symbol!("rtlsdr_read_sync", Read), reset: symbol!("rtlsdr_reset_buffer", Reset),
            usb_strings: symbol!("rtlsdr_get_usb_strings", UsbStrings),
            set_center: symbol!("rtlsdr_set_center_freq", SetU32), get_center: symbol!("rtlsdr_get_center_freq", GetU32),
            set_rate: symbol!("rtlsdr_set_sample_rate", SetU32), get_rate: symbol!("rtlsdr_get_sample_rate", GetU32),
            set_correction: symbol!("rtlsdr_set_freq_correction", SetInt), get_correction: symbol!("rtlsdr_get_freq_correction", GetInt),
            set_gain_mode: symbol!("rtlsdr_set_tuner_gain_mode", SetInt),
            set_gain: symbol!("rtlsdr_set_tuner_gain", SetInt), get_gain: symbol!("rtlsdr_get_tuner_gain", GetInt),
            get_gains: symbol!("rtlsdr_get_tuner_gains", Gains),
            set_agc: symbol!("rtlsdr_set_agc_mode", SetInt), set_testmode: symbol!("rtlsdr_set_testmode", SetInt),
            set_direct: symbol!("rtlsdr_set_direct_sampling", SetInt), get_direct: symbol!("rtlsdr_get_direct_sampling", GetInt),
        })
    }
}
