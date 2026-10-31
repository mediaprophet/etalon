use crate::{
    abi::{Api, Dev},
    runtime::{mock_runtime, DeploymentLease},
    *,
};
use core::ffi::{c_char, c_void};
use std::cell::Cell;

std::thread_local! {
    static OPENS: Cell<u32> = const { Cell::new(0) };
    static CLOSES: Cell<u32> = const { Cell::new(0) };
    static READS: Cell<u32> = const { Cell::new(0) };
    static CLOSE_STATUS: Cell<i32> = const { Cell::new(0) };
    static READ_COUNT: Cell<i32> = const { Cell::new(4) };
    static RATE: Cell<u32> = const { Cell::new(1_000_000) };
    static CENTER: Cell<u32> = const { Cell::new(100_000_000) };
    static PPM: Cell<i32> = const { Cell::new(0) };
    static GAIN: Cell<i32> = const { Cell::new(0) };
}

struct Lease(Cell<bool>);
impl DeploymentLease for Lease {
    fn current(&self) -> bool {
        self.0.get()
    }
}
struct Gate(Option<Stage>);
impl RtlSdrGate for Gate {
    fn allow(&self, context: &GateContext<'_>) -> bool {
        self.0 != Some(context.stage)
    }
}

unsafe extern "C" fn open(pointer: *mut *mut Dev, index: u32) -> i32 {
    assert_eq!(index, 3);
    OPENS.with(|value| value.set(value.get() + 1));
    unsafe {
        pointer.write(core::ptr::NonNull::<Dev>::dangling().as_ptr());
    }
    0
}
unsafe extern "C" fn close(_: *mut Dev) -> i32 {
    CLOSES.with(|value| value.set(value.get() + 1));
    CLOSE_STATUS.with(Cell::get)
}
unsafe extern "C" fn read(_: *mut Dev, output: *mut c_void, length: i32, count: *mut i32) -> i32 {
    READS.with(|value| value.set(value.get() + 1));
    unsafe {
        core::ptr::write_bytes(output.cast::<u8>(), 0xaa, length as usize);
        count.write(READ_COUNT.with(Cell::get));
    }
    0
}
unsafe extern "C" fn strings(
    _: *mut Dev,
    manufacturer: *mut c_char,
    product: *mut c_char,
    serial: *mut c_char,
) -> i32 {
    for output in [manufacturer, product, serial] {
        unsafe {
            output.write(b'A' as c_char);
            output.add(1).write(0);
        }
    }
    0
}
unsafe extern "C" fn set_center(_: *mut Dev, value: u32) -> i32 {
    CENTER.with(|cell| cell.set(value));
    0
}
unsafe extern "C" fn get_center(_: *mut Dev) -> u32 {
    CENTER.with(Cell::get)
}
unsafe extern "C" fn set_rate(_: *mut Dev, value: u32) -> i32 {
    RATE.with(|cell| cell.set(value));
    0
}
unsafe extern "C" fn get_rate(_: *mut Dev) -> u32 {
    RATE.with(Cell::get)
}
unsafe extern "C" fn set_ppm(_: *mut Dev, value: i32) -> i32 {
    PPM.with(|cell| cell.set(value));
    0
}
unsafe extern "C" fn get_ppm(_: *mut Dev) -> i32 {
    PPM.with(Cell::get)
}
unsafe extern "C" fn set_gain(_: *mut Dev, value: i32) -> i32 {
    GAIN.with(|cell| cell.set(value));
    0
}
unsafe extern "C" fn get_gain(_: *mut Dev) -> i32 {
    GAIN.with(Cell::get)
}
unsafe extern "C" fn gains(_: *mut Dev, output: *mut i32) -> i32 {
    if !output.is_null() {
        for (index, gain) in [-10, 0, 10].iter().enumerate() {
            unsafe {
                output.add(index).write(*gain);
            }
        }
    }
    3
}
unsafe extern "C" fn set_int(_: *mut Dev, _: i32) -> i32 {
    0
}
unsafe extern "C" fn get_zero(_: *mut Dev) -> i32 {
    0
}
unsafe extern "C" fn reset(_: *mut Dev) -> i32 {
    0
}

fn api() -> Api {
    OPENS.with(|v| v.set(0));
    CLOSES.with(|v| v.set(0));
    READS.with(|v| v.set(0));
    CLOSE_STATUS.with(|v| v.set(0));
    READ_COUNT.with(|v| v.set(4));
    Api {
        open,
        close,
        read,
        reset,
        usb_strings: strings,
        set_center,
        get_center,
        set_rate,
        get_rate,
        set_correction: set_ppm,
        get_correction: get_ppm,
        set_gain_mode: set_int,
        set_gain,
        get_gain,
        get_gains: gains,
        set_agc: set_int,
        set_testmode: set_int,
        set_direct: set_int,
        get_direct: get_zero,
    }
}
fn config() -> Config {
    Config::new(100_000_000, 1_000_000, 0, GainMode::Automatic, false).unwrap()
}

#[test]
fn dangerous_c_control_extremes_are_rejected_before_native_calls() {
    assert!(Config::new(1, 1_000_000, i32::MIN, GainMode::Automatic, false).is_err());
    assert!(Config::new(1, 1_000_000, 0, GainMode::ManualTenthsDb(i32::MIN), false).is_err());
    let lease = Lease(Cell::new(true));
    let runtime = mock_runtime(&lease, api());
    let mut device = runtime.open(&Gate(None)).unwrap();
    assert_eq!(
        device.configure(
            Config::new(
                100_000_000,
                1_000_000,
                0,
                GainMode::ManualTenthsDb(99),
                false
            )
            .unwrap(),
            &Gate(None)
        ),
        Err(Error::InvalidConfig)
    );
    assert_eq!(CLOSES.with(Cell::get), 1);
}

#[test]
fn forgotten_device_keeps_native_runtime_owned_without_relying_on_drop() {
    let lease = Lease(Cell::new(true));
    let runtime = mock_runtime(&lease, api());
    let device = runtime.open(&Gate(None)).unwrap();
    core::mem::forget(device);
    assert!(runtime.must_retain());
    assert_eq!(CLOSES.with(Cell::get), 0);
    drop(runtime);
    assert_eq!(CLOSES.with(Cell::get), 0);
}

#[test]
fn denied_post_read_wipes_and_closes_once() {
    let lease = Lease(Cell::new(true));
    let runtime = mock_runtime(&lease, api());
    let mut device = runtime.open(&Gate(None)).unwrap();
    device.configure(config(), &Gate(None)).unwrap();
    let mut output = [9; 4];
    assert_eq!(
        device
            .read_into(&Gate(Some(Stage::PostRead)), &mut output)
            .unwrap_err(),
        Error::NotAuthorized
    );
    assert_eq!(output, [0; 4]);
    assert_eq!(READS.with(Cell::get), 1);
    assert_eq!(CLOSES.with(Cell::get), 1);
    drop(device);
    assert_eq!(CLOSES.with(Cell::get), 1);
    assert!(!runtime.must_retain());
}

#[test]
fn partial_read_and_unknown_cleanup_never_publish_or_retry() {
    let lease = Lease(Cell::new(true));
    let runtime = mock_runtime(&lease, api());
    let mut device = runtime.open(&Gate(None)).unwrap();
    device.configure(config(), &Gate(None)).unwrap();
    READ_COUNT.with(|v| v.set(3));
    CLOSE_STATUS.with(|v| v.set(-7));
    let mut output = [9; 4];
    assert!(matches!(
        device.read_into(&Gate(None), &mut output),
        Err(Error::CleanupFailed {
            cleanup_code: -7,
            ..
        })
    ));
    assert_eq!(output, [0; 4]);
    assert!(runtime.must_retain());
    assert_eq!(device.close(), Err(Error::CloseOutcomeUnknown));
    drop(device);
    assert_eq!(CLOSES.with(Cell::get), 1);
}

#[test]
fn revoked_or_changed_configuration_never_reaches_native_read() {
    let lease = Lease(Cell::new(true));
    let runtime = mock_runtime(&lease, api());
    let mut device = runtime.open(&Gate(None)).unwrap();
    device.configure(config(), &Gate(None)).unwrap();
    CENTER.with(|v| v.set(101_000_000));
    let mut output = [9; 4];
    assert_eq!(
        device.read_into(&Gate(None), &mut output).unwrap_err(),
        Error::ConfigurationChanged
    );
    assert_eq!(READS.with(Cell::get), 0);
    assert_eq!(output, [0; 4]);
}

#[test]
fn successful_native_read_keeps_exact_receipt_and_raw_storage_payload() {
    let lease = Lease(Cell::new(true));
    let runtime = mock_runtime(&lease, api());
    let mut device = runtime.open(&Gate(None)).unwrap();
    device.configure(config(), &Gate(None)).unwrap();
    let mut raw = [0; 4];
    let report = device.read_into(&Gate(None), &mut raw).unwrap();
    assert_eq!(report.target_id(), 7);
    assert_eq!(report.device_index(), 3);
    assert_eq!(report.pair_count(), 2);
    assert_eq!(report.sequence(), 0);
    assert_eq!(raw, [0xaa; 4]);
    let mut encoded = [0; capture_payload::HEADER_BYTES + 4];
    let budget = u64::try_from(encoded.len()).unwrap();
    let count =
        capture_payload::encode_capture_payload(&report, &raw, budget, &mut encoded).unwrap();
    let historical = capture_payload::decode_capture_payload(&encoded[..count], budget).unwrap();
    assert_eq!(historical.target_id, report.target_id());
    assert_eq!(historical.generation, report.generation());
    assert_eq!(historical.host_read_sequence, report.sequence());
    assert_eq!(historical.requested, report.requested_config());
    assert_eq!(
        historical.usb_strings.serial,
        report.usb_identity().serial()
    );
    assert_eq!(historical.raw_iq, raw);
    device.close().unwrap();
    assert_eq!(CLOSES.with(Cell::get), 1);
    assert!(!runtime.must_retain());
}
