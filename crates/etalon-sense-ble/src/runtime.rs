use core::marker::PhantomData;
use std::{
    rc::Rc,
    sync::{mpsc, Arc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use etalon_sense_core::adapter::AdapterError;
use windows::{
    Devices::Bluetooth::Advertisement::{
        BluetoothLEAdvertisementReceivedEventArgs, BluetoothLEAdvertisementWatcher,
        BluetoothLEAdvertisementWatcherStatus,
    },
    Foundation::TypedEventHandler,
    Win32::System::WinRT::{RoInitialize, RoUninitialize, RO_INIT_MULTITHREADED},
};

use crate::{
    config::{run_collection_gate, BleCollectionGate, BleCollectionScope},
    queue::{CallbackState, StopSignal},
};

pub(super) struct BleRuntime {
    pub(super) watcher: Arc<BluetoothLEAdvertisementWatcher>,
    received_token: i64,
    _handler: TypedEventHandler<
        BluetoothLEAdvertisementWatcher,
        BluetoothLEAdvertisementReceivedEventArgs,
    >,
    pub(super) shared: Arc<CallbackState>,
    pub(super) stop_signal: Arc<StopSignal>,
    stop_worker: Option<JoinHandle<Result<(), AdapterError>>>,
    stopped: bool,
    // Keep the adapter-thread WinRT apartment through watcher/delegate teardown.
    _apartment: WinRtApartment,
}

impl BleRuntime {
    pub(super) fn new(
        watcher: Arc<BluetoothLEAdvertisementWatcher>,
        received_token: i64,
        handler: TypedEventHandler<
            BluetoothLEAdvertisementWatcher,
            BluetoothLEAdvertisementReceivedEventArgs,
        >,
        shared: Arc<CallbackState>,
        stop_signal: Arc<StopSignal>,
        stop_worker: Option<JoinHandle<Result<(), AdapterError>>>,
        apartment: WinRtApartment,
    ) -> Self {
        Self {
            watcher,
            received_token,
            _handler: handler,
            shared,
            stop_signal,
            stop_worker,
            stopped: false,
            _apartment: apartment,
        }
    }

    pub(super) fn shutdown(&mut self) -> Result<(), AdapterError> {
        if self.stopped {
            return Ok(());
        }
        self.shared.halt();
        self.stop_signal.request_stop();
        let stopped = match self.stop_worker.take() {
            Some(worker) => worker.join().unwrap_or(Err(AdapterError::BackendFailure)),
            None => self
                .stop_signal
                .stop_with(|| stop_watcher_bounded(&self.watcher).is_ok())
                .map_err(|_| AdapterError::BackendFailure)
                .and_then(|success| {
                    if success {
                        Ok(())
                    } else {
                        Err(AdapterError::BackendFailure)
                    }
                }),
        };
        let unsubscribed = self.watcher.RemoveReceived(self.received_token);
        let queue_cleared = self.shared.quiesce_and_clear().is_ok();
        self.stopped = stopped.is_ok() && unsubscribed.is_ok() && queue_cleared;
        if self.stopped {
            Ok(())
        } else {
            Err(AdapterError::BackendFailure)
        }
    }
}

impl Drop for BleRuntime {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = self.shutdown();
        }
    }
}

pub(super) fn spawn_stop_worker(
    watcher: Arc<BluetoothLEAdvertisementWatcher>,
    signal: Arc<StopSignal>,
    shared: Arc<CallbackState>,
    gate: Arc<dyn BleCollectionGate>,
    scope: BleCollectionScope,
    max_session_duration_ms: u32,
) -> Result<JoinHandle<Result<(), AdapterError>>, AdapterError> {
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let worker = thread::Builder::new()
        .name("etalon-ble-stop".into())
        .spawn(move || {
            let apartment = match WinRtApartment::initialize() {
                Ok(apartment) => {
                    let _ = ready_tx.send(Ok(()));
                    apartment
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return Err(error);
                }
            };
            let started_at = Instant::now();
            loop {
                if signal.requested() {
                    break;
                }
                if started_at.elapsed() >= Duration::from_millis(u64::from(max_session_duration_ms))
                {
                    shared.close_collection();
                    signal.request_stop();
                    break;
                }
                let mut checked = false;
                let mut check_policy = || {
                    checked = true;
                };
                let permitted = run_collection_gate(gate.as_ref(), scope, &mut check_policy);
                drop(check_policy);
                if !permitted || !checked {
                    shared.halt();
                    signal.request_stop();
                    break;
                }
                if !shared.is_collecting() {
                    signal.request_stop();
                    break;
                }
                if signal.wait_slice().is_err() {
                    return Err(AdapterError::BackendFailure);
                }
            }
            let result = signal
                .stop_with(|| stop_watcher_bounded(&watcher).is_ok())
                .map_err(|_| AdapterError::BackendFailure)
                .and_then(|success| {
                    if success {
                        Ok(())
                    } else {
                        Err(AdapterError::BackendFailure)
                    }
                });
            drop(apartment);
            result
        })
        .map_err(|_| AdapterError::Unavailable)?;

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(worker),
        Ok(Err(error)) => {
            let _ = worker.join();
            Err(error)
        }
        Err(_) => {
            signal.request_stop();
            let _ = worker.join();
            Err(AdapterError::Unavailable)
        }
    }
}

fn stop_watcher_bounded(watcher: &BluetoothLEAdvertisementWatcher) -> Result<(), AdapterError> {
    match watcher.Status() {
        Ok(status) if is_stopped(status) => return Ok(()),
        _ => {}
    }
    watcher.Stop().map_err(|_| AdapterError::BackendFailure)?;
    for _ in 0..100 {
        match watcher.Status() {
            Ok(status) if is_stopped(status) => return Ok(()),
            Ok(_) => thread::sleep(std::time::Duration::from_millis(10)),
            Err(_) => return Err(AdapterError::BackendFailure),
        }
    }
    Err(AdapterError::BackendFailure)
}

fn is_stopped(status: BluetoothLEAdvertisementWatcherStatus) -> bool {
    matches!(
        status,
        BluetoothLEAdvertisementWatcherStatus::Created
            | BluetoothLEAdvertisementWatcherStatus::Stopped
            | BluetoothLEAdvertisementWatcherStatus::Aborted
    )
}

pub(super) struct WinRtApartment(PhantomData<Rc<()>>);

impl WinRtApartment {
    pub(super) fn initialize() -> Result<Self, AdapterError> {
        unsafe { RoInitialize(RO_INIT_MULTITHREADED) }.map_err(|_| AdapterError::Unavailable)?;
        Ok(Self(PhantomData))
    }
}

impl Drop for WinRtApartment {
    fn drop(&mut self) {
        unsafe { RoUninitialize() }
    }
}
