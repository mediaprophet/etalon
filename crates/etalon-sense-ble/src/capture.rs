use std::{marker::PhantomData, rc::Rc};
use std::sync::{
    atomic::Ordering,
    Arc,
};

use etalon_sense_core::{
    adapter::{
        AdapterError, AuthorizedExtendedCapture, AuthorizedExtendedCaptureBuffer,
        ExtendedAdapterDescriptor, ExtendedCaptureSession, ExtendedCapturedSamples,
        ExtendedDiscoveredSource, ExtendedSensorAdapter,
    },
    authority::TrustedCaptureClock,
    quantity::SampleEncoding,
    time::ClockStamp,
    ContractError,
};
use windows::{
    Devices::Bluetooth::Advertisement::{
        BluetoothLEAdvertisementReceivedEventArgs, BluetoothLEAdvertisementWatcher,
        BluetoothLEAdvertisementWatcherStatus, BluetoothLEScanningMode,
    },
    Foundation::TypedEventHandler,
};

use crate::{
    config::{run_collection_gate, BleCollectionGate, BleCollectionScope, BleConfig},
    queue::{increment_dropped, CallbackState, StopSignal, QUEUE_CAPACITY},
    runtime::{spawn_stop_worker, BleRuntime, WinRtApartment},
    sample::raw_address_type,
    BleAdvertisementSample,
};

#[cfg(test)]
use crate::queue::FixedQueue;

/// Stateful passive BLE advertisement adapter. It is thread-affine because
/// its WinRT apartment is initialized on the discovery/start thread.
/// If `stop` returns a backend error, the session remains active but halted;
/// retry `stop` with that same session until teardown is confirmed.
pub struct BleRssiAdapter<C> {
    descriptor: ExtendedAdapterDescriptor,
    config: BleConfig,
    clock: C,
    gate: Arc<dyn BleCollectionGate>,
    source: ExtendedDiscoveredSource,
    discovered: bool,
    runtime: Option<BleRuntime>,
    shared: Option<Arc<CallbackState>>,
    active: Option<ExtendedCaptureSession>,
    active_scope: Option<BleCollectionScope>,
    next_session_id: u64,
    generation: u32,
    faulted: bool,
    last_records: [BleAdvertisementSample; QUEUE_CAPACITY],
    last_record_count: usize,
    last_capture_drop_interval: Option<(u64, u64)>,
    _thread_affinity: PhantomData<Rc<()>>,
}

impl<C: TrustedCaptureClock> BleRssiAdapter<C> {
    /// Construction performs no Bluetooth discovery or scanning.
    pub fn new(
        config: BleConfig,
        clock: C,
        gate: Arc<dyn BleCollectionGate>,
    ) -> Result<Self, AdapterError> {
        let descriptor = ExtendedAdapterDescriptor::new(
            config.adapter_id,
            config.capability.modality_id(),
        )?;
        let source = ExtendedDiscoveredSource::new(config.source_id, config.capability)?;
        Ok(Self {
            descriptor,
            config,
            clock,
            gate,
            source,
            discovered: false,
            runtime: None,
            shared: None,
            active: None,
            active_scope: None,
            next_session_id: 1,
            generation: 1,
            faulted: false,
            last_records: [BleAdvertisementSample::default(); QUEUE_CAPACITY],
            last_record_count: 0,
            last_capture_drop_interval: None,
            _thread_affinity: PhantomData,
        })
    }

    /// Records for the last successful capture, in payload sample order.
    pub fn last_capture_records(&self) -> &[BleAdvertisementSample] {
        &self.last_records[..self.last_record_count]
    }

    /// Number of callback records dropped since the current watcher started.
    pub fn dropped_advertisements(&self) -> u64 {
        self.shared.as_ref().map_or(0, |state| state.dropped.load(Ordering::Relaxed))
    }

    /// Cumulative dropped-counter snapshots immediately before and after the
    /// last successful queue drain. This is an interval around that drain,
    /// not an exact count attributable to its records: callback drops can
    /// increment the counter without holding the queue mutex.
    pub const fn last_capture_drop_count_interval(&self) -> Option<(u64, u64)> {
        self.last_capture_drop_interval
    }

    fn clear_last_capture(&mut self) {
        self.last_records.fill(BleAdvertisementSample::default());
        self.last_record_count = 0;
        self.last_capture_drop_interval = None;
    }

    fn halt_collection(&mut self) {
        if let Some(shared) = &self.shared { shared.halt(); }
        if let Some(runtime) = &self.runtime { runtime.stop_signal.request_stop(); }
        self.clear_last_capture();
    }

    /// Retry teardown immediately after a failed start. If the OS still does
    /// not confirm Stop/unsubscription, retain the runtime so Drop can retry.
    fn cleanup_failed_start(
        &mut self,
        watcher: Arc<BluetoothLEAdvertisementWatcher>,
        received_token: i64,
        handler: TypedEventHandler<
            BluetoothLEAdvertisementWatcher,
            BluetoothLEAdvertisementReceivedEventArgs,
        >,
        shared: Arc<CallbackState>,
        stop_signal: Arc<StopSignal>,
        apartment: WinRtApartment,
    ) -> bool {
        let mut runtime = BleRuntime::new(
            watcher, received_token, handler, Arc::clone(&shared), stop_signal, None, apartment,
        );
        if runtime.shutdown().is_ok() { return true; }
        self.shared = Some(shared);
        self.runtime = Some(runtime);
        self.faulted = true;
        false
    }
}

impl<C: TrustedCaptureClock> ExtendedSensorAdapter for BleRssiAdapter<C> {
    fn descriptor(&self) -> ExtendedAdapterDescriptor { self.descriptor }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> { self.clock.now() }

    fn discover_backend(
        &mut self,
        out: &mut [ExtendedDiscoveredSource],
    ) -> Result<usize, AdapterError> {
        if out.is_empty() { return Err(AdapterError::DiscoveryBufferTooSmall); }
        if self.active.is_some() { return Err(AdapterError::InvalidSession); }
        out[0] = self.source;
        self.discovered = true;
        Ok(1)
    }

    fn start_backend(
        &mut self,
        request: AuthorizedExtendedCapture,
    ) -> Result<ExtendedCaptureSession, AdapterError> {
        if !self.discovered || request.source() != self.source {
            return Err(AdapterError::AuthorizationDenied);
        }
        if self.runtime.is_some() || self.active.is_some() { return Err(AdapterError::InvalidSession); }
        if self.faulted { return Err(AdapterError::Unavailable); }
        request.validate_at(self.trusted_now()?)?;
        let scope = BleCollectionScope::from_request(&request);
        let mut checked = false;
        let mut check_start = || { checked = true; };
        let permitted = run_collection_gate(self.gate.as_ref(), scope, &mut check_start);
        drop(check_start);
        if !permitted || !checked {
            return Err(AdapterError::AuthorizationDenied);
        }
        let session_id = self.next_session_id;
        let next_session_id = session_id.checked_add(1).ok_or(AdapterError::BackendFailure)?;
        let session = ExtendedCaptureSession::new(
            self.descriptor.id(), self.source, session_id, self.generation,
        )?;

        let apartment = WinRtApartment::initialize()?;
        let watcher = Arc::new(
            BluetoothLEAdvertisementWatcher::new().map_err(|_| AdapterError::Unavailable)?,
        );
        watcher.SetScanningMode(BluetoothLEScanningMode::Passive)
            .map_err(|_| AdapterError::Unavailable)?;

        let shared = Arc::new(CallbackState::new(self.config.max_session_observations()));
        let stop_signal = Arc::new(StopSignal::new());
        let callback_state = Arc::clone(&shared);
        let callback_signal = Arc::clone(&stop_signal);
        let callback_gate = Arc::clone(&self.gate);
        let callback_scope = scope;
        let handler = TypedEventHandler::<
            BluetoothLEAdvertisementWatcher,
            BluetoothLEAdvertisementReceivedEventArgs,
        >::new(move |_, args| {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if !callback_signal.is_started() {
                    increment_dropped(&callback_state.dropped);
                    return;
                }
                if !callback_state.is_collecting() {
                    increment_dropped(&callback_state.dropped);
                    return;
                }
                let Ok(_exclusive) = callback_state.callback_gate.try_lock() else {
                    increment_dropped(&callback_state.dropped);
                    return;
                };
                if !callback_state.is_collecting() {
                    increment_dropped(&callback_state.dropped);
                    return;
                }
                let mut collected = false;
                let mut collect = || {
                    if collected || !callback_signal.is_started() || !callback_state.is_collecting() {
                        return;
                    }
                    collected = true;
                    let Some(args) = args.as_ref() else {
                        increment_dropped(&callback_state.dropped);
                        return;
                    };
                    let (Ok(address), Ok(rssi_dbm), Ok(timestamp)) = (
                        args.BluetoothAddress(),
                        args.RawSignalStrengthInDBm(),
                        args.Timestamp(),
                    ) else {
                        increment_dropped(&callback_state.dropped);
                        return;
                    };
                    let sample = BleAdvertisementSample {
                        bluetooth_address: address,
                        rssi_dbm,
                        os_timestamp_100ns: timestamp.UniversalTime,
                        os_address_type: raw_address_type(args.BluetoothAddressType()),
                    };
                    let accepted = match callback_state.queue.try_lock() {
                        Ok(mut queue) => queue.push(sample),
                        Err(_) => false,
                    };
                    if !accepted {
                        increment_dropped(&callback_state.dropped);
                    } else if callback_state.note_accepted() {
                        callback_state.close_collection();
                        callback_signal.request_stop();
                    }
                };
                let permitted = run_collection_gate(callback_gate.as_ref(), callback_scope, &mut collect);
                drop(collect);
                if !permitted {
                    callback_state.halt_from_callback();
                    callback_signal.request_stop();
                } else if !collected {
                    if callback_signal.is_started() && callback_state.is_collecting() {
                        callback_state.halt_from_callback();
                        callback_signal.request_stop();
                    } else {
                        increment_dropped(&callback_state.dropped);
                    }
                }
            }));
            if outcome.is_err() {
                callback_state.halt();
                callback_signal.request_stop();
            }
            Ok(())
        });
        let received_token = watcher.Received(&handler).map_err(|_| AdapterError::BackendFailure)?;
        let stop_worker = match spawn_stop_worker(
            Arc::clone(&watcher),
            Arc::clone(&stop_signal),
            Arc::clone(&shared),
            Arc::clone(&self.gate),
            scope,
            self.config.max_session_duration_ms(),
        ) {
            Ok(worker) => worker,
            Err(error) => {
                let _ = watcher.RemoveReceived(received_token);
                return Err(error);
            }
        };
        let start_result = stop_signal.start_with(|| {
            let mut attempted = false;
            let mut os_started = false;
            let mut start = || {
                if stop_signal.requested() { return; }
                attempted = true;
                os_started = watcher.Start().is_ok();
            };
            let permitted = run_collection_gate(self.gate.as_ref(), scope, &mut start);
            drop(start);
            (os_started, (permitted, attempted, os_started))
        });
        let (os_started, (permitted, attempted, start_ok)) = match start_result {
            Ok(result) => result,
            Err(_) => {
                stop_signal.request_stop();
                let worker_stopped = stop_worker.join().unwrap_or(Err(AdapterError::BackendFailure)).is_ok();
                let cleaned = self.cleanup_failed_start(
                    watcher, received_token, handler, shared, stop_signal, apartment,
                );
                return Err(if worker_stopped && cleaned {
                    AdapterError::AuthorizationDenied
                } else {
                    AdapterError::Unavailable
                });
            }
        };
        if !permitted || !attempted || !start_ok || !os_started || stop_signal.requested() {
            stop_signal.request_stop();
            let worker_stopped = stop_worker.join().unwrap_or(Err(AdapterError::BackendFailure)).is_ok();
            let cleaned = self.cleanup_failed_start(
                watcher, received_token, handler, shared, stop_signal, apartment,
            );
            return Err(if worker_stopped && cleaned && !(attempted && !start_ok) {
                AdapterError::AuthorizationDenied
            } else {
                AdapterError::Unavailable
            });
        }

        self.next_session_id = next_session_id;
        self.shared = Some(Arc::clone(&shared));
        self.runtime = Some(BleRuntime::new(
            watcher, received_token, handler, shared, stop_signal, Some(stop_worker), apartment,
        ));
        self.active = Some(session);
        self.active_scope = Some(scope);
        self.clear_last_capture();
        Ok(session)
    }

    fn capture_backend<'a>(
        &mut self,
        mut output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<ExtendedCapturedSamples<'a>, AdapterError> {
        let session = output.session();
        if self.active != Some(session) || !session.matches(self.descriptor, output.source()) {
            return Err(AdapterError::InvalidSession);
        }
        if self.faulted { return Err(AdapterError::Unavailable); }
        if self.shared.as_ref().is_some_and(|state| state.halted.load(Ordering::Acquire)) {
            self.clear_last_capture();
            return Err(AdapterError::Unavailable);
        }
        let layout = output.layout();
        if layout.rank() != 1
            || *layout.encoding() != SampleEncoding::I16
            || layout.strides_bytes()[0] != 2
        {
            return Err(AdapterError::Unsupported);
        }
        let count = usize::try_from(layout.extents()[0])
            .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        if count == 0 || count > self.config.max_batch_samples() || count > QUEUE_CAPACITY {
            return Err(AdapterError::Unsupported);
        }
        self.clear_last_capture();
        if let Err(error) = self.clock.now().and_then(|now| output.validate_at(now)) {
            self.halt_collection();
            return Err(error);
        }

        let watcher_status = self.runtime.as_ref()
            .ok_or(AdapterError::Unavailable)?
            .watcher.Status();
        let collection_closed = self.shared.as_ref()
            .is_some_and(|state| state.collection_closed.load(Ordering::Acquire));
        if !matches!(watcher_status, Ok(status) if status == BluetoothLEAdvertisementWatcherStatus::Started)
            && !collection_closed
        {
            self.faulted = true;
            self.halt_collection();
            return Err(AdapterError::Unavailable);
        }

        let scope = match self.active_scope {
            Some(scope) => scope,
            None => {
                self.halt_collection();
                return Err(AdapterError::AuthorizationDenied);
            }
        };
        let runtime = self.runtime.as_ref().ok_or(AdapterError::Unavailable)?;
        let shared = Arc::clone(&runtime.shared);
        let stop_signal = Arc::clone(&runtime.stop_signal);
        let gate = Arc::clone(&self.gate);
        let mut batch = [BleAdvertisementSample::default(); QUEUE_CAPACITY];
        let mut collected = false;
        let mut capture_result = Err(AdapterError::Unavailable);
        let mut drop_count_interval = None;
        let mut collect = || {
            if collected { return; }
            collected = true;
            let mut queue = match shared.queue.try_lock() {
                Ok(queue) => queue,
                Err(_) => {
                    capture_result = Err(AdapterError::BackendFailure);
                    return;
                }
            };
            let dropped_before = shared.dropped.load(Ordering::Relaxed);
            if !queue.pop_exact(&mut batch[..count]) { return; }
            let dropped_after = shared.dropped.load(Ordering::Relaxed);
            drop_count_interval = Some((dropped_before, dropped_after));
            drop(queue);
            for (sample, chunk) in batch[..count].iter().zip(output.writable_bytes().chunks_exact_mut(2)) {
                chunk.copy_from_slice(&sample.rssi_dbm.to_le_bytes());
            }
            capture_result = Ok(());
        };
        let permitted = run_collection_gate(gate.as_ref(), scope, &mut collect);
        drop(collect);
        if !permitted || !collected {
            output.writable_bytes().fill(0);
            shared.halt();
            stop_signal.request_stop();
            return Err(AdapterError::AuthorizationDenied);
        }
        if let Err(error) = capture_result { return Err(error); }
        self.last_records[..count].copy_from_slice(&batch[..count]);
        self.last_record_count = count;
        self.last_capture_drop_interval = drop_count_interval;
        Ok(output.complete())
    }

    fn stop_backend(&mut self, session: ExtendedCaptureSession) -> Result<(), AdapterError> {
        if self.active != Some(session) { return Err(AdapterError::InvalidSession); }
        self.halt_collection();
        let stop_result = self.runtime.as_mut()
            .map_or(Ok(()), BleRuntime::shutdown);
        if stop_result.is_err() {
            // Keep the same session and runtime so the caller can retry Stop.
            self.faulted = true;
            return Err(AdapterError::BackendFailure);
        }
        self.runtime = None;
        self.active = None;
        self.active_scope = None;
        match self.generation.checked_add(1) {
            Some(generation) => {
                self.generation = generation;
                self.faulted = false;
                Ok(())
            }
            None => {
                self.faulted = true;
                Err(AdapterError::BackendFailure)
            }
        }
    }
}

#[cfg(test)]
#[path = "capture_tests.rs"]
mod tests;
