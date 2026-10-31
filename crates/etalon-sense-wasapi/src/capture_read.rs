use super::*;

impl<C: TrustedCaptureClock> WasapiCaptureAdapter<'_, C> {
    pub(super) fn capture_impl<'a>(
        &mut self,
        output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<(ExtendedCapturedSamples<'a>, CaptureMetadata), AdapterError> {
        let mut output = CaptureOutput::new(output);
        let session = output.session();
        if self.active != Some(session) || !session.matches(self.descriptor, output.source()) {
            return Err(AdapterError::InvalidSession);
        }
        if Some(output.source()) != self.source {
            return Err(AdapterError::AuthorizationDenied);
        }
        self.last_metadata = None;
        if self.faulted {
            return Err(AdapterError::Unavailable);
        }
        let dispatch_time = match self.trusted_now() {
            Ok(stamp) => stamp,
            Err(error) => {
                self.fail_closed_capture();
                return Err(error);
            }
        };
        if let Err(error) = output.validate_at(dispatch_time) {
            self.fail_closed_capture();
            return Err(error);
        }
        let runtime = self.runtime.as_ref().ok_or(AdapterError::Unavailable)?;
        let format = runtime.format;
        validate_layout(output.layout(), format, self.config.max_frame_bytes)?;
        let requested_frames = output.layout().extents()[0];
        let target_bytes = usize::try_from(output.layout().required_byte_span()?)
            .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        let mut copied_frames = 0u32;
        let mut stalled_timeout_count = 0u8;
        let mut metadata = CaptureMetadata {
            format,
            host_clock_at_dispatch: dispatch_time,
            first_device_position: None,
            first_qpc_position: None,
            frames: requested_frames,
            silent_frames: 0,
            data_discontinuity: false,
            timestamp_error: false,
        };

        while copied_frames < requested_frames {
            let now = match self.check_capture_request(&output, dispatch_time) {
                Ok(now) => now,
                Err(error) => {
                    clear_prefix(&mut output, copied_frames, format.block_align());
                    self.fail_closed_capture();
                    return Err(error);
                }
            };
            if !self.remainder.is_empty() {
                let offset = (copied_frames as usize)
                    .checked_mul(usize::from(format.block_align()))
                    .ok_or(AdapterError::BackendFailure)?;
                let drained = self.remainder.drain(
                    &mut output.writable_bytes()[offset..target_bytes],
                    usize::from(format.block_align()),
                );
                let (frames, packet) = match drained {
                    Ok(value) => value,
                    Err(error) => {
                        self.fail_closed_capture();
                        return Err(error);
                    }
                };
                if let Err(error) =
                    merge_packet_metadata(&mut metadata, packet, frames as u32, copied_frames == 0)
                {
                    self.fail_closed_capture();
                    return Err(error);
                }
                copied_frames += frames as u32;
                if let Err(error) = self.check_capture_request(&output, dispatch_time) {
                    self.fail_closed_capture();
                    return Err(error);
                }
                continue;
            }
            let wait_timeout_ms = match capture_wait_timeout_ms(
                dispatch_time,
                now,
                self.config.max_capture_duration_ms,
                self.config.wait_timeout_ms,
            ) {
                Ok(timeout) => timeout,
                Err(error) => {
                    clear_prefix(&mut output, copied_frames, format.block_align());
                    self.fail_closed_capture();
                    return Err(error);
                }
            };
            let runtime = self.runtime.as_ref().ok_or(AdapterError::Unavailable)?;
            let queued = match unsafe { runtime.capture.GetNextPacketSize() } {
                Ok(count) => count,
                Err(_) => {
                    clear_prefix(&mut output, copied_frames, format.block_align());
                    self.fail_closed_capture();
                    return Err(AdapterError::BackendFailure);
                }
            };
            // Drain already queued packets before waiting for a new event.
            let wait = if queued == 0 {
                unsafe { WaitForSingleObject(runtime.event.0, wait_timeout_ms) }
            } else {
                WAIT_OBJECT_0
            };
            if wait == WAIT_TIMEOUT {
                // Re-read the host monotonic clock instead of waiting forever
                // on a stalled endpoint. Repeated OS timeouts without clock
                // progress also terminate the call fail-closed.
                let after_wait = match self.trusted_now() {
                    Ok(stamp) => stamp,
                    Err(error) => {
                        clear_prefix(&mut output, copied_frames, format.block_align());
                        self.fail_closed_capture();
                        return Err(error);
                    }
                };
                match timeout_clock_advanced(now, after_wait) {
                    Ok(true) => stalled_timeout_count = 0,
                    Ok(false) => stalled_timeout_count = stalled_timeout_count.saturating_add(1),
                    Err(error) => {
                        clear_prefix(&mut output, copied_frames, format.block_align());
                        self.fail_closed_capture();
                        return Err(error);
                    }
                }
                if stalled_timeout_count >= 2 {
                    clear_prefix(&mut output, copied_frames, format.block_align());
                    self.fail_closed_capture();
                    return Err(AdapterError::BackendFailure);
                }
                continue;
            }
            if wait != WAIT_OBJECT_0 {
                clear_prefix(&mut output, copied_frames, format.block_align());
                self.fail_closed_capture();
                return Err(AdapterError::BackendFailure);
            }
            if let Err(error) = self.check_capture_request(&output, dispatch_time) {
                clear_prefix(&mut output, copied_frames, format.block_align());
                self.fail_closed_capture();
                return Err(error);
            }
            let runtime = self.runtime.as_ref().ok_or(AdapterError::Unavailable)?;
            let available = match unsafe { runtime.capture.GetNextPacketSize() } {
                Ok(available) => available,
                Err(_) => {
                    clear_prefix(&mut output, copied_frames, format.block_align());
                    self.fail_closed_capture();
                    return Err(AdapterError::BackendFailure);
                }
            };
            if available == 0 {
                continue;
            }
            let remaining_frames = requested_frames - copied_frames;
            if !packet_fits_whole(available, remaining_frames) && self.remainder.capacity() == 0 {
                // Without remainder scratch the packet stays untouched. This
                // error faults capture; stop/restart with sufficient scratch
                // or a whole-packet request before retrying.
                clear_prefix(&mut output, copied_frames, format.block_align());
                if copied_frames > 0 {
                    self.fail_closed_capture();
                }
                return Err(AdapterError::Unsupported);
            }
            let mut data = ptr::null_mut();
            let mut packet_frames = 0u32;
            let mut flags = 0u32;
            let mut device_position = 0u64;
            let mut qpc_position = 0u64;
            let got_buffer = unsafe {
                runtime.capture.GetBuffer(
                    &mut data,
                    &mut packet_frames,
                    &mut flags,
                    Some(&mut device_position),
                    Some(&mut qpc_position),
                )
            };
            if got_buffer.is_err() {
                clear_prefix(&mut output, copied_frames, format.block_align());
                self.fail_closed_capture();
                return Err(AdapterError::BackendFailure);
            }
            let mut packet = PacketLease {
                capture: runtime.capture.clone(),
                frames: packet_frames,
                active: true,
            };
            if packet_frames == 0 {
                let _ = packet.release();
                clear_prefix(&mut output, copied_frames, format.block_align());
                self.fail_closed_capture();
                return Err(AdapterError::BackendFailure);
            }
            let take_frames = packet_frames.min(remaining_frames);
            let byte_count = match usize::try_from(take_frames)
                .ok()
                .and_then(|frames| frames.checked_mul(usize::from(format.block_align())))
            {
                Some(bytes) if bytes <= target_bytes => bytes,
                _ => {
                    let _ = packet.release();
                    clear_all(&mut output);
                    self.fail_closed_capture();
                    return Err(AdapterError::BackendFailure);
                }
            };
            let tail_frames = packet_frames - take_frames;
            let tail_bytes =
                match (tail_frames as usize).checked_mul(usize::from(format.block_align())) {
                    Some(bytes) if bytes <= self.remainder.capacity() => bytes,
                    _ => {
                        let _ = packet.release();
                        self.fail_closed_capture();
                        return Err(AdapterError::Unsupported);
                    }
                };
            if !byte_count
                .checked_add(tail_bytes)
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
            {
                let _ = packet.release();
                self.fail_closed_capture();
                return Err(AdapterError::BackendFailure);
            }
            let destination_offset = match usize::try_from(copied_frames)
                .ok()
                .and_then(|frames| frames.checked_mul(usize::from(format.block_align())))
            {
                Some(offset)
                    if offset
                        .checked_add(byte_count)
                        .is_some_and(|end| end <= target_bytes) =>
                {
                    offset
                }
                _ => {
                    let _ = packet.release();
                    clear_all(&mut output);
                    self.fail_closed_capture();
                    return Err(AdapterError::BackendFailure);
                }
            };
            let silent = flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0;
            if silent {
                output.writable_bytes()[destination_offset..destination_offset + byte_count]
                    .fill(0);
            } else if data.is_null() {
                let _ = packet.release();
                clear_all(&mut output);
                self.fail_closed_capture();
                return Err(AdapterError::BackendFailure);
            } else {
                unsafe {
                    ptr::copy_nonoverlapping(
                        data,
                        output.writable_bytes().as_mut_ptr().add(destination_offset),
                        byte_count,
                    )
                };
            }
            if tail_bytes != 0 {
                let tail_metadata = PacketMetadata {
                    device_position: device_position.checked_add(u64::from(take_frames)),
                    qpc_position: None,
                    silent,
                    discontinuity: false,
                    timestamp_error: flags & (AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32) != 0,
                };
                let staged = if silent {
                    self.remainder.stage_silence(tail_bytes, tail_metadata)
                } else {
                    let tail =
                        unsafe { core::slice::from_raw_parts(data.add(byte_count), tail_bytes) };
                    self.remainder.stage(tail, tail_metadata)
                };
                if let Err(error) = staged {
                    let _ = packet.release();
                    self.fail_closed_capture();
                    return Err(error);
                }
            }
            let released = packet.release();
            if released.is_err() {
                clear_all(&mut output);
                self.fail_closed_capture();
                return Err(AdapterError::BackendFailure);
            }
            let packet_metadata = PacketMetadata {
                device_position: Some(device_position),
                qpc_position: Some(qpc_position),
                silent,
                discontinuity: flags & (AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32) != 0,
                timestamp_error: flags & (AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32) != 0,
            };
            if let Err(error) = merge_packet_metadata(
                &mut metadata,
                packet_metadata,
                take_frames,
                copied_frames == 0,
            ) {
                self.fail_closed_capture();
                return Err(error);
            }
            copied_frames = match copied_frames.checked_add(take_frames) {
                Some(total) => total,
                None => {
                    clear_all(&mut output);
                    self.fail_closed_capture();
                    return Err(AdapterError::BackendFailure);
                }
            };
            if let Err(error) = self.check_capture_request(&output, dispatch_time) {
                clear_prefix(&mut output, copied_frames, format.block_align());
                self.fail_closed_capture();
                return Err(error);
            }
        }
        crate::observation::validate_produced_batch(
            output.layout(),
            target_bytes,
            metadata,
            format,
        )
        .map_err(|_| AdapterError::BackendFailure)?;
        self.last_metadata = Some(metadata);
        Ok((output.finish(), metadata))
    }
}

struct PacketLease {
    capture: IAudioCaptureClient,
    frames: u32,
    active: bool,
}
impl PacketLease {
    fn release(&mut self) -> Result<(), AdapterError> {
        self.active = false;
        unsafe { self.capture.ReleaseBuffer(self.frames) }.map_err(|_| AdapterError::BackendFailure)
    }
}
impl Drop for PacketLease {
    fn drop(&mut self) {
        if self.active {
            let _ = self.release();
        }
    }
}

fn merge_packet_metadata(
    metadata: &mut CaptureMetadata,
    packet: PacketMetadata,
    frames: u32,
    first: bool,
) -> Result<(), AdapterError> {
    if first {
        metadata.first_device_position = packet.device_position;
        metadata.first_qpc_position = packet.qpc_position;
    }
    if packet.silent {
        metadata.silent_frames = metadata
            .silent_frames
            .checked_add(frames)
            .ok_or(AdapterError::BackendFailure)?;
    }
    metadata.data_discontinuity |= packet.discontinuity;
    metadata.timestamp_error |= packet.timestamp_error;
    Ok(())
}
