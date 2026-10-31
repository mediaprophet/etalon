//! Native, caller-buffered local input helpers for Etalon sensing.
//! Opening an input is a host action; this crate provides no policy authority.
#![forbid(unsafe_code)]

pub mod stream;

use etalon_signal::{wave::Pcm16Wave, SignalError};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

#[derive(Debug)]
pub enum InputError {
    Io(io::Error),
    Capacity,
    InvalidWave(SignalError),
}

/// Read a local file into a caller-owned, bounded buffer and parse PCM16 WAVE.
/// The file is opened read-only. The host must apply any policy it requires
/// before calling this function. `buffer` may be modified when an error occurs;
/// it is never retained by this crate beyond the returned borrowed view.
///
/// The read continues until EOF and probes one extra byte when the buffer is
/// full, so growing inputs are rejected instead of silently truncated.
pub fn load_pcm16_wave<'a>(path: &Path, buffer: &'a mut [u8]) -> Result<Pcm16Wave<'a>, InputError> {
    let mut file = File::open(path).map_err(InputError::Io)?;
    let used = read_bounded(&mut file, buffer)?;
    Pcm16Wave::parse(&buffer[..used]).map_err(InputError::InvalidWave)
}

/// Read to EOF without heap allocation. Partial reads and interrupted system
/// calls are handled; input larger than the caller buffer fails explicitly.
pub fn read_bounded(reader: &mut impl Read, buffer: &mut [u8]) -> Result<usize, InputError> {
    let mut used = 0usize;
    loop {
        if used == buffer.len() {
            let mut probe = [0u8; 1];
            match reader.read(&mut probe) {
                Ok(0) => return Ok(used),
                Ok(_) => return Err(InputError::Capacity),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(InputError::Io(error)),
            }
        }
        match reader.read(&mut buffer[used..]) {
            Ok(0) => return Ok(used),
            Ok(count) if count <= buffer.len() - used => used += count,
            Ok(_) => {
                return Err(InputError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "reader returned an invalid byte count",
                )))
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(InputError::Io(error)),
        }
    }
}

#[cfg(test)]
mod tests;
