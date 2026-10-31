//! Small, allocation-free DSP primitives over caller-owned sample buffers.
//!
//! These routines transform or summarize samples only. They do not acquire
//! hardware data, establish calibration, identify a source, or authenticate a
//! measurement.
#![no_std]
#![forbid(unsafe_code)]

pub mod filter;
pub mod audio;
pub mod correlation;
pub mod detection;
pub mod decode;
pub mod dft;
pub mod execution;
pub mod fft;
pub mod iq;
pub mod phase;
pub mod pipeline;
pub mod goertzel;
pub mod stats;
pub mod temporal;
pub mod window;
pub mod vision;
pub mod wave;

/// Failures are explicit; no routine invents a nominal signal or estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalError {
    EmptyInput,
    LengthMismatch,
    NonFiniteInput,
    InvalidParameter,
    NumericRange,
}

#[cfg(test)]
extern crate std;
