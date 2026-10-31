#![no_std]
#![forbid(unsafe_code)]

pub mod adapter;
pub mod alignment;
pub mod authority;
pub mod buffer;
pub mod calibration;
pub mod capability;
pub mod capture_ingest;
pub mod conversion;
pub mod error;
pub mod evidence;
pub mod extended_capability;
pub mod guarded_capture;
pub mod ingest;
pub mod measurement;
pub mod observation;
pub mod policy;
pub mod processing;
pub mod processing_route;
pub mod processing_runtime;
pub mod quality;
pub mod quantity;
pub mod radio;
pub mod replay;
pub mod replay_ingest;
pub mod retention;
pub mod session;
pub mod spectrum;
pub mod stream_order;
pub mod synchronization;
pub mod time;

pub use error::ContractError;

#[cfg(test)]
extern crate std;
