//! Artifact metadata protocol. No token-count substitution, byte hashing,
//! authentication, publication, or provider restoration is performed here.
#![forbid(unsafe_code)]
mod identity;
mod session;
pub use identity::*;
pub use session::*;
