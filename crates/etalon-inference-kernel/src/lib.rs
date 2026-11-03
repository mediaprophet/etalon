#![forbid(unsafe_code)]
//! Dependency-free numerical operator and paged attention contracts.

// Retain the frozen source and API during ownership migration; only inherited style lints.
#[allow(clippy::manual_is_multiple_of, clippy::collapsible_if, clippy::len_without_is_empty)]
#[cfg_attr(test, allow(clippy::needless_range_loop, clippy::unnecessary_cast))]
pub mod operators;
#[allow(clippy::manual_is_multiple_of)]
pub mod paged_attention;
