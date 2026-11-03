//! Borrowed GGUF metadata traversal with explicit byte, nesting and work limits.
//!
//! A strict view validates the whole metadata directory; a KV cursor retains an
//! explicit error after a partial prefix. Neither interprets model semantics,
//! normalizes names, truncates shapes, nor requires tensor payload or padding.

mod cursor;
mod types;
mod view;

pub use cursor::KvCursor;
pub use types::{
    fnv1a64, Header, KvView, ParseError, ParseErrorKind, ParseLimits, TensorView, VersionPolicy,
    MAX_ARRAY_DEPTH,
};
pub use view::{MetadataView, TensorIter};

#[cfg(test)]
mod tests;
