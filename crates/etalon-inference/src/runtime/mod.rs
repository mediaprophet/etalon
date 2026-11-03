// Preserve inherited source/API while moving ownership; these are style-only exceptions.
#[allow(clippy::doc_lazy_continuation)]
#[cfg_attr(test, allow(clippy::module_inception))]
pub mod budget;
#[allow(clippy::explicit_counter_loop, clippy::op_ref)]
pub mod kv;
