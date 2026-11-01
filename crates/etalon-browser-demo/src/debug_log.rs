//! Bounded, local-only Rust log capture for the browser qualification page.

use serde_json::json;
use std::{
    collections::VecDeque,
    fmt::Write,
    sync::{Mutex, OnceLock},
};
use wasm_bindgen::prelude::*;

const MAX_EVENTS: usize = 128;
const MAX_JSON_BYTES: usize = 64 * 1024;
const MAX_ENVIRONMENT_BYTES: usize = 16 * 1024;
const MAX_MESSAGE_BYTES: usize = 512;
const MAX_TARGET_BYTES: usize = 96;

struct Event {
    sequence: u64,
    time_ms: f64,
    level: &'static str,
    target: String,
    message: String,
    truncated: bool,
}
struct State {
    events: VecDeque<Event>,
    sequence: u64,
    dropped: u64,
    environment: Option<serde_json::Value>,
}
static STATE: Mutex<State> = Mutex::new(State {
    events: VecDeque::new(),
    sequence: 0,
    dropped: 0,
    environment: None,
});
static INSTALLED: OnceLock<bool> = OnceLock::new();
static LOGGER: BrowserLogger = BrowserLogger;
struct BrowserLogger;

struct BoundedText {
    value: String,
    limit: usize,
    truncated: bool,
}
impl Write for BoundedText {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        let remaining = self.limit.saturating_sub(self.value.len());
        let mut end = value.len().min(remaining);
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        self.truncated |= end < value.len();
        self.value.push_str(&value[..end]);
        Ok(())
    }
}
fn text(value: std::fmt::Arguments<'_>, limit: usize) -> (String, bool) {
    let mut out = BoundedText {
        value: String::with_capacity(limit),
        limit,
        truncated: false,
    };
    let _ = out.write_fmt(value);
    (out.value, out.truncated)
}
impl log::Log for BrowserLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }
    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let (target, target_truncated) =
            text(format_args!("{}", record.target()), MAX_TARGET_BYTES);
        let (message, message_truncated) = text(*record.args(), MAX_MESSAGE_BYTES);
        let time_ms = js_sys::Date::now();
        if let Ok(mut state) = STATE.lock() {
            state.sequence = state.sequence.saturating_add(1);
            let sequence = state.sequence;
            if state.events.len() == MAX_EVENTS {
                state.events.pop_front();
                state.dropped = state.dropped.saturating_add(1);
            }
            state.events.push_back(Event {
                sequence,
                time_ms,
                level: record.level().as_str(),
                target,
                message,
                truncated: target_truncated || message_truncated,
            });
        }
    }
    fn flush(&self) {}
}

#[wasm_bindgen]
pub fn configure_debug_logging() -> bool {
    let installed = *INSTALLED.get_or_init(|| log::set_logger(&LOGGER).is_ok());
    if !installed {
        return false;
    }
    if let Ok(mut state) = STATE.lock() {
        state.events.clear();
        state.sequence = 0;
        state.dropped = 0;
        state.environment = None;
    }
    log::set_max_level(log::LevelFilter::Debug);
    log::debug!(target: "etalon_browser_demo", "Local debug capture enabled; maximum {MAX_EVENTS} events");
    true
}

#[wasm_bindgen]
pub fn set_debug_environment(encoded: &str) -> bool {
    if encoded.len() > MAX_ENVIRONMENT_BYTES {
        return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(encoded) else {
        return false;
    };
    if !value.is_object() || value.to_string().len() > MAX_ENVIRONMENT_BYTES {
        return false;
    }
    let Ok(mut state) = STATE.lock() else {
        return false;
    };
    state.environment = Some(value);
    true
}

#[wasm_bindgen]
pub fn debug_log_json() -> String {
    let Ok(state) = STATE.lock() else {
        return json!({"status":"capture-unavailable"}).to_string();
    };
    // Reserve metadata space; keep newest events and account for JSON escaping.
    let metadata_bytes = state
        .environment
        .as_ref()
        .map_or(4, |value| value.to_string().len())
        + 2048;
    let mut events = Vec::new();
    let mut bytes = 2usize;
    for event in state.events.iter().rev() {
        let value = json!({
            "sequence":event.sequence,"time_ms":event.time_ms,"level":event.level,
            "target":event.target,"message":event.message,"truncated":event.truncated,
        });
        let encoded_bytes = value.to_string().len() + 1;
        if bytes + encoded_bytes > MAX_JSON_BYTES - metadata_bytes {
            break;
        }
        bytes += encoded_bytes;
        events.push(value);
    }
    let omitted = state.events.len() - events.len();
    events.reverse();
    json!({"schema":"etalon.debug-log.v1",
        "source_revision":option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"),
        "library_version":env!("CARGO_PKG_VERSION"),"build_target":"wasm32-unknown-unknown",
        "environment":state.environment,
        "status":if INSTALLED.get() == Some(&true) {"enabled"} else {"not-installed"},
        "level":"debug","max_events":MAX_EVENTS,"max_message_bytes":MAX_MESSAGE_BYTES,
        "max_target_bytes":MAX_TARGET_BYTES,"max_json_bytes":MAX_JSON_BYTES,
        "dropped_events":state.dropped.saturating_add(omitted as u64),
        "export_omitted_events":omitted,"events":events,
        "automatic_upload":false})
    .to_string()
}
