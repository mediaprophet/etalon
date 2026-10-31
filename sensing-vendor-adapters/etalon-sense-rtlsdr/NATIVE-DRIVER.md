# Native RTL-SDR receive boundary

This implementation dynamically loads an explicitly admitted local librtlsdr
runtime and opens only the device index authenticated by its deployment token.
It does not install a driver, discover a preferred receiver or connect to a
network service. The application supplies and authenticates the runtime,
complete native dependency closure, USB driver and target compatibility.

## Authoritative ABI

The original Rust FFI declarations use the public Osmocom interface at commit
`797f8143266d983c56d8f35d2d442527529dd8a5`. Header, implementation, README and
license are retained as reference material under `../sdk-reference/rtlsdr`.
Their 80,417 source bytes have recorded SHA-256 digests and immutable URLs in
`provenance.json`; the download budget was 300,000 bytes. No downloaded code
was executed, and no upstream implementation body was copied into this crate.
GPL notices remain in the references. Native library distribution and license
obligations remain separate from merely selecting a local runtime.

The API uses an opaque device pointer, C `int` statuses, u32 tuning values and
`rtlsdr_read_sync(dev, buffer, int length, int *count)`. The upstream sync path
calls libusb with timeout zero. It may block indefinitely, and close can wait
for outstanding async activity. This crate does not start an async path and
requires exclusive ownership with no other native binding using the handle.

## Admission, identity and ownership

`RtlSdrRuntime::load` is an explicit unsafe boundary: hashing a library does not
make native code trustworthy, establish loaded-code identity or bound native
memory/latency. The host must hold immutable code, no-egress and device/policy
leases. Windows loading restricts native resolution to the admitted DLL's
directory and the system directory; Unix uses immediate local symbol binding.
Neither flag selection supplies a complete dependency-integrity sandbox.

One runtime instance permits only one open attempt for its generation. USB
manufacturer/product/serial descriptors are queried through the actual opened
handle and presented to host gates. Empty descriptors remain empty; descriptors
and enumeration indices are declarations, not cryptographic device identities.
The host must durably prevent generation reuse across runtime/device restarts.

The runtime tracks outstanding native ownership independently of the device
wrapper. Forgetting a wrapper therefore retains the runtime mapping. A failed
native close has an unknown outcome: its pointer is never retried or reused,
and the runtime mapping is retained. Explicit `close` reports cleanup outcomes;
Drop can only attempt cleanup. Shared lifecycle serialization covers this crate's
load/open/close/unload calls; unrelated bindings need host coordination.

## Configuration and sample meaning

Configuration explicitly selects center frequency, sample rate, frequency
correction, tuner automatic/manual gain and digital AGC. Synthetic test mode
and direct sampling are disabled. The typed constructor bounds ppm correction
to ±1000 and manual gain to ±1000 tenths of a dB to avoid dangerous arithmetic
inputs in the C boundary; these bounds do not assert receiver support. Manual
gain must also occur in the handle's bounded native gain list. Unsupported
configurations fail through driver status or getter validation.

The configured sample-rate getter is the upstream integer truncation of its
computed resampler rate. It is a reported integer, not a calibrated exact
physical rate. Gain, frequency and correction getters are likewise native
declarations. Digital AGC and gain-mode setters have no matching getter here;
setter acknowledgement is preserved separately from the reported values.
Bandwidth, oscillator accuracy and effective gain uncertainty remain unknown.

`reset_buffer` is attempted after configuration, but the pinned upstream
function ignores its register-write statuses. A zero return cannot prove FIFO
purging. Returned bytes may predate configuration, and no sample-time anchor,
sample continuity, sequence-to-sample relationship or absence of loss is claimed.

`read_into` requires a nonempty even-length caller buffer within the admitted
pair capacity and returns only exact-length success. Partial/odd/negative or
otherwise inconsistent native results are rejected. Error output is cleared.
Native identity and getter declarations are checked around acquisition, and
host policy is checked at Read, ReadReady immediately before the blocking call,
and PostRead. The optional policy adapter retains the original operation expiry
rather than renewing it after I/O. Checks cannot cancel an already executing
native call or undo RF acquisition; permission must cover the entire open
receiver lifetime, including intervals between reads.

The raw payload is unsigned-u8 interleaved I/Q. Optional conversion is digital
arithmetic, not RF calibration. The typed native allocation class remains
`ColdUnbounded`: fixed caller samples do not bound libusb, driver or runtime
allocation. There is no hard deadline, `SdrReader` bounded-duration claim,
background thread or CPU substitute.

All builds, tests, runtime loading and hardware execution remain deferred.
Source-only cases describe rejection of dangerous control values, invalid gain,
late policy denial, short reads, changed configuration, forgotten ownership and
unknown cleanup. None has been run. ABI/layout, numeric execution, receiver
behavior, driver interruption, local isolation and allocation measurement still
require qualification after the cleanup hold is lifted.
