# Native inference metadata and P64 preparation

`etalon_model::inference` owns numerical model metadata, tensor directories and
native P64 preparation. GGUF is an import/compatibility boundary. Its existing
`GgufTensorIndex`, `GgufTensorInfo` and `GgufHyperparams` names are aliases to the
neutral `TensorIndex`, `TensorInfo` and `ModelHyperparams` types. The historical
`ggml_type` field is a numerical encoding identifier; `WeightEncoding` distinguishes
supported primary encodings from opaque future IDs. No unknown ID is silently
mapped to a known codec.

`TensorIndex<H>` stores one authoritative hyperparameter value. Hosts can specialize
H to preserve architecture admission policy while projecting numerical metadata
through `HyperparamsSource`, without copying or cloning the tensor directory.
`PreparedP64` borrows the validated P64 byte owner, checks an explicit integrity
mode and metadata allocation budget, and owns only the small numerical directory.
All descriptors use absolute P64 tensor blob offsets with data origin zero.
Known primary encoding lengths must fit each individual blob. Unknown storage flags
are rejected for execution. Unknown tensor encodings remain inspectable but their
row decoder refuses them; raw alternate precision fields remain in the P64 view.

## Optional inference metadata extension, version 1

The P64 container remains version 4. Its fixed 64-byte hyperparameter record lacks
recurrent and PLE parameters, so importers append a 736-byte inference metadata
record after the existing tokenizer bytes when these fields are present. Ordinary
models without these fields retain their exact previous writer bytes. The Q42T
reader consumes its declared entries and ignores appended extension bytes.

The header reserved20 locator is little-endian:

| Bytes | Value |
|---|---|
| 0..8 | `ETLMDTA\0` |
| 8..12 | Record offset relative to tokenizer section start |
| 12..16 | Record length, 736 |
| 16..18 | Version, 1 |
| 18..20 | Zero |

This locator is assigned only while constructing a fresh header. Existing unknown
reserved bytes are never overwritten by a read or preparation operation. P64View
preserves all raw bytes and unknown reserved data, as before.

The record is little-endian:

| Bytes | Value |
|---|---|
| 0..8 | `ETLIMD1\0` |
| 8..10 | Version, 1 |
| 10..12 | Zero |
| 12..16 | Total length, 736 |
| 16..92 | Nineteen 32-bit ModelHyperparams fields in declaration order; f32 fields preserve IEEE bits |
| 92..96 | PLE present, 0 or 1 |
| 96..732 | Qwen4ExpPleConfig fields in declaration order, fixed arrays in element order; i64 fields are 8 bytes |
| 732..736 | CRC-32C of bytes 0..732 |

An absent PLE configuration has an all-zero 636-byte payload. The existing P64
metadata CRC also covers locator and record. Structure-only container admission
still validates the extension's own version, bounds and CRC. Unsupported versions,
malformed locators, unknown nonzero reserved metadata and conflicting explicit
metadata fail preparation; the raw P64View remains available for preservation. Recurrent
architectures requiring absent parameters, or Qwen4Exp missing its trained PLE
contract, fail preparation instead of receiving guessed defaults.

GGUF compatibility conversion appends the extension automatically. Caller-buffered
conversion can serialize the fixed record into its admitted tokenizer storage;
the conversion adapter validates the supplied record against the parsed numerical
metadata before it writes output. FFN conversion preserves the existing locator
and tokenizer region. Unknown tensor names that the old compiler encoded as
`tensor.<source_hash>` retain their original identity through `source_name_hash`;
raw P64 names, encoding fields and precision records remain authoritative.

This optional extension does not promise that an architecture has a qualified
executor. Preparation, codec support and numerical execution are separate gates.
