# Third-party reference material: licensing boundary

Everything under `sensing-vendor-adapters/sdk-reference/` except the
project-authored records listed at the end is unmodified upstream source. It is
kept for reference, provenance and ABI review only.

## Ring-fence

- These files are **not part of Etalon**. Each one keeps its upstream license,
  and the matching license text sits next to it in its directory.
- Etalon's license (CC-BY-NC-ND-4.0, see the repository `LICENSE`) does not
  apply to these files. Their licenses do not extend to any other part of this
  repository.
- No file in this directory is compiled, linked, included, generated from or
  loaded by any Etalon crate, build script or test. Etalon code that describes
  the same interfaces (for example `etalon-sense-fastrpc/src/abi.rs`) is an
  independent expression that cites these files as the reference.
- Redistributing this directory means meeting each upstream license on its own
  terms. Distributing Etalon without this directory does not affect Etalon.

## Components

| Directory | Upstream | License | License text |
|-----------|----------|---------|--------------|
| `linux/` | Linux kernel FastRPC driver (`fastrpc.c`, `fastrpc.h`) | GPL-2.0 with Linux-syscall-note (UAPI header) | `linux/GPL-2.0`, `linux/Linux-syscall-note` |
| `qualcomm-fastrpc/` | Qualcomm FastRPC userspace (`remote.h`, `rpcmem.h`, `fastrpc_ioctl.h`, `mod_table.*`, `adspmsgd_apps*`, `AEEStdDef.idl`, `README.md`) | BSD-3-Clause | `qualcomm-fastrpc/LICENSE.txt` |
| `qualcomm-qaic/` | Qualcomm QAIC IDL compiler sources and docs (`*.hs`, `*.c`, `*.idl`, `README.md`) | BSD-3-Clause-Clear | `qualcomm-qaic/LICENSE.txt` |
| `rtlsdr/` | osmocom librtlsdr (`librtlsdr.c`, `rtl-sdr.h`, `README`) | GPL-2.0-or-later | `rtlsdr/COPYING` |
| `rust-target/` | Rust compiler Hexagon target spec (`hexagon_unknown_none_elf.rs`) | Apache-2.0 OR MIT | `rust-target/LICENSE-APACHE`, `rust-target/LICENSE-MIT` |

Exact source URLs, upstream commits, byte lengths and SHA-256 digests are in
`provenance.json`, `qualcomm-fastrpc/service-reference-provenance.json`,
`qualcomm-qaic/provenance.json`, `rtlsdr/provenance.json` and
`rust-target/provenance.json`.

## Project-authored records (Etalon license)

`README.md`, `THIRD-PARTY-LICENSES.md`, `qualcomm-qaic/ABI-REVIEW.md` and the
`provenance.json` / `service-reference-provenance.json` files.
