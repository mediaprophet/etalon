# Forward FastRPC service ABI source review

Public QAIC commit `a66a0019fb488abae955bbc9b3a9de3dc15962d0` was read as
source. No compiler, generated executable, code generation or DSP code ran.
The nine retained files total 129,286 bytes; each file is below 65,536 bytes.
Exact URLs and hashes are in `provenance.json`. The upstream license is
BSD-3-Clause-Clear; retain the source notices and `LICENSE.txt` separately from
the FastRPC BSD-3-Clause reference license.

## Established conventions

1. `remote.idl` documents forward `remote_handle64.open`: the DSP PD loads the
   shared object and resolves its named skeleton entry. Its documented URI is
   `file:///<sofilename>?<interface>_skel_handle_invoke&_modver=1.0`.
   `ToHeader.hs` lines 195–203 emit exactly this form for the ordinary 32-bit
   scalar ABI. Domain selection is an additional `_dom` query parameter, which
   must match the selected runtime and admitted domain. This source does not
   justify inventing a domain spelling or substituting `_domain`.
2. `ToCSkel.hs` lines 307–321 establish distinct entry signatures:
   `int iface_skel_invoke(uint32_t, remote_arg*)` for a handle-free interface,
   and `int iface_skel_handle_invoke(remote_handle64, uint32_t, remote_arg*)`
   for an interface inheriting `remote_handle64`. The scalar type is u32 here;
   `_s64` is a distinct variant, not selected by a 64-bit handle alone.
3. `ToCSkel.hs` lines 379–409 enumerate methods and special-case inherited
   methods zero and one. `remote.idl` defines them as open and close. Application
   operation methods start at two. Their methods receive the native handle.
4. `CStub.c` uses the legacy `remote_handle_open/invoke/close` path for a
   handle-free generated stub. Its two-argument DSP skeleton is insufficient
   evidence for a `remote_handle64` forward-loader implementation.
5. The separately pinned FastRPC `mod_table.c` lines 332–355 establish standard
   handle-aware native lifecycle marshalling: open invokes with handle zero,
   scalar `0x00020001`, primitive int32 URI length, URI bytes, and one output
   handle; close invokes with handle zero, scalar `0x01000010`, and one input
   handle. Lines 574–580 distinguish `_modver=1.0` three-argument entries from
   the legacy two-argument model. That source implements the host reverse
   loader; it corroborates API conventions but is not a selected board's DSP
   firmware implementation.

## Implementation consequence

A two-argument raw byte-buffer method-zero skeleton must not be represented as
a `_modver=1.0` forward service. Forward compatibility requires the matching
three-argument entry, native open/close behavior and operation method two, or
separate authoritative evidence and qualification for a legacy forward model.
The application byte payload can remain an original bounded pointer-free
protocol; it must not be described as QAIC-generated sequence marshalling.

The Etalon original skeleton now implements the three-argument entry, native
open/close methods and operation method two. Open/close ignore the leading
native handle, as generated QAIC does; close uses its explicit input handle.
The shared URI parser fixes `_modver=1.0` and the exact handle-aware export.

The sources support implementing an original bounded native handle context.
They do not specify Etalon's storage policy, generation namespace, authorization
or anti-replay behavior. Those must be explicitly implemented and admitted.
Never turn a native handle into an unchecked host-provided pointer. Generation
echo is correlation, and source-compatible ABI is not successful device
execution, module signature verification or firmware qualification.
