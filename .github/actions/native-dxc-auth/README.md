# Authenticated static DXC metadata transport

This helper is scoped to the Windows Core MSRV job. It supplies curl with a
unique runner-temporary configuration and a private netrc entry for
`api.github.com` only. The job token has `contents: read`; it is not placed in
arguments, logs, workflow outputs or a generic Authorization header. Only the
configuration directory path reaches the Cargo step. Credentials are opaque UTF-8
values bounded to 8192 bytes; control characters and line separators are refused.
Quoted netrc serialization escapes quotes and backslashes without restricting the
token alphabet. Synthetic tests exercise curl parsing and host isolation locally.

Preparation uses that same curl configuration to require HTTP 200, an
authenticated rate-limit bucket greater than the public 60-request limit, the
exact release tag and ID, boolean immutability, and the pinned Windows archive
names, IDs, lengths, SHA-256 digests and download URLs. Token authentication
through netrc must succeed in Actions; unsupported authentication fails closed.
No response is fabricated and no curl executable is replaced.

The dependency still fetches release metadata itself, requires immutability,
downloads its native archive and verifies its compiled-in checksum. Metadata
preflight does not independently hash downloaded archive bytes. The published
crate and native compiler remain unchanged. See
[dependency provenance](../../../docs/native-accelerator-dependencies.md).

The temporary configuration enables HTTP failure status, HTTPS-only requests
and redirects, bounded connection/transfer timeouts and bounded retries. The
preflight logs only HTTP status and selected rate-limit/request-ID headers.
The original failure's cause remains unproven: an HTTP error body can trigger
the dependency's misleading mutable-release message.

Cleanup runs immediately on preparation failure and in an always-run workflow
step. It verifies the resolved runner-temp parent, unique name, ownership marker
and ordinary files before deletion; it never recursively removes arbitrary
content. Runner termination can prevent cleanup, so this job requires an
ephemeral hosted runner. Credentials remain accessible to code running as the
same runner account during Cargo; this transport is not process isolation.
