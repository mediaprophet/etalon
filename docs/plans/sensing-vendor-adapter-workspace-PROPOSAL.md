# Sensing vendor adapter workspace proposal

Status: source-only proposal; not integrated or qualified.

The isolated worktree now has `sensing-vendor-adapters/` as a separate
workspace. Prior drafts are retained in `sensing-vendor-drafts-archive/`, and
the isolated Etalon facade and agent no longer reference the vendor crates.
No primary checkout, root workspace manifest or lockfile was changed.

Keep `etalon-sense-core`, its caller-buffered contracts, and portable sensor
facade in the main Etalon workspace. Keep vendor inference runtimes and SDKs
out of its unconditional dependency resolution. Optional Cargo dependencies
can still affect a workspace lockfile, and the current `crates/*` membership
means adding an apparently optional crate there does not isolate it.

Move the OpenVINO and QNN implementations to an explicitly separate adapter
workspace outside the main workspace's wildcard members. That workspace may
depend on portable Etalon sensing interfaces using an agreed version or
coordinated local checkout. An application opting into a vendor adapter owns
that adapter workspace's lockfile, SDK installation, target-specific features,
runtime loading and deployment policy. Keep GPU, CPU and device feature wiring
separate from vendor inference runtime resolution. Do not assume that declaring
a target optional dependency removes it from all lockfile resolution.

Coordinate a single supported `ort` version with QualiaDB before shared
integration. The other modernization chat reports a Qualia policy of exact
`2.0.0-rc.15`; the isolated drafts were migrated from cached `rc.12` to locally
cached `rc.15` at source level. They require independent source review and
later qualification before being presented as compatible. Provider behavior,
APIs, dynamic-loading policy and CPU
fallback must be reviewed for the agreed version. Do not download anything as
part of this proposal.

QualiaDB's existing ORT dependency enables `download-binaries`; feature
unification in a combined workspace could therefore enable a build-time
download even when an adapter itself requests only `load-dynamic`. Establish
an offline, no-download dependency and artifact strategy before qualification;
do not rely on the adapter's feature list alone to enforce no egress.

Before integration, agree workspace paths, membership/exclusions, facade
feature ownership, lockfile ownership and a reproducible deployment contract.
No primary manifest, lockfile or shared accelerator API changes are authorized
by this proposal alone. Hard memory and no-egress isolation remain host
responsibilities for native vendor runtimes. QNN HTP remains model inference,
not a general DSP kernel implementation.
