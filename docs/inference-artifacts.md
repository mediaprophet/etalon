# Artifact output contracts

`etalon_inference_contract::artifact` provides a separate event protocol for
rendered text, HTML and binary artifacts. Artifact events do not consume token
counts. Their keys bind the session, execution mode, event sequence, media,
origin, byte length, content and provenance digests, observations, and optional
parent-event metadata.

`DeveloperArtifactSession` is an explicit standalone path requiring no controller
or policy permission. `GovernedArtifactSession` requires a host verifier for
admission, decisions and continued authority. Tracing and profiling flags do not
change either mode's requirements. A host verifier is an integration interface;
the metadata library does not authenticate it or implement Webizen itself.

An Allow decision can produce one `AcceptedArtifact`. The record has private
fields and cannot be cloned. A pending decision, copied key or parent digest is
not a delivery permission. Expired approval, lost authority, denial, steering,
clock regression, a full output slot or a verifier panic stops the session.
Event and byte limits are checked before acceptance. Empty output requires an
explicit allowance; all-zero digests are reserved for missing evidence.

The host must hash the actual immutable bytes, authenticate the acting subject
and controller, validate parent lineage, and retain the backing until delivery.
It must revalidate current authority and enqueue the accepted record under the
same authority lock boundary. Delivery then consumes that record outside those
locks. This library does not publish bytes, restore numerical state or undo
already delivered output. `AcceptedArtifact::into_key` returns reporting metadata,
not reusable authorization.

These APIs preserve the existing token contract and its generation conventions.
A host can impose stricter generation requirements at admission. The QualiaDB
runtime integration and actual replacement-output delivery require separate
qualification.
