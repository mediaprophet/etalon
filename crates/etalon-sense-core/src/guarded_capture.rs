//! Optional host-policy composition for synchronous capture operations.
//!
//! This layer has no default policy or human-consent assumption. Hosts provide
//! the exact opaque policy request, identity mapping, evaluator, current policy
//! epoch, and revocation snapshot. The final policy recheck encloses the
//! synchronous adapter operation. It cannot lock external policy state, cancel
//! blocking I/O, or make provider access atomic with revocation.

use crate::{
    adapter::{
        AdapterError, AuthorizedCapture, AuthorizedExtendedCapture, CaptureBuffer, CaptureSession,
        ExtendedCaptureBuffer, ExtendedCaptureSession, ExtendedSensorAdapter, SensorAdapter,
    },
    capability::SensorCapability,
    capture_ingest::{
        capture_and_ingest, capture_extended_and_ingest, CaptureIngestError, IngestMetadata,
    },
    extended_capability::ExtendedCapability,
    ingest::{ExtendedIngestStream, ExtendedIngestedFrame, IngestStream, IngestedFrame},
    policy::{
        authorize_once, dispatch_once, DecisionCommitment, PolicyEpochProvider, PolicyError,
        PolicyEvaluator, PolicyId, PolicyRequest, RevocationProvider,
    },
};

/// Exact capability value bound to the adapter operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CaptureCapability {
    BuiltIn(SensorCapability),
    Extended(ExtendedCapability),
}

/// Host-declared relationship between one exact source/capability and the full
/// policy request used to start and capture its session.
///
/// The start and capture requests must agree on actor, subject, source,
/// modality, context, purpose, revocation authority, and evidence; only the
/// operation ID may differ. Policy IDs remain host-defined opaque mappings.
/// This binding checks consistency but does not authenticate declarations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapturePolicyBinding {
    source_id: u64,
    source_policy_id: PolicyId,
    capability: CaptureCapability,
    purpose_id: u64,
    start_request: PolicyRequest,
    capture_request: PolicyRequest,
}

impl CapturePolicyBinding {
    pub fn new(
        source_id: u64,
        source_policy_id: PolicyId,
        capability: CaptureCapability,
        purpose_id: u64,
        start_request: PolicyRequest,
        capture_request: PolicyRequest,
    ) -> Result<Self, PolicyError> {
        if source_id == 0 || purpose_id == 0 {
            return Err(PolicyError::Contract(
                crate::ContractError::InvalidIdentifier,
            ));
        }
        if start_request.source() != source_policy_id
            || capture_request.source() != source_policy_id
            || start_request.purpose() != capture_request.purpose()
            || !same_policy_scope(&start_request, &capture_request)
        {
            return Err(PolicyError::RequestMismatch);
        }
        Ok(Self {
            source_id,
            source_policy_id,
            capability,
            purpose_id,
            start_request,
            capture_request,
        })
    }

    pub const fn source_id(self) -> u64 {
        self.source_id
    }
    pub const fn source_policy_id(self) -> PolicyId {
        self.source_policy_id
    }
    pub const fn capability(self) -> CaptureCapability {
        self.capability
    }
    pub const fn purpose_id(self) -> u64 {
        self.purpose_id
    }
    pub const fn start_request(self) -> PolicyRequest {
        self.start_request
    }
    pub const fn capture_request(self) -> PolicyRequest {
        self.capture_request
    }

    fn validate(
        self,
        request: &PolicyRequest,
        action: CaptureAction,
        source_id: u64,
        capability: CaptureCapability,
        purpose_id: u64,
    ) -> Result<(), PolicyError> {
        let expected_request = match action {
            CaptureAction::Start => self.start_request,
            CaptureAction::Capture => self.capture_request,
        };
        if source_id != self.source_id
            || capability != self.capability
            || purpose_id != self.purpose_id
            || request != &expected_request
            || request.source() != self.source_policy_id
        {
            return Err(PolicyError::RequestMismatch);
        }
        Ok(())
    }
}

/// Session returned only after a guarded start, retaining the exact policy
/// scope for every later guarded capture. Its raw adapter session is private.
#[derive(Debug, PartialEq)]
pub struct GuardedCaptureSession<S> {
    session: S,
    binding: CapturePolicyBinding,
}

/// An ingested frame paired with the exact request and decision commitment
/// accepted by the guarded dispatch that produced it.
///
/// This is an unverified host-policy assertion for caller audit and routing.
/// It is not signed evidence, an authorization attestation, or proof that the
/// policy evaluator or providers are trustworthy.
#[derive(Debug, PartialEq)]
pub struct GuardedIngestedFrame<F> {
    frame: F,
    request: PolicyRequest,
    decision: DecisionCommitment,
}

impl<F> GuardedIngestedFrame<F> {
    fn new(frame: F, request: PolicyRequest, decision: DecisionCommitment) -> Self {
        Self {
            frame,
            request,
            decision,
        }
    }

    pub const fn frame(&self) -> &F {
        &self.frame
    }
    pub fn into_frame(self) -> F {
        self.frame
    }
    pub const fn policy_request(&self) -> &PolicyRequest {
        &self.request
    }
    pub const fn decision_commitment(&self) -> &DecisionCommitment {
        &self.decision
    }
}

impl<S> GuardedCaptureSession<S> {
    fn new(session: S, binding: CapturePolicyBinding) -> Self {
        Self { session, binding }
    }
}

impl GuardedCaptureSession<CaptureSession> {
    pub const fn session_id(&self) -> u64 {
        self.session.session_id()
    }
    pub const fn generation(&self) -> u32 {
        self.session.generation()
    }
    pub const fn source_id(&self) -> u64 {
        self.session.source().source_id()
    }
}

impl GuardedCaptureSession<ExtendedCaptureSession> {
    pub const fn session_id(&self) -> u64 {
        self.session.session_id()
    }
    pub const fn generation(&self) -> u32 {
        self.session.generation()
    }
    pub const fn source_id(&self) -> u64 {
        self.session.source().source_id()
    }
}

fn same_policy_scope(start: &PolicyRequest, capture: &PolicyRequest) -> bool {
    start.subject() == capture.subject()
        && start.actor() == capture.actor()
        && start.source() == capture.source()
        && start.modality() == capture.modality()
        && start.context() == capture.context()
        && start.purpose() == capture.purpose()
        && start.revocation_authority() == capture.revocation_authority()
        && start.evidence() == capture.evidence()
}

#[derive(Clone, Copy)]
enum CaptureAction {
    Start,
    Capture,
}

/// Policy, adapter, or ingest failure from a guarded entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardedCaptureError {
    Policy(PolicyError),
    Adapter(AdapterError),
    Ingest(CaptureIngestError),
}

impl From<PolicyError> for GuardedCaptureError {
    fn from(error: PolicyError) -> Self {
        Self::Policy(error)
    }
}
impl From<AdapterError> for GuardedCaptureError {
    fn from(error: AdapterError) -> Self {
        Self::Adapter(error)
    }
}
impl From<CaptureIngestError> for GuardedCaptureError {
    fn from(error: CaptureIngestError) -> Self {
        Self::Ingest(error)
    }
}

/// Acquire a decision, then recheck the exact request, policy epoch, decision,
/// and revocation snapshot immediately before calling a synchronous operation.
fn dispatch_guarded<E, P, R, F, T>(
    request: PolicyRequest,
    evaluator: &E,
    epochs: &P,
    revocations: &R,
    operation: F,
) -> Result<(T, PolicyRequest, DecisionCommitment), PolicyError>
where
    E: PolicyEvaluator,
    P: PolicyEpochProvider,
    R: RevocationProvider,
    F: FnOnce(&PolicyRequest, &DecisionCommitment) -> T,
{
    let decision = authorize_once(request, evaluator, epochs, revocations)?;
    dispatch_once(
        decision,
        &request,
        evaluator,
        epochs,
        revocations,
        |accepted, commitment| (operation(accepted, commitment), *accepted, *commitment),
    )
}

/// Policy-check and start a built-in adapter session.
pub fn guarded_start_builtin<A, E, P, R>(
    adapter: &mut A,
    authorization: AuthorizedCapture,
    binding: CapturePolicyBinding,
    policy_request: PolicyRequest,
    evaluator: &E,
    epochs: &P,
    revocations: &R,
) -> Result<GuardedCaptureSession<CaptureSession>, GuardedCaptureError>
where
    A: SensorAdapter,
    E: PolicyEvaluator,
    P: PolicyEpochProvider,
    R: RevocationProvider,
{
    let source = authorization.source();
    binding.validate(
        &policy_request,
        CaptureAction::Start,
        source.source_id(),
        CaptureCapability::BuiltIn(source.capability()),
        authorization.purpose_id(),
    )?;
    let (started, _, _) =
        dispatch_guarded(policy_request, evaluator, epochs, revocations, |_, _| {
            adapter.start(authorization)
        })?;
    started
        .map(|session| GuardedCaptureSession::new(session, binding))
        .map_err(GuardedCaptureError::Adapter)
}

/// Policy-check and start a host-registered adapter session.
pub fn guarded_start_extended<A, E, P, R>(
    adapter: &mut A,
    authorization: AuthorizedExtendedCapture,
    binding: CapturePolicyBinding,
    policy_request: PolicyRequest,
    evaluator: &E,
    epochs: &P,
    revocations: &R,
) -> Result<GuardedCaptureSession<ExtendedCaptureSession>, GuardedCaptureError>
where
    A: ExtendedSensorAdapter,
    E: PolicyEvaluator,
    P: PolicyEpochProvider,
    R: RevocationProvider,
{
    let source = authorization.source();
    binding.validate(
        &policy_request,
        CaptureAction::Start,
        source.source_id(),
        CaptureCapability::Extended(source.capability()),
        authorization.purpose_id(),
    )?;
    let (started, _, _) =
        dispatch_guarded(policy_request, evaluator, epochs, revocations, |_, _| {
            adapter.start(authorization)
        })?;
    started
        .map(|session| GuardedCaptureSession::new(session, binding))
        .map_err(GuardedCaptureError::Adapter)
}

/// Stop a guarded built-in session. Retaining the wrapper permits retry if the
/// backend does not confirm shutdown. Discard it after a successful stop.
pub fn guarded_stop_builtin<A: SensorAdapter>(
    adapter: &mut A,
    session: &GuardedCaptureSession<CaptureSession>,
) -> Result<(), GuardedCaptureError> {
    adapter
        .stop(session.session)
        .map_err(GuardedCaptureError::Adapter)
}

/// Stop a guarded host-registered session. A failed stop leaves the wrapper
/// available for retry; discard it after a successful stop.
pub fn guarded_stop_extended<A: ExtendedSensorAdapter>(
    adapter: &mut A,
    session: &GuardedCaptureSession<ExtendedCaptureSession>,
) -> Result<(), GuardedCaptureError> {
    adapter
        .stop(session.session)
        .map_err(GuardedCaptureError::Adapter)
}

/// Guard and ingest one built-in capture. Policy is re-evaluated immediately
/// before the synchronous capture/ingest call. Providers must still check their
/// trusted clock at acquisition and arrange cancellation around blocking I/O.
pub fn guarded_capture_and_ingest<'a, A, E, P, R>(
    adapter: &mut A,
    session: &GuardedCaptureSession<CaptureSession>,
    authorization: AuthorizedCapture,
    buffer: CaptureBuffer<'a>,
    stream: &mut IngestStream,
    metadata: IngestMetadata,
    evaluator: &E,
    epochs: &P,
    revocations: &R,
) -> Result<GuardedIngestedFrame<IngestedFrame<'a>>, GuardedCaptureError>
where
    A: SensorAdapter,
    E: PolicyEvaluator,
    P: PolicyEpochProvider,
    R: RevocationProvider,
{
    let source = authorization.source();
    let binding = session.binding;
    let policy_request = binding.capture_request;
    binding.validate(
        &policy_request,
        CaptureAction::Capture,
        source.source_id(),
        CaptureCapability::BuiltIn(source.capability()),
        authorization.purpose_id(),
    )?;
    let (ingested, accepted_request, decision) =
        dispatch_guarded(policy_request, evaluator, epochs, revocations, |_, _| {
            capture_and_ingest(
                adapter,
                session.session,
                authorization,
                buffer,
                stream,
                metadata,
            )
        })?;
    ingested
        .map(|frame| GuardedIngestedFrame::new(frame, accepted_request, decision))
        .map_err(GuardedCaptureError::Ingest)
}

/// Guard and ingest one host-registered capture with the exact extended
/// capability and the caller's full policy request.
pub fn guarded_capture_extended_and_ingest<'a, A, E, P, R>(
    adapter: &mut A,
    session: &GuardedCaptureSession<ExtendedCaptureSession>,
    authorization: AuthorizedExtendedCapture,
    buffer: ExtendedCaptureBuffer<'a>,
    stream: &mut ExtendedIngestStream,
    metadata: IngestMetadata,
    evaluator: &E,
    epochs: &P,
    revocations: &R,
) -> Result<GuardedIngestedFrame<ExtendedIngestedFrame<'a>>, GuardedCaptureError>
where
    A: ExtendedSensorAdapter,
    E: PolicyEvaluator,
    P: PolicyEpochProvider,
    R: RevocationProvider,
{
    let source = authorization.source();
    let binding = session.binding;
    let policy_request = binding.capture_request;
    binding.validate(
        &policy_request,
        CaptureAction::Capture,
        source.source_id(),
        CaptureCapability::Extended(source.capability()),
        authorization.purpose_id(),
    )?;
    let (ingested, accepted_request, decision) =
        dispatch_guarded(policy_request, evaluator, epochs, revocations, |_, _| {
            capture_extended_and_ingest(
                adapter,
                session.session,
                authorization,
                buffer,
                stream,
                metadata,
            )
        })?;
    ingested
        .map(|frame| GuardedIngestedFrame::new(frame, accepted_request, decision))
        .map_err(GuardedCaptureError::Ingest)
}

#[cfg(test)]
#[path = "guarded_capture_tests.rs"]
mod tests;
