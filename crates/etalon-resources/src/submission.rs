#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionError {
    Rejected,
}

pub fn submit_checked(
    queue: &etalon::Queue,
    commands: impl IntoIterator<Item = etalon::CommandBuffer>,
) -> Result<Option<etalon::SubmissionIndex>, SubmissionError> {
    let submission = queue.submit(commands);
    #[cfg(not(target_arch = "wasm32"))]
    if submission.is_none() {
        return Err(SubmissionError::Rejected);
    }
    if submission.is_some() {
        crate::provider_scope::record_route(2);
    }
    Ok(submission)
}
