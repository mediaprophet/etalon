use crate::TestingContext;

impl TestingContext {
    /// Utility to allow future asynchronous polling.
    pub async fn async_poll(
        &self,
        poll_type: etalon::PollType,
    ) -> Result<etalon::PollStatus, etalon::PollError> {
        self.device.poll(poll_type)
    }
}
