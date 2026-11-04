use super::*;

impl WgpuComputeContext {
    pub(super) fn ensure_live(&self) -> Result<(), ForgeError> {
        #[cfg(not(target_arch = "wasm32"))]
        if !self.provider_lease.matches(&self.device, &self.queue) || !self.provider_lease.is_live()
        {
            return Err(ForgeError::DeviceLost(
                "compute device owner is no longer live".into(),
            ));
        }
        if self.device.owns_queue(&self.queue) == Some(false) {
            return Err(ForgeError::GpuValidation(
                "queue belongs to a different device".into(),
            ));
        }
        if self.device.is_valid_state() == Some(false) {
            return Err(ForgeError::DeviceLost(
                "compute device is no longer live".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires native GPU"]
    fn lost_owner_refuses_upload_and_readback_without_advancing_weights() {
        let mut context = WgpuComputeContext::new(4096).unwrap();
        eprintln!("lifecycle Forge adapter: {:?}", context.adapter);
        let view = context
            .allocate_and_write(&[0; 4], 0, 0, BindingUsage::StorageReadWrite)
            .unwrap();
        let resident = context.resident_weight_bytes();
        context.device.destroy();
        assert!(context.allocate_weight(&[0; 4], 0, 0).is_err());
        assert_eq!(context.resident_weight_bytes(), resident);
        assert!(context.read_buffer_f32(&view).is_err());
    }
}
