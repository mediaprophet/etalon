use crate::{abi::{Api, Dev}, config::checked, Error};

/// Unauthenticated USB string descriptors from the opened native handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsbIdentity {
    manufacturer: [u8; 256], product: [u8; 256], serial: [u8; 256],
    lengths: [u16; 3],
}

impl UsbIdentity {
    pub fn manufacturer(&self) -> &[u8] { &self.manufacturer[..usize::from(self.lengths[0])] }
    pub fn product(&self) -> &[u8] { &self.product[..usize::from(self.lengths[1])] }
    pub fn serial(&self) -> &[u8] { &self.serial[..usize::from(self.lengths[2])] }
}

pub(crate) unsafe fn query(api: Api, handle: *mut Dev) -> Result<UsbIdentity, Error> {
    let mut identity = UsbIdentity { manufacturer: [0; 256], product: [0; 256],
        serial: [0; 256], lengths: [0; 3] };
    checked("get_usb_strings", unsafe { (api.usb_strings)(handle,
        identity.manufacturer.as_mut_ptr().cast(), identity.product.as_mut_ptr().cast(),
        identity.serial.as_mut_ptr().cast()) })?;
    for (index, bytes) in [&identity.manufacturer, &identity.product, &identity.serial].iter().enumerate() {
        identity.lengths[index] = bytes.iter().position(|byte| *byte == 0)
            .ok_or(Error::InvalidIdentity)? as u16;
    }
    Ok(identity)
}
