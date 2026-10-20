//! Electromagnetic field and signal wave-propagation visualization pipeline.

#[derive(Clone, Debug)]
pub struct EmfFieldDesc {
    pub frequency_hz: f32,
    pub amplitude: f32,
    pub attenuation: f32,
}
