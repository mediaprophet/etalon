//! Environmental, Space Weather, Atmosphere, Fire Ecology & Geotechnical Mechanics
//!
//! Provides zero-heap continuum physics and multi-hazard environmental coupling.

pub mod atmosphere;
pub mod coupling;
pub mod excavation;
pub mod fire_ecology;
pub mod geomechanics;
pub mod space_weather;

pub use atmosphere::*;
pub use coupling::*;
pub use excavation::*;
pub use fire_ecology::*;
pub use geomechanics::*;
pub use space_weather::*;
