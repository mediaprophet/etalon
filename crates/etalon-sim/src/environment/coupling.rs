//! Multi-Domain Earth-System Coupler (`coupling.rs`)
//!
//! Synthesizes heliophysical solar-wind electromagnetic forcing, synoptic
//! atmospheric boundary layers, edaphic moisture depletion, and wildland
//! fire behavior into a deterministic, zero-heap multi-hazard state report.
//!
//! Adheres strictly to `AGENTS.md` §0 (Zero heap in hot paths, deterministic).

use super::atmosphere::{BarometricState, BoundaryLayerProfile};
use super::fire_ecology::{
    evaluate_dead_fine_fuel_moisture, evaluate_fireline_intensity, FirelineVerdict,
    ForestFuelStratum, FuelMoistureState, DEAD_FUEL_EXTINCTION_MOISTURE,
};
#[cfg(test)]
use super::fire_ecology::FirelineRegime;
use super::space_weather::{
    evaluate_space_weather_coupling, AtmosphericElectricCircuit, GeomagneticStormState,
    HaleMagneticCycle, SchwabeSolarCycle, SolarWindState, BASELINE_TSI_W_M2,
};

/// Integrated multi-domain earth-system state report.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeliophysicalFireWeatherReport {
    /// Planetary geomagnetic storm state and auroral dissipation.
    pub storm: GeomagneticStormState,
    /// Global Electric Circuit potential and ground field.
    pub gec: AtmosphericElectricCircuit,
    /// Moist air density at target altitude in kg/m³.
    pub air_density_kg_m3: f32,
    /// Atmospheric boundary-layer horizontal wind speed at 10m in m/s.
    pub wind_10m_m_s: f32,
    /// Dead Fine Fuel Moisture fraction ($DFMC \in [0.02, 0.35]$).
    pub dfmc_fraction: f32,
    /// Fire behavior verdict and Byram fireline intensity.
    pub fireline: FirelineVerdict,
    /// Joint probability of dry lightning ignition ($[0.0, 1.0]$).
    pub dry_lightning_ignition_risk: f32,
}

/// Evaluates integrated multi-domain heliophysical space weather, atmospheric
/// thermodynamics, and fire weather coupling.
pub fn evaluate_heliophysical_fire_weather(
    solar: &SchwabeSolarCycle,
    wind_plasma: &SolarWindState,
    baro: &BarometricState,
    abl: &BoundaryLayerProfile,
    fuel: &ForestFuelStratum,
    moisture: &FuelMoistureState,
    slope_deg: f32,
    elevation_m: f32,
) -> HeliophysicalFireWeatherReport {
    // 1. Space weather coupling modulated by atmospheric moisture
    let hale = HaleMagneticCycle::default();
    let (storm, mut gec) = evaluate_space_weather_coupling(solar, &hale, wind_plasma);
    let humidity_suppression = (1.0 - baro.relative_humidity * 0.75).clamp(0.0, 1.0);
    gec.dry_lightning_susceptibility = (gec.dry_lightning_susceptibility * humidity_suppression).clamp(0.0, 1.0);

    // 2. Air density & synoptic 10m wind speed (10m above zero-plane displacement d)
    let air_density_kg_m3 = baro.air_density(elevation_m);
    let wind_10m_m_s = abl.sample_wind_speed(abl.zero_plane_displacement_d + 10.0);

    // 3. Solar TSI anomaly modulates solar irradiance into dead fine fuel moisture
    let tsi_delta = solar.tsi_w_m2 - BASELINE_TSI_W_M2;
    let solar_irradiance = (800.0 + tsi_delta * 45.0 + elevation_m * 0.05).clamp(100.0, 1400.0);
    let dfmc_fraction = evaluate_dead_fine_fuel_moisture(
        baro.temperature_c,
        baro.relative_humidity,
        solar_irradiance,
        moisture.kbdi_drought_index,
    );

    // 4. Fireline intensity under coupled wind, slope, and dynamic fuel moisture
    let mut dynamic_moisture = *moisture;
    dynamic_moisture.dead_fuel_moisture_fraction = dfmc_fraction;
    let fireline = evaluate_fireline_intensity(fuel, &dynamic_moisture, wind_10m_m_s, slope_deg);

    // 5. Combined dry lightning ignition risk: GEC atmospheric electrostatic potential
    //    gated by fuel ignition feasibility (fuels must be drier than extinction threshold)
    let ignition_feasibility = if dfmc_fraction < DEAD_FUEL_EXTINCTION_MOISTURE {
        ((DEAD_FUEL_EXTINCTION_MOISTURE - dfmc_fraction) / DEAD_FUEL_EXTINCTION_MOISTURE).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dry_lightning_ignition_risk = (gec.dry_lightning_susceptibility * ignition_feasibility).clamp(0.0, 1.0);

    HeliophysicalFireWeatherReport {
        storm,
        gec,
        air_density_kg_m3,
        wind_10m_m_s,
        dfmc_fraction,
        fireline,
        dry_lightning_ignition_risk,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benign_moist_conditions() {
        let solar = SchwabeSolarCycle::default();
        let wind = SolarWindState::default();
        let mut baro = BarometricState::default();
        baro.relative_humidity = 0.85;
        baro.temperature_c = 14.0;
        let abl = BoundaryLayerProfile::default();
        let fuel = ForestFuelStratum::mature_dry_sclerophyll(5.0);
        let mut moisture = FuelMoistureState::default();
        moisture.kbdi_drought_index = 15.0;

        let report = evaluate_heliophysical_fire_weather(&solar, &wind, &baro, &abl, &fuel, &moisture, 0.0, 100.0);

        assert!(report.dfmc_fraction >= DEAD_FUEL_EXTINCTION_MOISTURE);
        assert_eq!(report.fireline.regime, FirelineRegime::Extinguished);
        assert_eq!(report.dry_lightning_ignition_risk, 0.0);
        assert!(report.air_density_kg_m3 > 1.15);
    }

    #[test]
    fn test_extreme_solar_storm_drought_pyrocb() {
        let mut solar = SchwabeSolarCycle::default();
        solar.tsi_w_m2 = BASELINE_TSI_W_M2 + 1.25;
        solar.spotless_days_ratio = 0.0;

        let mut wind = SolarWindState::default();
        wind.imf_bz_nt = -16.0; // Deep southward reconnection
        wind.velocity_km_s = 720.0;

        let mut baro = BarometricState::default();
        baro.relative_humidity = 0.07;
        baro.temperature_c = 42.0;

        let mut abl = BoundaryLayerProfile::default();
        abl.friction_velocity_u_star = 2.6; // Gale-force synoptic boundary layer

        let fuel = ForestFuelStratum::mature_dry_sclerophyll(12.0);
        let mut moisture = FuelMoistureState::default();
        moisture.kbdi_drought_index = 185.0;

        let report = evaluate_heliophysical_fire_weather(&solar, &wind, &baro, &abl, &fuel, &moisture, 18.0, 600.0);

        assert!(report.storm.is_reconnection_active);
        assert!(report.storm.kp_index >= 5.0);
        assert!(report.dfmc_fraction < 0.05);
        assert_eq!(report.fireline.regime, FirelineRegime::CatastrophicCrowningPyroCb);
        assert!(report.fireline.fireline_intensity_kw_m > 10_000.0);
        assert!(report.dry_lightning_ignition_risk > 0.4);
    }
}
