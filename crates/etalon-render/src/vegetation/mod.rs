//! Vegetation simulation, physical bending, leaf biophysics, FSPM metamers, and ground cover.

pub mod bending;
pub mod flora_archetype;
pub mod ground_cover;
pub mod leaf_biophysics;
pub mod metamer_fspm;

pub use bending::*;
pub use flora_archetype::*;
pub use ground_cover::*;
pub use leaf_biophysics::*;
pub use metamer_fspm::*;

/// Derives physical vegetation bending and wind response parameters directly from
/// a live biophysical flora archetype evaluator (coupling turgor, lignification, and Young's modulus).
pub fn derive_bending_params_from_flora(
    evaluator: &FloraBiomechanicsEvaluator,
    wind_direction: [f32; 2],
    wind_strength: f32,
) -> VegetationBendingParams {
    let stiffness = evaluator.effective_stiffness_modulus().clamp(0.05, 0.98);
    let recovery_rate = 2.0 + 8.0 * stiffness;
    let max_bend_angle = (1.4 - 0.9 * stiffness).clamp(0.2, 1.4);

    VegetationBendingParams {
        stiffness,
        recovery_rate,
        max_bend_angle,
        wind_direction,
        wind_strength,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_bending_params_from_flora() {
        // Woody tree has high lignification -> high stiffness, fast recovery, small max bend
        let tree_eval = FloraBiomechanicsEvaluator::new(FloraArchetypeConfig::woody_tree(), 10.0);
        let tree_params = derive_bending_params_from_flora(&tree_eval, [1.0, 0.0], 0.5);

        assert!(tree_params.stiffness > 0.8, "Tree must be stiff");
        assert!(tree_params.recovery_rate > 8.0, "Tree must recover quickly");
        assert!(tree_params.max_bend_angle < 0.7, "Tree must resist large deflections");

        // Flaccid herb under drought -> low stiffness, slow recovery, high max bend
        let mut herb_eval = FloraBiomechanicsEvaluator::new(FloraArchetypeConfig::culinary_herb(), 0.5);
        herb_eval.turgor_pressure_mpa = 0.0;
        let herb_params = derive_bending_params_from_flora(&herb_eval, [0.0, 1.0], 0.2);

        assert!(herb_params.stiffness < 0.2, "Wilted herb must be flexible");
        assert!(herb_params.recovery_rate < 4.0, "Wilted herb must recover slowly");
        assert!(herb_params.max_bend_angle > 1.0, "Wilted herb must bend far under force");
    }
}
