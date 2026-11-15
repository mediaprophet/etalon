//! Unit Tests for `etalon_sim::maxwell_emf`.

use super::dispersion_ade::{DebyePole, DrudePole, LorentzPole};
use super::poynting_energy::{
    admit_stokes_vector, c02_reduction_f64_budget, c02_small_op_f64_budget,
    convert_iq_emf_to_qac_conjugated, dft_forward_unscaled, instantaneous_poynting_and_energy,
    jones_to_stokes_candidate_v1, mode0_carrier_wavelength_m,
    mode6_photon_attenuation_and_independent_dose, mode7_quasi_static_dipole_potential_v,
    mueller_apply, mueller_ideal_linear_polarizer_x, peak_phasor_time_average_poynting,
    ray_geometric_delay_s, reconstruct_rf_emf_negative_carrier,
    reconstruct_rf_qac_positive_carrier, rms_phasor_time_average_poynting,
    rotate_eb_hamilton_xyzw, rotate_stokes_basis_passive,
    spectral_density_wavelength_midpoint_integral, verify_discrete_poynting_power_balance,
};
use super::vector_calculus::{
    spectral_plane_wave_curl, verify_discrete_charge_conservation, verify_div_curl_zero_2nd,
    verify_stokes_circulation_cell, yee_curl_e_2nd, yee_curl_e_4th,
};
use super::yee_fdtd::{
    admit_yee_timestep, run_yee_periodic_plane_wave_fixture, yee_cfl_limit_3d,
    yee_step_leapfrog_in_place, YeeMaterialParams, YeeSolverProfile,
};
use super::{
    BoundaryCondition, EmfOperatorError, Grid3dSpec, C0_M_PER_S, EPS0_REF_F_PER_M, Z0_REF_OHM,
};

#[test]
fn test_etalon_sim_maxwell_emf_oracles_and_yee_convergence() {
    let mat = YeeMaterialParams::vacuum();
    let (s, u) =
        instantaneous_poynting_and_energy([1.0, 0.0, 0.0], [0.0, 1.0 / C0_M_PER_S, 0.0], &mat)
            .unwrap();
    let q_s = 1.0 / Z0_REF_OHM;
    let q_u = EPS0_REF_F_PER_M;
    assert!((s[2] - q_s).abs() <= c02_small_op_f64_budget(q_s));
    assert!((u - q_u).abs() <= c02_small_op_f64_budget(q_u));

    let e_peak = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
    let h_peak = [(0.0, 0.0), (1.0 / Z0_REF_OHM, 0.0), (0.0, 0.0)];
    let inv_sqrt2 = core::f64::consts::FRAC_1_SQRT_2;
    let e_rms = [(inv_sqrt2, 0.0), (0.0, 0.0), (0.0, 0.0)];
    let h_rms = [(0.0, 0.0), (inv_sqrt2 / Z0_REF_OHM, 0.0), (0.0, 0.0)];
    let s_peak_avg = peak_phasor_time_average_poynting(e_peak, h_peak).unwrap();
    let s_rms_avg = rms_phasor_time_average_poynting(e_rms, h_rms).unwrap();
    assert!((s_peak_avg[2] - s_rms_avg[2]).abs() <= c02_small_op_f64_budget(q_s));
    assert_eq!(
        mode0_carrier_wavelength_m(0.0, false),
        Err(EmfOperatorError::StaticCarrierNoWavelength)
    );

    let (e_rot, b_rot, _) = rotate_eb_hamilton_xyzw(
        [0.0, 0.0, inv_sqrt2, inv_sqrt2],
        [1.0, 0.0, 0.0],
        [0.0, 1.0 / C0_M_PER_S, 0.0],
    )
    .unwrap();
    assert!((e_rot[1] - 1.0).abs() <= c02_small_op_f64_budget(1.0));
    assert!((b_rot[0] + 1.0 / C0_M_PER_S).abs() <= c02_small_op_f64_budget(1.0 / C0_M_PER_S));

    let stokes_circ = jones_to_stokes_candidate_v1((inv_sqrt2, 0.0), (0.0, inv_sqrt2)).unwrap();
    assert!((stokes_circ[3] - (-1.0)).abs() <= c02_small_op_f64_budget(1.0));
    let stokes_rot =
        rotate_stokes_basis_passive([1.0, 1.0, 0.0, 0.0], core::f64::consts::FRAC_PI_4).unwrap();
    assert!((stokes_rot[2] - (-1.0)).abs() <= c02_small_op_f64_budget(1.0));
    assert!(
        (ray_geometric_delay_s(0.299_792_458, &mat).unwrap() - 1e-9).abs()
            <= c02_small_op_f64_budget(1e-9)
    );
    assert!(admit_stokes_vector([1.0, 1.001, 0.0, 0.0]).is_err());

    let grid32 = Grid3dSpec {
        nx: 32,
        ny: 32,
        nz: 32,
        dx: 1.0 / 32.0,
        dy: 1.0 / 32.0,
        dz: 1.0 / 32.0,
    };
    let cfl = yee_cfl_limit_3d(&grid32, &mat).unwrap();
    assert!(admit_yee_timestep(&grid32, &mat, 1.01 * cfl).is_err());

    let r32 = run_yee_periodic_plane_wave_fixture(32, 1.0, 0.5, &mat).unwrap();
    let r64 = run_yee_periodic_plane_wave_fixture(64, 1.0, 0.5, &mat).unwrap();
    let r128 = run_yee_periodic_plane_wave_fixture(128, 1.0, 0.5, &mat).unwrap();
    assert!(r32.one_step_rel_error <= 1e-10);
    assert!(
        (r32.continuum_phase_error_rad / r64.continuum_phase_error_rad).log2() >= 1.8
            && (r64.continuum_phase_error_rad / r128.continuum_phase_error_rad).log2() >= 1.8
    );
    assert!(r128.continuum_phase_error_rad <= 0.002);

    let (analytic, mid256) =
        spectral_density_wavelength_midpoint_integral(1e12, 2e12, 1e-12, 256).unwrap();
    assert!((analytic - 1.0).abs() <= c02_small_op_f64_budget(1.0));
    assert!((mid256 - 1.0).abs() <= 1e-5);

    let m_pol = mueller_ideal_linear_polarizer_x();
    assert_eq!(
        mueller_apply(&m_pol, [2.0, 0.0, 0.0, 0.0]).unwrap(),
        [1.0, 1.0, 0.0, 0.0]
    );

    let mut z = [(0.0_f64, 0.0_f64); 8];
    for (n, item) in z.iter_mut().enumerate() {
        let ang = 2.0 * core::f64::consts::PI * (n as f64) / 8.0;
        *item = (ang.cos(), ang.sin());
    }
    let mut bins = [(0.0_f64, 0.0_f64); 8];
    dft_forward_unscaled(&z, None, &mut bins).unwrap();
    assert!((bins[1].0 - 8.0).hypot(bins[1].1) <= c02_reduction_f64_budget(8, 8.0));

    let iq_emf = (0.6_f64, -0.8_f64);
    let iq_qac = convert_iq_emf_to_qac_conjugated(iq_emf);
    assert!(
        (reconstruct_rf_emf_negative_carrier(iq_emf, 10_000.0, 2.5e-5, 0.3)
            - reconstruct_rf_qac_positive_carrier(iq_qac, 10_000.0, 2.5e-5, 0.3))
        .abs()
            <= c02_small_op_f64_budget(1.0)
    );

    let (mu, flu_out, dose_gy, e_j) = mode6_photon_attenuation_and_independent_dose(
        1000.0, 0.01, 0.2, 100.0, 0.002, 0.01, 1e5,
    )
    .unwrap();
    assert!((mu - 10.0).abs() <= c02_small_op_f64_budget(10.0));
    assert!((flu_out - 13.533_528_323_661_27).abs() <= c02_small_op_f64_budget(100.0));
    assert!((dose_gy - 0.2).abs() <= c02_small_op_f64_budget(0.2));
    assert!((e_j - 1.602_176_634e-14).abs() <= c02_small_op_f64_budget(1.602_176_634e-14));

    let phi_ax =
        mode7_quasi_static_dipole_potential_v([0.0, 0.0, 1e-8], [0.0; 3], [0.0, 0.0, 0.1], 1.0)
            .unwrap();
    assert!(
        (phi_ax - 7.957_747_154_594_767e-8).abs()
            <= c02_small_op_f64_budget(7.957_747_154_594_767e-8)
    );

    let debye = DebyePole {
        delta_epsilon: 2.0,
        tau_s: 1e-10,
    };
    assert!(debye.continuous_susceptibility(1e10).unwrap().1 > 0.0);
    let drude = DrudePole {
        omega_p_rad_s: 1e15,
        gamma_rad_s: 1e13,
    };
    assert!(drude.continuous_susceptibility(2e15).unwrap().0 < 0.0);
    let lorentz = LorentzPole {
        delta_epsilon: 1.5,
        omega_0_rad_s: 1e14,
        gamma_rad_s: 1e12,
    };
    assert!(lorentz.continuous_susceptibility(5e13).unwrap().0 > 0.0);
}

#[test]
fn test_etalon_sim_vector_calculus_and_charge_conservation() {
    let grid = Grid3dSpec {
        nx: 8,
        ny: 8,
        nz: 8,
        dx: 0.125,
        dy: 0.125,
        dz: 0.125,
    };
    let n = grid.validate().unwrap();
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut ex = vec![0.0; n];
    let mut ey = vec![0.0; n];
    let mut ez = vec![0.0; n];
    let mut hx = vec![0.0; n];
    let mut hy = vec![0.0; n];
    let mut hz = vec![0.0; n];
    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let x = (ix as f64) * grid.dx;
                ey[c] = (two_pi * x).cos();
                hz[c] = (two_pi * x).sin();
            }
        }
    }
    let mut cx = vec![0.0; n];
    let mut cy = vec![0.0; n];
    let mut cz = vec![0.0; n];
    let mut div_s = vec![0.0; n];
    let (me, mh) = verify_div_curl_zero_2nd(
        &grid,
        BoundaryCondition::Periodic,
        &ex,
        &ey,
        &ez,
        &hx,
        &hy,
        &hz,
        &mut cx,
        &mut cy,
        &mut cz,
        &mut div_s,
    )
    .unwrap();
    assert!(me <= 1e-12 && mh <= 1e-12);

    yee_curl_e_2nd(
        &grid,
        BoundaryCondition::Periodic,
        &ex,
        &ey,
        &ez,
        &mut cx,
        &mut cy,
        &mut cz,
    )
    .unwrap();
    let stokes = verify_stokes_circulation_cell(
        &grid,
        BoundaryCondition::Periodic,
        &ex,
        &ey,
        &ez,
        &cx,
        &cy,
        &cz,
        2,
        2,
        2,
    )
    .unwrap();
    assert!(stokes.iter().all(|&r| r <= 1e-15));

    let mut cz4 = vec![0.0; n];
    yee_curl_e_4th(
        &grid,
        BoundaryCondition::Periodic,
        &ex,
        &ey,
        &ez,
        &mut cx,
        &mut cy,
        &mut cz4,
    )
    .unwrap();
    let exact = spectral_plane_wave_curl(
        [two_pi, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5 * grid.dx, 0.0, 0.0],
        0.0,
    )[2];
    assert!((cz4[0] - exact).abs() < (cz[0] - exact).abs());

    let mat = YeeMaterialParams::vacuum();
    let dt = 0.5 * yee_cfl_limit_3d(&grid, &mat).unwrap();
    let eps = mat.epsilon_f_per_m();
    let ex_n = ex.clone();
    let ey_n = ey.clone();
    let ez_n = ez.clone();
    let hx_h = hx.clone();
    let hy_h = hy.clone();
    let hz_h = hz.clone();
    let jx = vec![1e-5; n];
    let jy = vec![-2e-5; n];
    let jz = vec![1.5e-5; n];
    yee_step_leapfrog_in_place(
        &grid,
        YeeSolverProfile::LosslessPeriodicNoPhiV1,
        BoundaryCondition::Periodic,
        &mat,
        dt,
        &mut ex,
        &mut ey,
        &mut ez,
        &mut hx,
        &mut hy,
        &mut hz,
        &jx,
        &jy,
        &jz,
        &mut cx,
        &mut cy,
        &mut cz,
    )
    .unwrap();
    let dx_n: Vec<f64> = ex_n.iter().map(|&v| eps * v).collect();
    let dy_n: Vec<f64> = ey_n.iter().map(|&v| eps * v).collect();
    let dz_n: Vec<f64> = ez_n.iter().map(|&v| eps * v).collect();
    let dx_np1: Vec<f64> = ex.iter().map(|&v| eps * v).collect();
    let dy_np1: Vec<f64> = ey.iter().map(|&v| eps * v).collect();
    let dz_np1: Vec<f64> = ez.iter().map(|&v| eps * v).collect();
    let mut s1 = vec![0.0; n];
    let mut s2 = vec![0.0; n];
    let mut s3 = vec![0.0; n];
    assert!(
        verify_discrete_charge_conservation(
            &grid,
            BoundaryCondition::Periodic,
            dt,
            &dx_n,
            &dy_n,
            &dz_n,
            &dx_np1,
            &dy_np1,
            &dz_np1,
            &jx,
            &jy,
            &jz,
            &mut s1,
            &mut s2,
            &mut s3,
        )
        .unwrap()
            <= 1e-13
    );
    assert!(
        verify_discrete_poynting_power_balance(
            &grid,
            BoundaryCondition::Periodic,
            &mat,
            dt,
            &ex_n,
            &ey_n,
            &ez_n,
            &ex,
            &ey,
            &ez,
            &hx_h,
            &hy_h,
            &hz_h,
            &jx,
            &jy,
            &jz,
            &mut cx,
            &mut cy,
            &mut cz,
        )
        .unwrap()
            <= 1e-13
    );
}
