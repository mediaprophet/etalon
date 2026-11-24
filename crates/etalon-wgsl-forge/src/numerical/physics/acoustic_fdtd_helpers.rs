//! CPU reference steppers, GPU dispatchers, and verification tests for 3D acoustic FDTD.

use super::*;
use crate::numerical::execute::{BindingUsage, QualiaCompute, WgpuComputeContext, WgpuPipeline};
use crate::numerical::{ForgeError, Schedule};

/// CPU reference stepper for one full 3D staggered fluid FDTD leapfrog step (`ACOUSTIC_FLUID_CELL_STRIDE = 4`).
pub fn acoustic_fluid_fdtd_step_cpu(
    state: &mut [f32],
    params: &AcousticFluidFdtdGpuParams,
) -> Result<(), ForgeError> {
    params.validate()?;
    let expected_len = params.cell_count() * ACOUSTIC_FLUID_CELL_STRIDE;
    if state.len() != expected_len {
        return Err(ForgeError::GpuValidation(format!(
            "acoustic_fluid_fdtd_step_cpu: state length {} != expected {}",
            state.len(),
            expected_len
        )));
    }
    let [nx, ny, nz] = [
        params.dims[0] as usize,
        params.dims[1] as usize,
        params.dims[2] as usize,
    ];
    let [dx, dy, dz] = params.spacing_m;
    let dt = params.dt_s;
    let half_damp = 0.5 * params.damping_per_s * dt;
    let decay = (1.0 - half_damp) / (1.0 + half_damp);
    let v_scale = (dt / params.density_kg_m3) / (1.0 + half_damp);
    let p_scale = (params.bulk_modulus_pa * dt) / (1.0 + half_damp);
    let bnd = params.boundary;
    let base_of = |ix: usize, iy: usize, iz: usize| ((ix * ny + iy) * nz + iz) * 4;

    // Pass 0: update staggered velocities (vx, vy, vz) from pressure gradients.
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let b = base_of(ix, iy, iz);
                let p_curr = state[b];
                state[b + 1] = if ix + 1 < nx {
                    decay * state[b + 1]
                        - v_scale * ((state[base_of(ix + 1, iy, iz)] - p_curr) / dx)
                } else {
                    match bnd {
                        AcousticFluidGpuBoundary::Periodic => {
                            decay * state[b + 1]
                                - v_scale * ((state[base_of(0, iy, iz)] - p_curr) / dx)
                        }
                        AcousticFluidGpuBoundary::Rigid => 0.0,
                        AcousticFluidGpuBoundary::PressureRelease => {
                            decay * state[b + 1] - v_scale * ((-p_curr) / dx)
                        }
                    }
                };
                state[b + 2] = if iy + 1 < ny {
                    decay * state[b + 2]
                        - v_scale * ((state[base_of(ix, iy + 1, iz)] - p_curr) / dy)
                } else {
                    match bnd {
                        AcousticFluidGpuBoundary::Periodic => {
                            decay * state[b + 2]
                                - v_scale * ((state[base_of(ix, 0, iz)] - p_curr) / dy)
                        }
                        AcousticFluidGpuBoundary::Rigid => 0.0,
                        AcousticFluidGpuBoundary::PressureRelease => {
                            decay * state[b + 2] - v_scale * ((-p_curr) / dy)
                        }
                    }
                };
                state[b + 3] = if iz + 1 < nz {
                    decay * state[b + 3]
                        - v_scale * ((state[base_of(ix, iy, iz + 1)] - p_curr) / dz)
                } else {
                    match bnd {
                        AcousticFluidGpuBoundary::Periodic => {
                            decay * state[b + 3]
                                - v_scale * ((state[base_of(ix, iy, 0)] - p_curr) / dz)
                        }
                        AcousticFluidGpuBoundary::Rigid => 0.0,
                        AcousticFluidGpuBoundary::PressureRelease => {
                            decay * state[b + 3] - v_scale * ((-p_curr) / dz)
                        }
                    }
                };
            }
        }
    }

    // Pass 1: update cell-centered pressure p from updated velocity divergence.
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let b = base_of(ix, iy, iz);
                let on_outer =
                    ix == 0 || ix + 1 == nx || iy == 0 || iy + 1 == ny || iz == 0 || iz + 1 == nz;
                if matches!(bnd, AcousticFluidGpuBoundary::PressureRelease) && on_outer {
                    state[b] = 0.0;
                    continue;
                }
                let (vx_curr, vy_curr, vz_curr) = (state[b + 1], state[b + 2], state[b + 3]);
                let vx_m = if ix > 0 {
                    state[base_of(ix - 1, iy, iz) + 1]
                } else {
                    match bnd {
                        AcousticFluidGpuBoundary::Periodic => state[base_of(nx - 1, iy, iz) + 1],
                        AcousticFluidGpuBoundary::Rigid => 0.0,
                        AcousticFluidGpuBoundary::PressureRelease => vx_curr,
                    }
                };
                let vy_m = if iy > 0 {
                    state[base_of(ix, iy - 1, iz) + 2]
                } else {
                    match bnd {
                        AcousticFluidGpuBoundary::Periodic => state[base_of(ix, ny - 1, iz) + 2],
                        AcousticFluidGpuBoundary::Rigid => 0.0,
                        AcousticFluidGpuBoundary::PressureRelease => vy_curr,
                    }
                };
                let vz_m = if iz > 0 {
                    state[base_of(ix, iy, iz - 1) + 3]
                } else {
                    match bnd {
                        AcousticFluidGpuBoundary::Periodic => state[base_of(ix, iy, nz - 1) + 3],
                        AcousticFluidGpuBoundary::Rigid => 0.0,
                        AcousticFluidGpuBoundary::PressureRelease => vz_curr,
                    }
                };
                let div_v = (vx_curr - vx_m) / dx + (vy_curr - vy_m) / dy + (vz_curr - vz_m) / dz;
                state[b] = decay * state[b] - p_scale * div_v;
            }
        }
    }
    Ok(())
}

/// CPU reference stepper for one full 3D Virieux staggered elastic velocity-stress step (`ELASTIC_VIRIEUX_CELL_STRIDE = 9`).
pub fn elastic_virieux_fdtd_step_cpu(
    state: &mut [f32],
    params: &ElasticVirieuxFdtdGpuParams,
) -> Result<(), ForgeError> {
    params.validate()?;
    let expected_len = params.cell_count() * ELASTIC_VIRIEUX_CELL_STRIDE;
    if state.len() != expected_len {
        return Err(ForgeError::GpuValidation(format!(
            "elastic_virieux_fdtd_step_cpu: state length {} != expected {}",
            state.len(),
            expected_len
        )));
    }
    let [nx, ny, nz] = [
        params.dims[0] as usize,
        params.dims[1] as usize,
        params.dims[2] as usize,
    ];
    let [dx, dy, dz] = params.spacing_m;
    let dt = params.dt_s;
    let dt_rho = dt / params.density_kg_m3;
    let lambda = params.lame_lambda_pa();
    let mu = params.shear_modulus_pa;
    let l2m = lambda + 2.0 * mu;
    let periodic = matches!(params.boundary, ElasticVirieuxGpuBoundary::Periodic);
    let base_of = |ix: usize, iy: usize, iz: usize| ((ix * ny + iy) * nz + iz) * 9;

    // Pass 0: update velocities (vx, vy, vz) from Cauchy stress divergence.
    for ix in 0..nx {
        let ixp = if ix + 1 < nx {
            ix + 1
        } else if periodic {
            0
        } else {
            ix
        };
        let ixm = if ix > 0 {
            ix - 1
        } else if periodic {
            nx - 1
        } else {
            0
        };
        for iy in 0..ny {
            let iyp = if iy + 1 < ny {
                iy + 1
            } else if periodic {
                0
            } else {
                iy
            };
            let iym = if iy > 0 {
                iy - 1
            } else if periodic {
                ny - 1
            } else {
                0
            };
            for iz in 0..nz {
                let izp = if iz + 1 < nz {
                    iz + 1
                } else if periodic {
                    0
                } else {
                    iz
                };
                let izm = if iz > 0 {
                    iz - 1
                } else if periodic {
                    nz - 1
                } else {
                    0
                };
                let b = base_of(ix, iy, iz);
                let dvx = (state[base_of(ixp, iy, iz) + 3] - state[b + 3]) / dx
                    + (state[b + 6] - state[base_of(ix, iym, iz) + 6]) / dy
                    + (state[b + 8] - state[base_of(ix, iy, izm) + 8]) / dz;
                let dvy = (state[b + 6] - state[base_of(ixm, iy, iz) + 6]) / dx
                    + (state[base_of(ix, iyp, iz) + 4] - state[b + 4]) / dy
                    + (state[b + 7] - state[base_of(ix, iy, izm) + 7]) / dz;
                let dvz = (state[b + 8] - state[base_of(ixm, iy, iz) + 8]) / dx
                    + (state[b + 7] - state[base_of(ix, iym, iz) + 7]) / dy
                    + (state[base_of(ix, iy, izp) + 5] - state[b + 5]) / dz;
                state[b] += dt_rho * dvx;
                state[b + 1] += dt_rho * dvy;
                state[b + 2] += dt_rho * dvz;
            }
        }
    }

    // Pass 1: update 6-component Cauchy stress from velocity gradients.
    for ix in 0..nx {
        let ixp = if ix + 1 < nx {
            ix + 1
        } else if periodic {
            0
        } else {
            ix
        };
        let ixm = if ix > 0 {
            ix - 1
        } else if periodic {
            nx - 1
        } else {
            0
        };
        for iy in 0..ny {
            let iyp = if iy + 1 < ny {
                iy + 1
            } else if periodic {
                0
            } else {
                iy
            };
            let iym = if iy > 0 {
                iy - 1
            } else if periodic {
                ny - 1
            } else {
                0
            };
            for iz in 0..nz {
                let izp = if iz + 1 < nz {
                    iz + 1
                } else if periodic {
                    0
                } else {
                    iz
                };
                let izm = if iz > 0 {
                    iz - 1
                } else if periodic {
                    nz - 1
                } else {
                    0
                };
                let b = base_of(ix, iy, iz);
                let on_outer =
                    ix == 0 || ix + 1 == nx || iy == 0 || iy + 1 == ny || iz == 0 || iz + 1 == nz;
                if !periodic && on_outer {
                    for s in 3..9 {
                        state[b + s] = 0.0;
                    }
                    continue;
                }
                let (vx, vy, vz) = (state[b], state[b + 1], state[b + 2]);
                let dvx_dx = (vx - state[base_of(ixm, iy, iz)]) / dx;
                let dvy_dy = (vy - state[base_of(ix, iym, iz) + 1]) / dy;
                let dvz_dz = (vz - state[base_of(ix, iy, izm) + 2]) / dz;

                state[b + 3] += dt * (l2m * dvx_dx + lambda * (dvy_dy + dvz_dz));
                state[b + 4] += dt * (l2m * dvy_dy + lambda * (dvx_dx + dvz_dz));
                state[b + 5] += dt * (l2m * dvz_dz + lambda * (dvx_dx + dvy_dy));
                state[b + 6] += dt
                    * mu
                    * ((state[base_of(ix, iyp, iz)] - vx) / dy
                        + (state[base_of(ixp, iy, iz) + 1] - vy) / dx);
                state[b + 7] += dt
                    * mu
                    * ((state[base_of(ix, iy, izp) + 1] - vy) / dz
                        + (state[base_of(ix, iyp, iz) + 2] - vz) / dy);
                state[b + 8] += dt
                    * mu
                    * ((state[base_of(ix, iy, izp)] - vx) / dz
                        + (state[base_of(ixp, iy, iz) + 2] - vz) / dx);
            }
        }
    }
    Ok(())
}

/// Dispatch one full 3D staggered fluid FDTD leapfrog step using a caller-supplied [`WgpuComputeContext`].
pub fn acoustic_fluid_fdtd_step_with_context(
    ctx: &mut WgpuComputeContext,
    state_in: &[f32],
    params: &AcousticFluidFdtdGpuParams,
) -> Result<Vec<f32>, ForgeError> {
    params.validate()?;
    let expected_len = params.cell_count() * ACOUSTIC_FLUID_CELL_STRIDE;
    if state_in.len() != expected_len {
        return Err(ForgeError::GpuValidation(format!(
            "acoustic_fluid_fdtd_step_with_context: state length {} != expected {}",
            state_in.len(),
            expected_len
        )));
    }
    run_two_pass_fdtd_with_context(
        ctx,
        ACOUSTIC_FLUID_FDTD_WGSL,
        ACOUSTIC_FLUID_FDTD_ENTRY,
        state_in,
        params.cell_count(),
        params.to_uniform_words(0),
        params.to_uniform_words(1),
    )
}

/// Convenience GPU wrapper that allocates a [`WgpuComputeContext`] and runs [`acoustic_fluid_fdtd_step_with_context`].
pub fn acoustic_fluid_fdtd_step_gpu(
    state_in: &[f32],
    params: &AcousticFluidFdtdGpuParams,
) -> Result<Vec<f32>, ForgeError> {
    params.validate()?;
    let capacity = (state_in.len() * 16).max(4 << 20);
    let mut ctx = WgpuComputeContext::new(capacity)?;
    acoustic_fluid_fdtd_step_with_context(&mut ctx, state_in, params)
}

/// Dispatch one full 3D Virieux staggered elastic velocity-stress step using a caller-supplied [`WgpuComputeContext`].
pub fn elastic_virieux_fdtd_step_with_context(
    ctx: &mut WgpuComputeContext,
    state_in: &[f32],
    params: &ElasticVirieuxFdtdGpuParams,
) -> Result<Vec<f32>, ForgeError> {
    params.validate()?;
    let expected_len = params.cell_count() * ELASTIC_VIRIEUX_CELL_STRIDE;
    if state_in.len() != expected_len {
        return Err(ForgeError::GpuValidation(format!(
            "elastic_virieux_fdtd_step_with_context: state length {} != expected {}",
            state_in.len(),
            expected_len
        )));
    }
    run_two_pass_fdtd_with_context(
        ctx,
        ELASTIC_VIRIEUX_FDTD_WGSL,
        ELASTIC_VIRIEUX_FDTD_ENTRY,
        state_in,
        params.cell_count(),
        params.to_uniform_words(0),
        params.to_uniform_words(1),
    )
}

/// Convenience GPU wrapper that allocates a [`WgpuComputeContext`] and runs [`elastic_virieux_fdtd_step_with_context`].
pub fn elastic_virieux_fdtd_step_gpu(
    state_in: &[f32],
    params: &ElasticVirieuxFdtdGpuParams,
) -> Result<Vec<f32>, ForgeError> {
    params.validate()?;
    let capacity = (state_in.len() * 16).max(4 << 20);
    let mut ctx = WgpuComputeContext::new(capacity)?;
    elastic_virieux_fdtd_step_with_context(&mut ctx, state_in, params)
}

fn run_two_pass_fdtd_with_context(
    ctx: &mut WgpuComputeContext,
    wgsl_source: &str,
    entry_point: &str,
    state_in: &[f32],
    cell_count: usize,
    params_pass0: [f32; 12],
    params_pass1: [f32; 12],
) -> Result<Vec<f32>, ForgeError> {
    let len = state_in.len();
    let zeros = vec![0.0f32; len];
    let schedule = Schedule {
        workgroup_size: 64,
        items_per_invocation: 1,
        vector_width: 1,
    };

    ctx.clear_transient_allocations();
    let v_in0 = ctx.allocate_and_write(
        bytemuck::cast_slice(state_in),
        0,
        0,
        BindingUsage::StorageRead,
    )?;
    let v_mid0 = ctx.allocate_and_write(
        bytemuck::cast_slice(&zeros),
        1,
        0,
        BindingUsage::StorageReadWrite,
    )?;
    let v_p0 = ctx.allocate_and_write(
        bytemuck::cast_slice(&params_pass0),
        2,
        0,
        BindingUsage::StorageRead,
    )?;
    {
        let pipe0 = WgpuPipeline::compile(ctx, wgsl_source, entry_point)?;
        pipe0.dispatch(&[v_in0, v_mid0, v_p0], &schedule, cell_count)?;
    }
    let mut mid = ctx.read_buffer_f32(&v_mid0)?;
    mid.truncate(len);

    ctx.clear_transient_allocations();
    let v_in1 =
        ctx.allocate_and_write(bytemuck::cast_slice(&mid), 0, 0, BindingUsage::StorageRead)?;
    let v_out1 = ctx.allocate_and_write(
        bytemuck::cast_slice(&zeros),
        1,
        0,
        BindingUsage::StorageReadWrite,
    )?;
    let v_p1 = ctx.allocate_and_write(
        bytemuck::cast_slice(&params_pass1),
        2,
        0,
        BindingUsage::StorageRead,
    )?;
    {
        let pipe1 = WgpuPipeline::compile(ctx, wgsl_source, entry_point)?;
        pipe1.dispatch(&[v_in1, v_out1, v_p1], &schedule, cell_count)?;
    }
    let mut out = ctx.read_buffer_f32(&v_out1)?;
    out.truncate(len);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_kernels_validate_cleanly() {
        let rep_fluid =
            crate::validate_wgsl(ACOUSTIC_FLUID_FDTD_WGSL).expect("fluid FDTD WGSL valid");
        assert_eq!(
            rep_fluid.entry_points,
            vec![ACOUSTIC_FLUID_FDTD_ENTRY.to_string()]
        );
        assert_eq!(rep_fluid.binding_count, 3);

        let rep_elastic =
            crate::validate_wgsl(ELASTIC_VIRIEUX_FDTD_WGSL).expect("elastic Virieux WGSL valid");
        assert_eq!(
            rep_elastic.entry_points,
            vec![ELASTIC_VIRIEUX_FDTD_ENTRY.to_string()]
        );
        assert_eq!(rep_elastic.binding_count, 3);
    }

    #[test]
    fn courant_and_capacity_gates_reject_invalid_inputs() {
        let bad_fluid = AcousticFluidFdtdGpuParams {
            dims: [8, 8, 8],
            spacing_m: [0.05, 0.05, 0.05],
            dt_s: 1.0,
            density_kg_m3: 1000.0,
            bulk_modulus_pa: 2.25e9,
            damping_per_s: 0.0,
            boundary: AcousticFluidGpuBoundary::Periodic,
        };
        assert!(bad_fluid.validate().is_err());

        let bad_elastic = ElasticVirieuxFdtdGpuParams {
            dims: [8, 8, 8],
            spacing_m: [0.05, 0.05, 0.05],
            dt_s: 1.0,
            density_kg_m3: 2000.0,
            bulk_modulus_pa: 8.0e9,
            shear_modulus_pa: 3.0e9,
            boundary: ElasticVirieuxGpuBoundary::Periodic,
        };
        assert!(bad_elastic.validate().is_err());
    }

    #[test]
    fn cpu_reference_fluid_and_elastic_steps_propagate_wave() {
        let fluid_params = AcousticFluidFdtdGpuParams {
            dims: [6, 6, 6],
            spacing_m: [0.1, 0.1, 0.1],
            dt_s: 1e-5,
            density_kg_m3: 1000.0,
            bulk_modulus_pa: 2.25e9,
            damping_per_s: 0.0,
            boundary: AcousticFluidGpuBoundary::Periodic,
        };
        let mut fluid_state = vec![0.0f32; fluid_params.cell_count() * ACOUSTIC_FLUID_CELL_STRIDE];
        fluid_state[((3 * 6 + 3) * 6 + 3) * 4] = 1000.0;
        acoustic_fluid_fdtd_step_cpu(&mut fluid_state, &fluid_params).unwrap();
        assert!(fluid_state.iter().any(|&v| v != 0.0));

        let elastic_params = ElasticVirieuxFdtdGpuParams {
            dims: [6, 6, 6],
            spacing_m: [0.1, 0.1, 0.1],
            dt_s: 5e-6,
            density_kg_m3: 2000.0,
            bulk_modulus_pa: 8.0e9,
            shear_modulus_pa: 3.0e9,
            boundary: ElasticVirieuxGpuBoundary::FreeSurface,
        };
        let mut elastic_state =
            vec![0.0f32; elastic_params.cell_count() * ELASTIC_VIRIEUX_CELL_STRIDE];
        elastic_state[((3 * 6 + 3) * 6 + 3) * 9] = 0.05;
        elastic_virieux_fdtd_step_cpu(&mut elastic_state, &elastic_params).unwrap();
        assert!(elastic_state.iter().any(|&v| v != 0.0));
    }

    #[test]
    #[serial_test::serial(gpu)]
    fn gpu_matches_cpu_oracle_fluid_and_elastic() {
        if !crate::numerical::test_gpu_available() {
            return;
        }
        let fluid_params = AcousticFluidFdtdGpuParams {
            dims: [4, 4, 4],
            spacing_m: [0.1, 0.1, 0.1],
            dt_s: 1e-5,
            density_kg_m3: 1000.0,
            bulk_modulus_pa: 2.25e9,
            damping_per_s: 0.5,
            boundary: AcousticFluidGpuBoundary::Periodic,
        };
        let mut fluid_cpu = vec![0.0f32; fluid_params.cell_count() * ACOUSTIC_FLUID_CELL_STRIDE];
        fluid_cpu[((2 * 4 + 2) * 4 + 2) * 4] = 500.0;
        let fluid_gpu = acoustic_fluid_fdtd_step_gpu(&fluid_cpu, &fluid_params).unwrap();
        acoustic_fluid_fdtd_step_cpu(&mut fluid_cpu, &fluid_params).unwrap();
        for (a, b) in fluid_cpu.iter().zip(fluid_gpu.iter()) {
            assert!((a - b).abs() <= 1e-4, "fluid GPU/CPU mismatch: {a} vs {b}");
        }

        let elastic_params = ElasticVirieuxFdtdGpuParams {
            dims: [4, 4, 4],
            spacing_m: [0.1, 0.1, 0.1],
            dt_s: 5e-6,
            density_kg_m3: 2000.0,
            bulk_modulus_pa: 8.0e9,
            shear_modulus_pa: 3.0e9,
            boundary: ElasticVirieuxGpuBoundary::Periodic,
        };
        let mut elastic_cpu =
            vec![0.0f32; elastic_params.cell_count() * ELASTIC_VIRIEUX_CELL_STRIDE];
        elastic_cpu[((2 * 4 + 2) * 4 + 2) * 9] = 0.02;
        let elastic_gpu = elastic_virieux_fdtd_step_gpu(&elastic_cpu, &elastic_params).unwrap();
        elastic_virieux_fdtd_step_cpu(&mut elastic_cpu, &elastic_params).unwrap();
        for (a, b) in elastic_cpu.iter().zip(elastic_gpu.iter()) {
            assert!(
                (a - b).abs() <= 1e-3,
                "elastic GPU/CPU mismatch: {a} vs {b}"
            );
        }
    }
}
