//! Zero-Heap 4D RK4 Driven Spring & Thermal ODE Response Kernel (`response_ode.rs`).
//!
//! Consolidates `DrivenSpring`, `SpringState`, `step_spring`, `ThermalTargets`, and `step_thermal`
//! with a self-contained zero-heap 4-state 4th-order Runge–Kutta integrator [`Rk4Integrator4D`].

/// Error type returned by [`Rk4Integrator4D`], [`step_spring`], and [`step_thermal`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseOdeError {
    InvalidParameters = 0,
    ConvergenceFailed = 1,
    ComputationError = 2,
}

/// Result alias for ODE response kernels.
pub type ResponseOdeResult<T> = Result<T, ResponseOdeError>;

/// 4-state ODE derivative trait for [`Rk4Integrator4D`].
pub trait ODEFunction4D {
    fn derivatives(&self, t: f64, y: &[f64; 4]) -> [f64; 4];
}

/// 4-state ODE snapshot.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ODEState4D {
    pub y: [f64; 4],
    pub t: f64,
    pub dt: f64,
}

/// Zero-heap 4th-order Runge–Kutta ODE integrator with fixed `[f64; 4]` state.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Rk4Integrator4D {
    pub state: [f64; 4],
    pub dt: f64,
    pub t: f64,
    pub max_iterations: u32,
    pub iteration: u32,
}

impl Rk4Integrator4D {
    #[inline]
    pub const fn new(dt: f64, max_iterations: u32) -> Self {
        Self { state: [0.0; 4], dt, t: 0.0, max_iterations, iteration: 0 }
    }

    pub fn integrate<F: ODEFunction4D>(
        &mut self,
        f: &F,
        t0: f64,
        y0: [f64; 4],
        t_final: f64,
    ) -> ResponseOdeResult<ODEState4D> {
        if !self.dt.is_finite()
            || self.dt <= 0.0
            || !t0.is_finite()
            || !t_final.is_finite()
            || t_final < t0
            || y0.iter().any(|v| !v.is_finite())
        {
            return Err(ResponseOdeError::InvalidParameters);
        }
        self.state = y0;
        self.t = t0;
        self.iteration = 0;

        while self.t < t_final {
            if self.iteration >= self.max_iterations {
                return Err(ResponseOdeError::ConvergenceFailed);
            }
            let step = (t_final - self.t).min(self.dt);
            self.rk4_step(f, step)?;
            self.iteration += 1;
        }
        Ok(ODEState4D { y: self.state, t: self.t, dt: self.dt })
    }

    #[inline]
    fn rk4_step<F: ODEFunction4D>(&mut self, f: &F, dt: f64) -> ResponseOdeResult<()> {
        let k1 = f.derivatives(self.t, &self.state);
        if k1.iter().any(|v| !v.is_finite()) {
            return Err(ResponseOdeError::ComputationError);
        }
        let mut y_tmp = [0.0; 4];
        for i in 0..4 {
            y_tmp[i] = self.state[i] + 0.5 * dt * k1[i];
        }
        let k2 = f.derivatives(self.t + 0.5 * dt, &y_tmp);
        if k2.iter().any(|v| !v.is_finite()) {
            return Err(ResponseOdeError::ComputationError);
        }
        for i in 0..4 {
            y_tmp[i] = self.state[i] + 0.5 * dt * k2[i];
        }
        let k3 = f.derivatives(self.t + 0.5 * dt, &y_tmp);
        if k3.iter().any(|v| !v.is_finite()) {
            return Err(ResponseOdeError::ComputationError);
        }
        for i in 0..4 {
            y_tmp[i] = self.state[i] + dt * k3[i];
        }
        let k4 = f.derivatives(self.t + dt, &y_tmp);
        if k4.iter().any(|v| !v.is_finite()) {
            return Err(ResponseOdeError::ComputationError);
        }
        for i in 0..4 {
            self.state[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        if self.state.iter().any(|v| !v.is_finite()) {
            return Err(ResponseOdeError::ComputationError);
        }
        self.t += dt;
        Ok(())
    }
}

/// 1D spring position and velocity state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringState {
    pub position: f64,
    pub velocity: f64,
}

/// Stiffness and damping per unit mass: `stiffness * (target - position) - damping * velocity`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrivenSpring {
    pub target: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub max_displacement: f64,
    pub max_speed: f64,
}

impl ODEFunction4D for DrivenSpring {
    #[inline]
    fn derivatives(&self, _t: f64, y: &[f64; 4]) -> [f64; 4] {
        [y[1], self.stiffness * (self.target - y[0]) - self.damping * y[1], 0.0, 0.0]
    }
}

/// Steps a [`DrivenSpring`] over `dt` (`0.0..=0.05` s), substepped at $\le 1/240$ s via [`Rk4Integrator4D`].
pub fn step_spring(state: SpringState, model: DrivenSpring, dt: f64) -> ResponseOdeResult<SpringState> {
    if [
        state.position,
        state.velocity,
        model.target,
        model.stiffness,
        model.damping,
        model.max_displacement,
        model.max_speed,
        dt,
    ]
    .iter()
    .any(|n| !n.is_finite())
        || !(0.0..=0.05).contains(&dt)
        || !(0.0..=1000.0).contains(&model.stiffness)
        || !(0.0..=100.0).contains(&model.damping)
        || model.max_displacement <= 0.0
        || model.max_speed <= 0.0
    {
        return Err(ResponseOdeError::InvalidParameters);
    }
    if dt == 0.0 {
        return Ok(state);
    }
    let mut solver = Rk4Integrator4D::new(dt.min(1.0 / 240.0), 16);
    let integrated = solver.integrate(&model, 0.0, [state.position, state.velocity, 0.0, 0.0], dt)?;
    let lower = model.target - model.max_displacement;
    let upper = model.target + model.max_displacement;
    let position = integrated.y[0].clamp(lower, upper);
    let mut velocity = integrated.y[1].clamp(-model.max_speed, model.max_speed);
    if (position == lower && velocity < 0.0) || (position == upper && velocity > 0.0) {
        velocity = 0.0;
    }
    Ok(SpringState { position, velocity })
}

/// Two-channel driven thermal reservoir targets and relaxation rates ($s^{-1}$).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalTargets {
    pub temperatures: [f64; 2],
    pub rates: [f64; 2],
}

impl ODEFunction4D for ThermalTargets {
    #[inline]
    fn derivatives(&self, _t: f64, y: &[f64; 4]) -> [f64; 4] {
        [(self.temperatures[0] - y[0]) * self.rates[0], (self.temperatures[1] - y[1]) * self.rates[1], 0.0, 0.0]
    }
}

/// Steps two thermal channels toward `targets` over `dt` (`0.0..=0.05` s) via [`Rk4Integrator4D`].
pub fn step_thermal(temperatures: [f64; 2], targets: ThermalTargets, dt: f64) -> ResponseOdeResult<[f64; 2]> {
    if temperatures
        .iter()
        .chain(targets.temperatures.iter())
        .chain(targets.rates.iter())
        .any(|n| !n.is_finite())
        || !dt.is_finite()
        || !(0.0..=0.05).contains(&dt)
        || targets.rates.iter().any(|n| !(0.0..=10.0).contains(n))
    {
        return Err(ResponseOdeError::InvalidParameters);
    }
    if dt == 0.0 {
        return Ok(temperatures);
    }
    let mut solver = Rk4Integrator4D::new(dt.min(1.0 / 240.0), 16);
    let state = solver.integrate(&targets, 0.0, [temperatures[0], temperatures[1], 0.0, 0.0], dt)?;
    Ok([state.y[0], state.y[1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> DrivenSpring {
        DrivenSpring { target: 1.0, stiffness: 25.0, damping: 10.0, max_displacement: 2.0, max_speed: 10.0 }
    }

    #[test]
    fn driven_spring_and_thermal_ode_match_analytic_references() {
        let mut state = SpringState { position: 0.0, velocity: 0.0 };
        assert_eq!(step_spring(state, model(), 0.0).unwrap(), state);
        let one = step_spring(state, model(), 0.05).unwrap();
        let two = step_spring(step_spring(state, model(), 0.025).unwrap(), model(), 0.025).unwrap();
        assert!((one.position - two.position).abs() < 1e-12 && (one.velocity - two.velocity).abs() < 1e-12);

        for _ in 0..20 {
            state = step_spring(state, model(), 0.05).unwrap();
        }
        assert!((state.position - (1.0 - 6.0 * (-5.0_f64).exp())).abs() < 1e-7);
        assert!((state.velocity - 25.0 * (-5.0_f64).exp()).abs() < 1e-7);

        let mut clamped_m = model();
        clamped_m.max_displacement = 0.1;
        let clamped = step_spring(SpringState { position: 4.0, velocity: 9.0 }, clamped_m, 0.01).unwrap();
        assert_eq!((clamped.position, clamped.velocity), (1.1, 0.0));
        assert_eq!(step_spring(state, model(), 0.051), Err(ResponseOdeError::InvalidParameters));

        let initial = [140.0, 112.0];
        let targets = ThermalTargets { temperatures: [85.0, 75.0], rates: [0.028, 0.04] };
        assert_eq!(step_thermal(initial, targets, 0.0).unwrap(), initial);
        let mut t_state = initial;
        for _ in 0..200 {
            t_state = step_thermal(t_state, targets, 0.05).unwrap();
        }
        for i in 0..2 {
            let expected = targets.temperatures[i] + (initial[i] - targets.temperatures[i]) * (-targets.rates[i] * 10.0).exp();
            assert!((t_state[i] - expected).abs() < 1e-9);
        }
    }
}
