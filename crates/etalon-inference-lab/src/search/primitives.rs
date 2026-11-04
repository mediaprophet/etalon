//! Search and optimization engine: Sobol quasi-random initial design,
//! a k-NN surrogate with Expected Improvement acquisition (Bayesian-style),
//! and a Track-and-Stop multi-armed bandit for A/B testing.
//!
//! The ask-and-tell API:
//! 1. Create a `SearchEngine` with a `ConfigurationSpace` and budget.
//! 2. Call `ask()` to get the next configuration to try.
//! 3. Run the experiment and call `tell(result)` with the outcome.
//! 4. Repeat until budget exhausted. Call `best()` for the Pareto-optimal result.

// ── Sobol quasi-random sequence ───────────────────────────────────────────────

/// A Sobol quasi-random sequence generator for uniform coverage of the \[0,1\]^d
/// unit hypercube. Uses the Joe & Kuo direction numbers (compact implementation).
pub struct SobolSequence {
    dim: usize,
    index: u64,
    /// Direction numbers (bit-packed). For dim ≤ 20 this is sufficient.
    direction: Vec<u64>,
}

impl SobolSequence {
    /// Creates `SobolSequence` with the given state.
    pub fn new(dim: usize) -> Self {
        let direction = sobol_direction_numbers(dim);
        Self {
            dim,
            index: 0,
            direction,
        }
    }

    /// Generate the next point in \[0,1\]^d.
    pub fn next(&mut self) -> Vec<f64> {
        self.index += 1;
        let mut x = vec![0u64; self.dim];
        let mut v = self.index;
        let mut bit = 0usize;
        while v > 0 {
            if v & 1 != 0 {
                for d in 0..self.dim {
                    x[d] ^= self.direction[d * 64 + bit];
                }
            }
            v >>= 1;
            bit += 1;
        }
        x.iter().map(|&v| (v as f64) / (u64::MAX as f64)).collect()
    }

    /// Reset the sequence to the beginning.
    pub fn reset(&mut self) {
        self.index = 0;
    }
}

/// Precomputed Sobol direction numbers for up to 20 dimensions.
/// These are the first 20 Joe & Kuo primitive polynomials + direction numbers.
fn sobol_direction_numbers(dim: usize) -> Vec<u64> {
    // Each dimension d has direction numbers v[d][j] for j = 0..63.
    // We store them as a flat array: direction[d * 64 + j].
    // For compactness, we use the standard initialization for the first 20 dims.
    // Source: Joe & Kuo (2008), "Constructing Sobol sequences with better two-dimensional projections"
    let mut dir = vec![0u64; dim * 64];

    // Initialize dimension 0: v[0][j] = 1 (the identity sequence).
    if dim > 0 {
        for j in 0..64 {
            dir[j] = 1u64 << (63 - j);
        }
    }

    // For dimensions 1..dim-1, use primitive polynomials and initial direction numbers.
    // (p, m, initial v values) for each dimension.
    let seeds: &[(u32, u32, &[u32])] = &[
        (1, 1, &[1]),                        // d=1
        (3, 2, &[1, 3]),                     // d=2
        (1, 3, &[1, 3, 5]),                  // d=3
        (5, 3, &[1, 1, 7]),                  // d=4
        (9, 4, &[1, 3, 7, 13]),              // d=5
        (9, 4, &[1, 1, 5, 11]),              // d=6
        (21, 5, &[1, 1, 7, 13, 23]),         // d=7
        (29, 5, &[1, 3, 3, 9, 17]),          // d=8
        (5, 5, &[1, 3, 5, 11, 25]),          // d=9
        (7, 5, &[1, 1, 1, 3, 15]),           // d=10
        (15, 5, &[1, 1, 5, 11, 25]),         // d=11
        (17, 5, &[1, 3, 1, 9, 21]),          // d=12
        (65, 6, &[1, 1, 5, 11, 25, 53]),     // d=13
        (21, 6, &[1, 3, 1, 7, 19, 37]),      // d=14
        (15, 6, &[1, 1, 3, 11, 25, 45]),     // d=15
        (735, 6, &[1, 1, 1, 3, 13, 29]),     // d=16
        (165, 6, &[1, 1, 5, 7, 21, 43]),     // d=17
        (15, 6, &[1, 3, 5, 11, 25, 45]),     // d=18
        (511, 7, &[1, 1, 1, 3, 13, 29, 61]), // d=19
    ];

    for d in 1..dim.min(seeds.len() + 1) {
        let (poly, m, init) = seeds[d - 1];
        let m = m as usize;
        let base = d * 64;

        // Set initial direction numbers.
        for j in 0..m {
            let v = (init[j] as u64) << (63 - j);
            dir[base + j] = v;
        }

        // Extend using the recurrence: v[j] = v[j-m] XOR (v[j-m] * 2^m)
        // with the primitive polynomial bits.
        for j in m..64 {
            let mut v = dir[base + j - m];
            v >>= m;
            // Apply polynomial: for each bit set in poly, XOR with shifted v.
            let mut p = poly;
            let mut k = 1u32;
            while p > 0 {
                if p & 1 != 0 {
                    let shifted = dir[base + j - k as usize];
                    v ^= shifted >> (m - k as usize);
                }
                p >>= 1;
                k += 1;
            }
            dir[base + j] = v;
        }
    }

    dir
}

// ── k-NN surrogate model ──────────────────────────────────────────────────────

/// A simple k-nearest-neighbor surrogate model for the objective function.
/// Predicts the objective at a new point as the distance-weighted average of
/// the k nearest observed points. This is a lightweight alternative to a full
/// Random Forest (SMAC3) or Gaussian Process, suitable for small budgets.
pub struct KnnSurrogate {
    /// Observed normalized inputs.
    points: Vec<Vec<f64>>,
    /// Observed objective values (higher is better).
    values: Vec<f64>,
    /// Number of neighbors to consider.
    k: usize,
}

impl KnnSurrogate {
    /// Creates the component.
    pub fn new(k: usize) -> Self {
        Self {
            points: Vec::new(),
            values: Vec::new(),
            k: k.max(1),
        }
    }

    /// Add an observation.
    pub fn add(&mut self, x: Vec<f64>, y: f64) {
        self.points.push(x);
        self.values.push(y);
    }

    /// Number of observations.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Predict the objective at a new point.
    /// Returns (mean, std) — the std is used for exploration in EI.
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 1.0);
        }

        // Compute distances to all observed points.
        let mut dists: Vec<(usize, f64)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d = euclidean(x, p);
                (i, if d < 1e-12 { 1e-12 } else { d })
            })
            .collect();

        // Sort by distance ascending.
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k.min(dists.len());
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        let mut vals = Vec::with_capacity(k);
        for &(i, d) in dists.iter().take(k) {
            let w = 1.0 / d; // inverse distance weighting
            weighted_sum += w * self.values[i];
            weight_total += w;
            vals.push(self.values[i]);
        }

        let mean = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        };

        // Standard deviation of the k nearest values (exploration signal).
        let std = if vals.len() > 1 {
            let avg = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / vals.len() as f64;
            var.sqrt()
        } else {
            0.5
        };

        (mean, std)
    }
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ── Expected Improvement acquisition ──────────────────────────────────────────

/// Expected Improvement (EI) acquisition function.
/// EI(x) = (mu - f_best) * Phi(Z) + sigma * phi(Z)
/// where Z = (mu - f_best) / sigma.
/// Higher EI = more promising to try.
pub fn expected_improvement(mu: f64, sigma: f64, f_best: f64) -> f64 {
    if sigma < 1e-12 {
        // No uncertainty: EI = max(0, mu - f_best).
        return (mu - f_best).max(0.0);
    }
    let z = (mu - f_best) / sigma;
    // Normal CDF and PDF approximations.
    let phi_z = normal_pdf(z);
    let cdf_z = normal_cdf(z);
    (mu - f_best) * cdf_z + sigma * phi_z
}

/// Standard normal PDF.
fn normal_pdf(x: f64) -> f64 {
    let c = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    c * (-0.5 * x * x).exp()
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn normal_cdf(x: f64) -> f64 {
    // Use the error function approximation.
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ── Search engine (ask-and-tell API) ──────────────────────────────────────────
