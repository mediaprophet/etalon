use crate::config_space::Configuration;
/// A multi-armed bandit using the Track-and-Stop algorithm for best-arm identification.
/// Given a small set of pre-selected configurations, it adaptively allocates samples
/// to identify the best one with statistical rigor.
pub struct TrackAndStopBandit {
    /// Arm configurations (pre-selected).
    arms: Vec<Configuration>,
    /// Observed rewards per arm.
    rewards: Vec<Vec<f64>>,
    /// Total samples per arm.
    counts: Vec<usize>,
    /// Whether the bandit has declared a winner.
    stopped: bool,
    /// The winning arm index (if stopped).
    winner: Option<usize>,
    /// Number of samples per arm in the initial round.
    initial_samples: usize,
    /// GLR threshold for stopping (Chernoff rule).
    glr_threshold: f64,
}

impl TrackAndStopBandit {
    /// Creates the component.
    pub fn new(arms: Vec<Configuration>) -> Self {
        let n = arms.len();
        Self {
            arms,
            rewards: vec![Vec::new(); n],
            counts: vec![0; n],
            stopped: false,
            winner: None,
            initial_samples: 3,
            glr_threshold: 10.0, // Conservative threshold.
        }
    }

    /// Number of arms.
    pub fn n_arms(&self) -> usize {
        self.arms.len()
    }

    /// Whether the bandit has stopped (declared a winner).
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// The winning arm index (if stopped).
    pub fn winner(&self) -> Option<usize> {
        self.winner
    }

    /// Get the configuration for an arm.
    pub fn arm_config(&self, i: usize) -> Option<&Configuration> {
        self.arms.get(i)
    }

    /// Which arm to sample next. Returns `None` if stopped.
    pub fn ask(&self) -> Option<usize> {
        if self.stopped {
            return None;
        }
        let n = self.arms.len();
        if n == 0 {
            return None;
        }

        // Phase 1: ensure each arm has at least `initial_samples` samples.
        for i in 0..n {
            if self.counts[i] < self.initial_samples {
                return Some(i);
            }
        }

        // Phase 2: Track-and-Stop — sample the arm with the highest GLR statistic.
        // The GLR (Generalized Likelihood Ratio) test identifies the arm most likely
        // to be the best. We sample the current best arm more often (tracking).
        let means: Vec<f64> = (0..n)
            .map(|i| {
                let r = &self.rewards[i];
                if r.is_empty() {
                    0.0
                } else {
                    r.iter().sum::<f64>() / r.len() as f64
                }
            })
            .collect();

        // Find the current best arm.
        let best = means
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Check stopping condition: GLR test.
        let total_samples: usize = self.counts.iter().sum();
        if total_samples > n * self.initial_samples * 2 {
            let glr = self.compute_glr(&means, best);
            if glr > self.glr_threshold {
                return None; // Signal to stop — check_winner will be called.
            }
        }

        // Tracking: sample the best arm with probability proportional to its lead.
        // Simple D-tracking: sample the best arm 50% of the time, round-robin the rest.
        let round = total_samples % n;
        if round == best % n {
            Some(best)
        } else {
            Some(round)
        }
    }

    /// Report a reward for an arm.
    pub fn tell(&mut self, arm: usize, reward: f64) {
        if !reward.is_finite() {
            return;
        }
        if arm >= self.arms.len() {
            return;
        }
        self.rewards[arm].push(reward);
        self.counts[arm] += 1;

        // Check if we should stop.
        if self.counts.iter().all(|&c| c >= self.initial_samples) {
            let means: Vec<f64> = (0..self.arms.len())
                .map(|i| {
                    let r = &self.rewards[i];
                    if r.is_empty() {
                        0.0
                    } else {
                        r.iter().sum::<f64>() / r.len() as f64
                    }
                })
                .collect();
            let best = means
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let glr = self.compute_glr(&means, best);
            if glr > self.glr_threshold {
                self.stopped = true;
                self.winner = Some(best);
            }
        }
    }

    /// Compute the GLR (Generalized Likelihood Ratio) statistic.
    /// Higher GLR = more confidence that the best arm is truly the best.
    fn compute_glr(&self, means: &[f64], best: usize) -> f64 {
        let n = means.len();
        if n <= 1 {
            return f64::INFINITY;
        }
        let best_mean = means[best];
        let mut min_gap = f64::INFINITY;
        for i in 0..n {
            if i == best {
                continue;
            }
            let gap = best_mean - means[i];
            if gap < min_gap {
                min_gap = gap;
            }
        }
        if min_gap <= 0.0 {
            return 0.0; // No clear winner.
        }
        // GLR ≈ sum_i n_i * (best_mean - mean_i)^2 / (2 * sigma^2)
        // With sigma^2 estimated from the rewards.
        let total_var: f64 = (0..n)
            .map(|i| {
                let r = &self.rewards[i];
                if r.len() < 2 {
                    0.01
                } else {
                    let m = means[i];
                    let v = r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / r.len() as f64;
                    v
                }
            })
            .sum::<f64>()
            / n as f64;
        let sigma2 = total_var.max(0.01);

        let mut glr = 0.0;
        for i in 0..n {
            if i == best {
                continue;
            }
            let gap = best_mean - means[i];
            glr += self.counts[i] as f64 * gap * gap / (2.0 * sigma2);
        }
        glr
    }

    /// Get the mean reward for each arm.
    pub fn arm_means(&self) -> Vec<f64> {
        (0..self.arms.len())
            .map(|i| {
                let r = &self.rewards[i];
                if r.is_empty() {
                    0.0
                } else {
                    r.iter().sum::<f64>() / r.len() as f64
                }
            })
            .collect()
    }
}
