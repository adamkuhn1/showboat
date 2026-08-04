//! MCTS rollout hot loop. This is the compute-heavy core the redirect moved to
//! Rust: evaluating a candidate shot means simulating it and (optionally) playing
//! several random continuation shots to estimate its value. Thousands of these
//! run per AI decision, so the whole loop is native.
//!
//! The rollout policy here is deliberately simple and *uninformed* (uniform over
//! a geometric candidate set) — that is the pure-search BASELINE. When the ONNX
//! value/policy net is wired in (later milestone), the JS-side MCTS supplies
//! priors and this rollout is replaced by the learned value; nothing here scripts
//! a shot, it only simulates physics.

use crate::ball::Ball;
use crate::cue::{apply_cue, CueAction};
use crate::engine::{simulate_shot, SimResult};
use crate::table::Table;

/// A tiny deterministic PRNG (xorshift64*) so rollouts are reproducible given a
/// seed — important for debuggable self-play and stable overlay data.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Simulate a single candidate shot on a copy of the balls, returning both the
/// resting state and the event trace. Does not mutate the input.
pub fn simulate_candidate(
    balls: &[Ball],
    table: &Table,
    action: &CueAction,
) -> (Vec<Ball>, SimResult) {
    let mut copy: Vec<Ball> = balls.to_vec();
    if let Some(cue) = copy.iter_mut().find(|b| b.id == 0) {
        apply_cue(cue, action);
    }
    let result = simulate_shot(&mut copy, table, false);
    (copy, result)
}

/// Number of the shooter's target balls pocketed in a sim result, given the
/// target ball ids. Used as a cheap rollout reward.
pub fn count_targets_pocketed(result: &SimResult, targets: &[u8]) -> u32 {
    result
        .pocketed
        .iter()
        .filter(|id| targets.contains(id))
        .count() as u32
}

/// Evaluate one candidate shot by simulating it and running `depth` random
/// continuation shots (uniform aim/power over the given bounds), returning the
/// total number of target balls pocketed across the sequence. This is the
/// baseline rollout value — real, physics-grounded, and unscripted.
///
/// `n_rollouts` independent playouts are averaged; this is the loop that runs
/// thousands of times per decision.
pub fn rollout_value(
    balls: &[Ball],
    table: &Table,
    action: &CueAction,
    targets: &[u8],
    depth: u32,
    n_rollouts: u32,
    seed: u64,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut total = 0.0_f64;

    for _ in 0..n_rollouts.max(1) {
        // Apply the candidate shot first.
        let (mut state, first) = simulate_candidate(balls, table, action);
        let mut reward = count_targets_pocketed(&first, targets) as f64;

        // Random continuation playout.
        for _ in 0..depth {
            // If the cue ball was pocketed, stop this playout (a scratch ends the
            // sequence for reward purposes).
            let cue_alive = state.iter().any(|b| b.id == 0 && !b.pocketed);
            if !cue_alive {
                break;
            }
            let any_target = state
                .iter()
                .any(|b| targets.contains(&b.id) && !b.pocketed);
            if !any_target {
                break;
            }
            let a = CueAction {
                phi: rng.range(0.0, std::f64::consts::TAU),
                power: rng.range(0.2, 1.0),
                side_spin: rng.range(-1.0, 1.0),
                top_spin: rng.range(-1.0, 1.0),
            };
            let (next, res) = simulate_candidate(&state, table, &a);
            reward += count_targets_pocketed(&res, targets) as f64;
            state = next;
        }
        total += reward;
    }

    total / n_rollouts.max(1) as f64
}
