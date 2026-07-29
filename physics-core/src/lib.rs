//! Showboat physics core (Rust → WASM).
//!
//! Event-based, Han-2005-lineage 2D pool physics plus the MCTS rollout hot loop.
//! Lives in Rust because an AI decision runs thousands of full-shot rollouts, and
//! that inner loop honestly wants a compiled language; the equations mirror the
//! Python training simulator (pooltool lineage) so train-time and play-time
//! physics stay in parity.
//!
//! The JS/TS side (canvas render, rules UI, ONNX inference, overlay) calls into
//! this module through the thin `wasm` boundary below. Native `cargo test`
//! exercises the same code without the wasm toolchain.

pub mod ball;
pub mod collisions;
pub mod constants;
pub mod cue;
pub mod engine;
pub mod motion;
pub mod predict;
pub mod rollout;
pub mod table;
pub mod vec;

pub mod wasm;

#[cfg(test)]
mod tests;
