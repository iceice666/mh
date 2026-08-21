//! mh — a small, opinionated PTC-first agent harness.
//!
//! Layout (spec §28, single-crate variant):
//! - [`ptc`]: MicroQuickJS wrapper + PTC runtime (ABI, budget,
//!   cancellation)
//! - [`tools`]: capability model, result store, core tools
//! - [`session`]: append-only event store, resume
//! - [`context`]: context compiler with token budgeting
//! - [`model`]: OpenAI-compatible adapter + native FC lowering
//! - [`agent`]: the agent loop

pub mod agent;
pub mod checkpoint;
pub mod context;
pub mod identity;
pub mod model;
pub mod ptc;
pub mod session;
pub mod tools;
pub mod workspace;
