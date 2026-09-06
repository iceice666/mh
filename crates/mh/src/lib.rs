//! mh — a small, opinionated PTC-first agent harness.
//!
//! v0.5 layout: a task is durable; model calls, context windows, PTC
//! executions, workers, and subprocesses are disposable mechanisms.
//! - [`ptc`]: MicroQuickJS wrapper + PTC runtime (ABI, budget, cancellation)
//! - [`tools`]: capability model, result store, core tools
//! - [`session`]: append-only event store, durable task views, resume
//! - [`goal`]: durable goal state, explicit completion, context checkpoints
//! - [`context`]: context compiler with token budgeting and rollover
//! - [`process`]: durable background process handles
//! - [`runtime`]: durable service layer (workers, processes, task inspection)
//! - [`model`]: OpenAI-compatible adapter + native FC lowering
//! - [`agent`]: the agent loop, shared by root and worker executions

pub mod agent;
pub mod checkpoint;
pub mod context;
pub mod delegation;
pub mod goal;
pub mod identity;
pub mod isolation;
pub mod model;
pub mod process;
pub mod ptc;
pub mod runtime;
pub mod session;
pub mod tools;
pub mod workspace;
