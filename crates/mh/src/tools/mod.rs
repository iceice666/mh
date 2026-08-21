//! Tool layer: capabilities, result store, core tool set.

pub mod capability;
pub mod fs_tools;
pub mod store;

pub use capability::{Capabilities, ProcessPolicy, ToolEffect};
pub use store::{ResultId, ResultMetadata, ResultStore, StoredResult};
