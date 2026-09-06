//! PTC runtime: MicroQuickJS wrapper, ABI globals, budget, cancellation.

pub mod prelude;
pub mod runtime;
pub mod trust;
pub mod wrapper;

pub use prelude::{Prelude, PreludeError};
pub use runtime::{
    HostCallOutcome, PtcBudget, PtcDiagnostic, PtcDiagnosticKind, PtcEvent, PtcEventSink,
    PtcEventSinkError, PtcExecution, PtcOutcome, PtcResult, PtcRuntime,
};
pub use trust::{Trust, TrustDecision, TrustError, TrustStore};
