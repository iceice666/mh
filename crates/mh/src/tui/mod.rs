//! Conversational full-screen front end for the durable runtime.
//!
//! Composition only: [`app`] reduces state, [`render`] paints, [`runtime`]
//! talks to the session and the agent, [`terminal`] owns raw mode. The loop
//! below is the only place they meet, and it is the only place that reads the
//! keyboard.

mod app;
mod render;
mod runtime;
mod terminal;

#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Duration;

use crossterm::event::{self, Event};

#[cfg(test)]
use app::Action;
use app::{App, Level, Plan, Target};
use runtime::{Command, Observer, RunRequest, UiEvent, Worker};

use crate::CliError;

/// Input poll interval. Short enough that typing feels immediate, long enough
/// that an idle session costs nothing.
const TICK: Duration = Duration::from_millis(25);
/// Events processed before yielding back to the keyboard and the screen, so a
/// fast token stream cannot starve input.
const EVENT_BUDGET: usize = 256;

pub(super) fn run(workspace: &Path, cancelled: Arc<AtomicBool>) -> Result<(), CliError> {
    let (mut screen, guard) = terminal::enter().map_err(CliError::Io)?;
    let fatal = Arc::new(AtomicBool::new(false));
    let hook = install_panic_hook(&fatal, &cancelled);

    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let observer = runtime::spawn_observer(workspace.to_path_buf(), event_tx.clone())?;
    let mut loop_state = Loop {
        app: App::new(workspace),
        worker: None,
        command: None,
        observer: Some(observer),
        events: event_tx,
        cancelled,
        fatal,
    };
    let result = loop_state.pump(workspace, &mut screen, &event_rx);
    loop_state.shutdown();
    drop(guard);
    std::panic::set_hook(hook);
    result
}

/// Installs a hook that marks the process fatal, cancels the run, and restores
/// the terminal before the original hook prints. Returns the original hook so
/// it can be reinstated on a clean exit.
fn install_panic_hook(
    fatal: &Arc<AtomicBool>,
    cancelled: &Arc<AtomicBool>,
) -> Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync + 'static> {
    let previous: Arc<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync + 'static> =
        Arc::from(std::panic::take_hook());
    let hook_fatal = fatal.clone();
    let hook_cancelled = cancelled.clone();
    let chained = previous.clone();
    std::panic::set_hook(Box::new(move |info| {
        hook_fatal.store(true, Ordering::Relaxed);
        hook_cancelled.store(true, Ordering::Relaxed);
        terminal::restore_best_effort();
        chained(info);
    }));
    Box::new(move |info| previous(info))
}

struct Loop {
    app: App,
    worker: Option<Worker>,
    command: Option<Command>,
    observer: Option<Observer>,
    events: Sender<UiEvent>,
    cancelled: Arc<AtomicBool>,
    fatal: Arc<AtomicBool>,
}

impl Loop {
    fn pump(
        &mut self,
        workspace: &Path,
        screen: &mut terminal::Screen,
        events: &Receiver<UiEvent>,
    ) -> Result<(), CliError> {
        loop {
            if self.fatal.load(Ordering::Relaxed) && !self.app.is_fatal() {
                self.app.set_fatal();
            }
            let drained = self.drain(events);
            self.reap_worker();
            self.reap_command();
            if let Some(plan) = self.app.take_followup() {
                self.execute(workspace, plan)?;
            }
            if let Some(task_id) = self.app.take_cancel_interrupt()
                && self.app.local_task() == Some(task_id)
            {
                self.cancelled.store(true, Ordering::Relaxed);
                if let Some(worker) = self.worker.as_ref() {
                    worker.interrupt();
                }
            }
            if self.app.take_session_change() {
                // The journal we were streaming into no longer exists: stop
                // feeding the old run and keep the error visible.
                if let Some(worker) = self.worker.as_ref() {
                    self.cancelled.store(true, Ordering::Relaxed);
                    worker.interrupt();
                }
                self.app.notice(
                    self.app.target(),
                    Level::Warn,
                    "Session changed; review task selection",
                );
            }
            if self.cancelled.load(Ordering::Relaxed) && self.worker.is_none() {
                // A real SIGINT with nothing running: consume it rather than
                // closing a UI the user did not ask to close.
                self.cancelled.store(false, Ordering::Relaxed);
            }

            if !self.app.is_fatal() && self.app.take_dirty() {
                screen
                    .draw(|frame| render::draw(frame, &mut self.app))
                    .map_err(CliError::Io)?;
            }

            if self.app.is_quitting() && self.worker.is_none() && self.command.is_none() {
                return Ok(());
            }
            if drained >= EVENT_BUDGET {
                continue;
            }

            match event::poll(TICK) {
                Ok(true) => match event::read() {
                    Ok(input) => self.input(workspace, input)?,
                    Err(error) => return self.abort(CliError::Io(error)),
                },
                Ok(false) => {}
                Err(error) => return self.abort(CliError::Io(error)),
            }
        }
    }

    fn input(&mut self, workspace: &Path, input: Event) -> Result<(), CliError> {
        match input {
            Event::Key(key) => {
                if let Some(action) = self.app.on_key(key) {
                    let plan = self.app.plan(action);
                    self.execute(workspace, plan)?;
                }
            }
            Event::Paste(text) => self.app.on_paste(&text),
            Event::Resize(_, _) => self.app.on_resize(),
            Event::FocusGained | Event::FocusLost | Event::Mouse(_) => {}
        }
        Ok(())
    }

    fn drain(&mut self, events: &Receiver<UiEvent>) -> usize {
        let mut handled = 0;
        let mut refresh = false;
        while handled < EVENT_BUDGET {
            match events.try_recv() {
                Ok(event) => {
                    handled += 1;
                    refresh |= self.handle(event);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        if refresh && let Some(observer) = self.observer.as_ref() {
            observer.refresh();
        }
        handled
    }

    /// Folds one runtime event into the app. Returns whether the durable
    /// projection should be refreshed.
    fn handle(&mut self, event: UiEvent) -> bool {
        match event {
            UiEvent::Snapshot(Ok(snapshot)) => {
                self.app.apply_snapshot(snapshot);
                false
            }
            UiEvent::Snapshot(Err(error)) => {
                self.app.apply_snapshot_error(&error);
                false
            }
            UiEvent::Registered { run_id, task_id } => {
                self.app.apply_registered(run_id, task_id);
                true
            }
            UiEvent::Admitted {
                run_id,
                task_id,
                after_seq,
            } => {
                self.app.apply_admitted(run_id, task_id, after_seq);
                true
            }
            UiEvent::Agent {
                run_id,
                task_id,
                event,
            } => self.app.apply_agent(run_id, task_id, event),
            UiEvent::Trust {
                run_id,
                prelude,
                reply,
            } => {
                self.app.apply_trust(run_id, prelude, reply);
                false
            }
            UiEvent::Receipt {
                target,
                cancel,
                result,
            } => {
                self.app.apply_receipt(target, cancel, result);
                true
            }
        }
    }

    fn execute(&mut self, workspace: &Path, plan: Plan) -> Result<(), CliError> {
        match plan {
            Plan::Nothing => {}
            Plan::StartRun { run_id, request } => {
                let target = match &request {
                    RunRequest::New(_) => Target::New,
                    RunRequest::Resume(task_id) => Target::Task(*task_id),
                };
                match runtime::spawn_worker(
                    workspace.to_path_buf(),
                    run_id,
                    request,
                    self.cancelled.clone(),
                    self.events.clone(),
                ) {
                    Ok(worker) => self.worker = Some(worker),
                    Err(error) => {
                        self.app.drop_local();
                        self.app.submit_failed(target, &error.to_string());
                    }
                }
            }
            Plan::LocalSteer { task_id, text } => {
                let target = Target::Task(task_id);
                match self.worker.as_ref() {
                    Some(worker) if worker.steer(text.clone()) => {
                        self.app.local_steer_sent(text);
                        self.app.submit_accepted(target);
                    }
                    _ => self
                        .app
                        .submit_failed(target, "agent control channel closed"),
                }
            }
            Plan::DurableSteer { task_id, text } => {
                match runtime::spawn_steer(
                    workspace.to_path_buf(),
                    task_id,
                    text,
                    self.events.clone(),
                ) {
                    Ok(command) => self.command = Some(command),
                    Err(error) => {
                        self.app
                            .apply_receipt(task_id, false, Err(error.to_string()));
                    }
                }
            }
            Plan::DurableCancel { task_id } => {
                match runtime::spawn_cancel(workspace.to_path_buf(), task_id, self.events.clone()) {
                    Ok(command) => self.command = Some(command),
                    Err(error) => {
                        self.app
                            .apply_receipt(task_id, true, Err(error.to_string()));
                    }
                }
            }
            Plan::Interrupt => {
                self.cancelled.store(true, Ordering::Relaxed);
                if let Some(worker) = self.worker.as_ref() {
                    worker.interrupt();
                }
            }
            Plan::Quit => {
                if let Some(worker) = self.worker.as_ref() {
                    self.cancelled.store(true, Ordering::Relaxed);
                    worker.interrupt();
                }
                if let Some(observer) = self.observer.take() {
                    observer.stop();
                }
            }
        }
        Ok(())
    }

    /// Joins the local worker only once it has actually finished, so the loop
    /// never blocks on a model call.
    fn reap_worker(&mut self) {
        if !self.worker.as_ref().is_some_and(Worker::is_finished) {
            return;
        }
        let worker = self.worker.take().expect("worker was present");
        let run_id = worker.run_id;
        let outcome = worker.join();
        self.app.worker_finished(run_id, outcome);
        if let Some(observer) = self.observer.as_ref() {
            observer.refresh();
        }
    }

    fn reap_command(&mut self) {
        if self.command.as_ref().is_some_and(Command::is_finished) {
            self.command.take().expect("command was present").join();
        }
    }

    /// I/O failure: stop the run, then unwind normally so the terminal and
    /// every thread this loop created are still cleaned up.
    fn abort(&mut self, error: CliError) -> Result<(), CliError> {
        self.app.set_fatal();
        self.cancelled.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.as_ref() {
            worker.interrupt();
        }
        Err(error)
    }

    fn shutdown(&mut self) {
        if let Some(observer) = self.observer.take() {
            observer.stop();
        }
        if let Some(worker) = self.worker.take() {
            self.cancelled.store(true, Ordering::Relaxed);
            worker.interrupt();
            let run_id = worker.run_id;
            let outcome = worker.join();
            self.app.worker_finished(run_id, outcome);
        }
        if let Some(command) = self.command.take() {
            command.join();
        }
    }
}

/// Also used by the tests, which drive the same reducer without a terminal.
#[cfg(test)]
impl Loop {
    fn headless(workspace: &Path) -> (Self, Receiver<UiEvent>) {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        (
            Self {
                app: App::new(workspace),
                worker: None,
                command: None,
                observer: None,
                events: event_tx,
                cancelled: Arc::new(AtomicBool::new(false)),
                fatal: Arc::new(AtomicBool::new(false)),
            },
            event_rx,
        )
    }

    fn key(&mut self, workspace: &Path, key: crossterm::event::KeyEvent) {
        if let Some(action) = self.app.on_key(key) {
            let plan = self.app.plan(action);
            let _ = self.execute(workspace, plan);
        }
    }

    fn action(&mut self, workspace: &Path, action: Action) {
        let plan = self.app.plan(action);
        let _ = self.execute(workspace, plan);
    }
}
