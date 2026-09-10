#!/usr/bin/env python3
"""Real-terminal smoke test for the `mh` TUI.

Drives the actual binary through a PTY against a loopback Responses fixture,
so what is asserted is what a user's terminal would receive. Standard library
only: pty, termios, fcntl, select, socket, subprocess, threading.

    python3 crates/mh/tests/support/tui_smoke.py --binary target/debug/mh

The `--child` mode is an implementation detail: the parent re-executes this
file so the child can claim the slave as its controlling tty before exec'ing
the binary, which is what makes `IsTerminal` and cursor addressing real.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import pty
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time

DEADLINE = 10.0
COLS, ROWS = 120, 36

CSI_ALT_ENTER = "\x1b[?1049h"
CSI_ALT_LEAVE = "\x1b[?1049l"
CSI_PASTE_ON = "\x1b[?2004h"
CSI_PASTE_OFF = "\x1b[?2004l"
CSI_CURSOR_SHOW = "\x1b[?25h"

CTRL_C = "\x03"
CTRL_N = "\x0e"
CTRL_Q = "\x11"
CTRL_R = "\x12"
CTRL_X = "\x18"
ENTER = "\r"

PASSES: list[str] = []


class Failure(Exception):
    """A scenario assertion failed; diagnostics are attached by the caller."""


# ---------------------------------------------------------------------------
# Child mode: claim the controlling tty, then become the binary
# ---------------------------------------------------------------------------


def run_child(binary: str, argv: list[str]) -> None:
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    os.execve(binary, [binary, *argv], os.environ.copy())


# ---------------------------------------------------------------------------
# Provider fixture
# ---------------------------------------------------------------------------


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def text_delta(delta: str) -> bytes:
    return sse({"type": "response.output_text.delta", "delta": delta})


def reasoning_delta(delta: str) -> bytes:
    return sse({"type": "response.reasoning_summary_text.delta", "delta": delta})


def completed(response_id: str) -> bytes:
    return sse(
        {
            "type": "response.completed",
            "response": {"id": response_id, "output": []},
        }
    )


def ptc_call(source: str) -> list[bytes]:
    arguments = json.dumps({"source": source})
    return [
        sse(
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "name": "ptc", "arguments": ""},
            }
        ),
        sse(
            {
                "type": "response.function_call_arguments.done",
                "name": "ptc",
                "arguments": arguments,
            }
        ),
        sse({"type": "response.completed", "response": {"id": "resp_ptc", "output": []}}),
    ]


class Turn:
    """One provider response.

    `hold` keeps the connection open after the listed frames, emitting only
    SSE comments. Comments are not events, so the client learns nothing new,
    but the socket keeps producing lines — which is what lets the existing
    blocking reader in `model.rs` notice a supersede or an interrupt.
    """

    def __init__(self, frames: list[bytes], *, hold: bool = False, status: int = 200):
        self.frames = frames
        self.hold = hold
        self.status = status


class Provider:
    """Loopback Responses endpoint driven by a scripted list of turns."""

    def __init__(self, turns: list[Turn]):
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.port = self.listener.getsockname()[1]
        self.turns = list(turns)
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def bodies(self) -> list[dict]:
        with self.lock:
            return list(self.requests)

    def close(self) -> None:
        self.stop.set()
        self.release.set()
        try:
            self.listener.close()
        except OSError:
            pass
        self.thread.join(timeout=2)

    def _serve(self) -> None:
        while not self.stop.is_set():
            try:
                self.listener.settimeout(0.2)
                stream, _ = self.listener.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            threading.Thread(
                target=self._handle, args=(stream,), daemon=True
            ).start()

    def _handle(self, stream: socket.socket) -> None:
        with stream:
            body = self._read_request(stream)
            if body is None:
                return
            with self.lock:
                self.requests.append(body)
                turn = self.turns.pop(0) if self.turns else Turn([completed("resp_empty")])
            try:
                self._respond(stream, turn)
            except OSError:
                # A superseded or interrupted client drops the connection; that
                # is a pass condition, not a fixture failure.
                pass

    @staticmethod
    def _read_request(stream: socket.socket) -> dict | None:
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = stream.recv(4096)
            if not chunk:
                return None
            buffer += chunk
        head, _, rest = buffer.partition(b"\r\n\r\n")
        headers = head.decode("latin-1").splitlines()
        length = 0
        for line in headers:
            name, _, value = line.partition(":")
            if name.strip().lower() == "content-length":
                length = int(value.strip())
        while len(rest) < length:
            chunk = stream.recv(4096)
            if not chunk:
                break
            rest += chunk
        try:
            return json.loads(rest[:length] or b"{}")
        except json.JSONDecodeError:
            return {}

    def _respond(self, stream: socket.socket, turn: Turn) -> None:
        reason = "OK" if turn.status == 200 else "Internal Server Error"
        stream.sendall(
            f"HTTP/1.1 {turn.status} {reason}\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n\r\n".encode()
        )
        if turn.status != 200:
            stream.sendall(
                sse({"type": "error", "message": "provider exploded"})
            )
            return
        for frame in turn.frames:
            stream.sendall(frame)
        if not turn.hold:
            return
        self.release.clear()
        while not self.release.wait(0.05):
            if self.stop.is_set():
                return
            stream.sendall(b": heartbeat\n\n")


# ---------------------------------------------------------------------------
# PTY session
# ---------------------------------------------------------------------------


class Screen:
    """Minimal VT100 cell model.

    Ratatui renders diffs: a single visible line arrives as several fragments
    separated by cursor-position sequences. Substring checks against the raw
    byte stream therefore prove nothing about what the user sees, so the
    fragments are replayed into a grid and assertions run on that.
    """

    CSI = re.compile(r"\x1b\[([0-9;?]*)([ -/]*)([@-~])")
    OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
    SHORT = re.compile(r"\x1b(?:[()][A-Za-z0-9]|[@-Z\\-_])")

    def __init__(self, cols: int, rows: int):
        self.resize(cols, rows)

    def resize(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        self.cells = [[" "] * cols for _ in range(rows)]
        self.row = self.column = 0

    def clear(self) -> None:
        self.cells = [[" "] * self.cols for _ in range(self.rows)]
        self.row = self.column = 0

    def feed(self, data: str) -> None:
        index = 0
        while index < len(data):
            character = data[index]
            if character == "\x1b":
                index += self._escape(data, index)
                continue
            if character == "\r":
                self.column = 0
            elif character == "\n":
                self.row = min(self.row + 1, self.rows - 1)
            elif character == "\b":
                self.column = max(0, self.column - 1)
            elif character >= " ":
                self._put(character)
            index += 1

    def _escape(self, data: str, index: int) -> int:
        match = self.CSI.match(data, index)
        if match:
            self._csi(match.group(1), match.group(3))
            return match.end() - index
        for pattern in (self.OSC, self.SHORT):
            match = pattern.match(data, index)
            if match:
                return match.end() - index
        return 1

    def _csi(self, params: str, final: str) -> None:
        numbers = [int(value) for value in params.split(";") if value.isdigit()]
        first = numbers[0] if numbers else None
        if final == "H":
            self.row = (numbers[0] if len(numbers) > 0 else 1) - 1
            self.column = (numbers[1] if len(numbers) > 1 else 1) - 1
            self.row = max(0, min(self.row, self.rows - 1))
            self.column = max(0, min(self.column, self.cols - 1))
        elif final == "J":
            self._erase_display(first or 0)
        elif final == "K":
            self._erase_line(first or 0)
        elif final == "A":
            self.row = max(0, self.row - (first or 1))
        elif final == "B":
            self.row = min(self.rows - 1, self.row + (first or 1))
        elif final == "C":
            self.column = min(self.cols - 1, self.column + (first or 1))
        elif final == "D":
            self.column = max(0, self.column - (first or 1))
        elif final == "h" and params == "?1049":
            self.clear()

    def _erase_display(self, mode: int) -> None:
        if mode in (2, 3):
            self.clear()
        elif mode == 0:
            self._erase_line(0)
            for row in range(self.row + 1, self.rows):
                self.cells[row] = [" "] * self.cols

    def _erase_line(self, mode: int) -> None:
        if mode == 0:
            span = range(self.column, self.cols)
        elif mode == 1:
            span = range(0, self.column + 1)
        else:
            span = range(0, self.cols)
        for column in span:
            self.cells[self.row][column] = " "

    def _put(self, character: str) -> None:
        if self.column >= self.cols:
            self.column = 0
            self.row = min(self.row + 1, self.rows - 1)
        self.cells[self.row][self.column] = character
        # Wide glyphs occupy two cells; the placeholder keeps columns aligned.
        width = 2 if _wide(character) else 1
        for offset in range(1, width):
            if self.column + offset < self.cols:
                self.cells[self.row][self.column + offset] = ""
        self.column += width

    def text(self) -> str:
        return "\n".join(
            "".join(cell for cell in row).rstrip() for row in self.cells
        )


def _wide(character: str) -> bool:
    code = ord(character)
    return (
        0x1100 <= code <= 0x115F
        or 0x2E80 <= code <= 0xA4CF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE6F
        or 0xFF00 <= code <= 0xFF60
        or 0xFFE0 <= code <= 0xFFE6
        or 0x20000 <= code <= 0x3FFFD
    )


class Session:
    def __init__(self, binary: str, workspace: str, env: dict[str, str], argv=()):
        self.master, self.slave = pty.openpty()
        self.saved_attrs = termios.tcgetattr(self.slave)
        self.last_attrs = self.saved_attrs
        self.display = Screen(COLS, ROWS)
        self.resize(COLS, ROWS)
        self.buffer = ""
        self.child = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--child", binary, *argv],
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
            cwd=workspace,
            env=env,
            start_new_session=True,
            close_fds=True,
        )

    def resize(self, cols: int, rows: int) -> None:
        fcntl.ioctl(
            self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0)
        )
        self.display.resize(cols, rows)

    def sigwinch(self) -> None:
        try:
            os.killpg(os.getpgid(self.child.pid), 28)  # SIGWINCH
        except (ProcessLookupError, PermissionError):
            pass

    def send(self, data: str) -> None:
        os.write(self.master, data.encode())

    def pump(self, timeout: float = 0.2) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                readable, _, _ = select.select([self.master], [], [], remaining)
            except OSError:
                break
            if not readable:
                continue
            try:
                chunk = os.read(self.master, 65536)
            except OSError as error:
                if error.errno in (errno.EIO, errno.EBADF):
                    break
                raise
            if not chunk:
                break
            decoded = chunk.decode("utf-8", "replace")
            self.buffer += decoded
            self.display.feed(decoded)
        return self.buffer

    def screen(self, settle: float = 0.2) -> str:
        """Visible cells after letting pending output land."""
        self.pump(settle)
        return self.display.text()

    def wait_screen(self, needle: str, timeout: float = DEADLINE) -> str:
        """Waits for text to be visible on screen, not merely transmitted."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.05)
            visible_text = self.display.text()
            if needle in visible_text:
                return visible_text
        raise Failure(
            f"never saw {needle!r} on screen; last frame was:\n{self.display.text()}"
        )

    def wait_for(self, needle: str, timeout: float = DEADLINE) -> None:
        """Waits for raw bytes, for escape-sequence assertions only."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if needle in self.pump(0.1):
                return
            if self.child.poll() is not None and needle not in self.pump(0.1):
                break
        raise Failure(f"never saw {needle!r} in the byte stream")

    def wait_exit(self, timeout: float = DEADLINE) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.05)
            # Sample the line discipline while the pty is still attached: on
            # Darwin the slave's termios becomes unreadable once the session
            # leader is gone, so checking only after exit proves nothing.
            self._sample_attrs()
            code = self.child.poll()
            if code is not None:
                self._sample_attrs()
                return code
        raise Failure("the child never exited")

    def _sample_attrs(self) -> None:
        try:
            self.last_attrs = termios.tcgetattr(self.slave)
        except termios.error:
            pass

    def restored(self) -> bool:
        """Whether raw mode was handed back before the process detached."""
        self._sample_attrs()
        return self.last_attrs == self.saved_attrs

    def close(self) -> None:
        # The child is a session leader, so signal the whole group: a run that
        # is mid-request must not be left behind holding the runner lock.
        if self.child.poll() is None:
            for signal_number, wait in ((15, 3.0), (9, 2.0)):
                try:
                    os.killpg(os.getpgid(self.child.pid), signal_number)
                except (ProcessLookupError, PermissionError):
                    break
                try:
                    self.child.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        for handle in (self.master, self.slave):
            try:
                os.close(handle)
            except OSError:
                pass


def base_env(home: str, provider: Provider | None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OPENAI_API_KEY", "MH_API_KEY", "MH_BASE_URL", "MH_MODEL"}
    }
    env.update(
        {
            "HOME": home,
            "XDG_CONFIG_HOME": os.path.join(home, "config"),
            "TERM": "xterm-256color",
            "MH_MODEL": "test",
            "MH_MODEL_TIMEOUT_SECS": "5",
        }
    )
    if provider is not None:
        env["MH_API_KEY"] = "test"
        env["MH_BASE_URL"] = provider.base_url
    os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
    return env


def journal(workspace: str) -> list[dict]:
    path = os.path.join(workspace, ".mh", "session.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return records


def wait_journal(
    workspace: str,
    predicate,
    what: str,
    timeout: float = DEADLINE,
    session: "Session | None" = None,
):
    """Waits for durable state, keeping the pty drained while it waits.

    A blocked pty master stalls the very process being waited on, so a wait
    that ignores the terminal can time out on a run that was never stuck.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        records = journal(workspace)
        if predicate(records):
            return records
        if session is not None:
            session.pump(0.05)
        else:
            time.sleep(0.05)
    types = [record.get("type") for record in journal(workspace)]
    detail = f"journal never satisfied: {what}\njournal: {types}"
    if session is not None:
        detail += f"\nlast frame:\n{session.display.text()}"
    raise Failure(detail)


def events(records: list[dict], kind: str) -> list[dict]:
    return [record for record in records if record.get("type") == kind]


def report(name: str) -> None:
    PASSES.append(name)
    print(f"PASS {name}", flush=True)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_empty(binary: str) -> None:
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        session = Session(binary, workspace, base_env(home, None))
        try:
            screen = session.wait_screen("New task")
            if "No tasks yet" not in screen:
                raise Failure(f"empty sidebar missing:\n{screen}")
            raw = session.buffer
            if CSI_ALT_ENTER not in raw:
                raise Failure("alternate screen was never entered")
            if CSI_PASTE_ON not in raw:
                raise Failure("bracketed paste was never enabled")
            session.send(CTRL_Q)
            code = session.wait_exit()
            if code != 0:
                raise Failure(f"exit status {code}")
            raw = session.buffer
            for needle, label in (
                (CSI_ALT_LEAVE, "alternate screen leave"),
                (CSI_PASTE_OFF, "bracketed paste disable"),
                (CSI_CURSOR_SHOW, "cursor show"),
            ):
                if needle not in raw:
                    raise Failure(f"{label} was never emitted")
            if not session.restored():
                raise Failure("slave termios was not restored")
            if os.path.exists(os.path.join(workspace, ".mh")):
                raise Failure("browsing created .mh")
        finally:
            session.close()
        report("empty_workspace_is_read_only")


def scenario_stream_and_steer(binary: str) -> None:
    provider = Provider(
        [
            Turn(
                [reasoning_delta("Inspecting"), text_delta("partial")],
                hold=True,
            ),
            Turn([text_delta("waiting"), completed("resp_waiting")]),
            Turn(ptc_call('return finish({ summary: "done", force: true });')),
            Turn([text_delta("follow-up answer"), completed("resp_followup")]),
        ]
    )
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        session = Session(binary, workspace, base_env(home, provider))
        try:
            session.wait_screen("New task")
            # A pasted multi-line Chinese prompt must survive intact.
            session.send("\x1b[200~修復測試\n  第二行\x1b[201~")
            screen = session.wait_screen("修復測試", timeout=5)
            if "第二行" not in screen:
                raise Failure(f"paste lost a line:\n{screen}")
            session.send(ENTER)

            session.wait_screen("Inspecting")
            session.wait_screen("partial")
            records = journal(workspace)
            if events(records, "assistant_message"):
                raise Failure("provisional text was journalled as an assistant message")

            session.send("focus tests")
            session.send(ENTER)
            wait_journal(
                workspace,
                lambda records: len(events(records, "steering_applied")) == 1,
                "one steering applied",
                session=session,
            )
            provider.release.set()
            session.wait_screen("waiting-user")
            records = journal(workspace)
            queued = events(records, "steering_queued")
            applied = events(records, "steering_applied")
            if (len(queued), len(applied)) != (1, 1):
                raise Failure(f"steering was double-sent: {len(queued)}/{len(applied)}")
            bodies = provider.bodies()
            if len(bodies) < 2:
                raise Failure("the steering turn never reached the provider")
            rendered = json.dumps(bodies[1], ensure_ascii=False)
            if "focus tests" not in rendered:
                raise Failure("steering was not in the next request")
            if any(
                event["content"] == "partial"
                for event in events(records, "assistant_message")
            ):
                raise Failure("provisional text became a durable message")
            report("stream_then_local_steering_is_durable_once")

            # Follow-up on a waiting-user task resumes the same task.
            session.send("carry on")
            session.send(ENTER)
            wait_journal(
                workspace,
                lambda records: bool(events(records, "task_completed")),
                "task completed",
                session=session,
            )
            records = journal(workspace)
            started = events(records, "task_started")
            if len(started) != 1:
                raise Failure(f"the waiting-user follow-up started {len(started)} tasks")
            completions = events(records, "task_completed")
            if len(completions) != 1:
                raise Failure(f"{len(completions)} completion records")
            report("waiting_user_followup_completes_same_task")

            # Completion ends the task, not the conversation. A message in the
            # same composer starts a linked task and receives another answer.
            session.send("explain the result")
            session.send(ENTER)
            session.wait_screen("follow-up answer")
            records = journal(workspace)
            started = events(records, "task_started")
            if len(started) != 2:
                raise Failure(f"the completed follow-up started {len(started)} tasks")
            if started[1].get("previous_task") != 1:
                raise Failure(f"follow-up task was not linked: {started[1]}")
            rendered = json.dumps(provider.bodies()[-1], ensure_ascii=False)
            if "waiting" not in rendered or "explain the result" not in rendered:
                raise Failure("completed follow-up lost recent conversation context")
            report("completed_task_accepts_a_linked_followup")

            session.send(CTRL_Q)
            if session.wait_exit() != 0:
                raise Failure("exit after completion follow-up failed")
        finally:
            session.close()
            provider.close()


def scenario_interrupt_and_cancel(binary: str) -> None:
    provider = Provider(
        [
            Turn([text_delta("one")], hold=True),
            Turn([text_delta("two")], hold=True),
            Turn([text_delta("three")], hold=True),
        ]
    )
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        session = Session(binary, workspace, base_env(home, provider))
        try:
            session.wait_screen("New task")
            session.send("first task")
            session.send(ENTER)
            session.wait_screen("one")

            # Ctrl-C interrupts the run and keeps the UI usable. The runtime
            # reports a mid-generation interrupt as a cancelled run rather
            # than a durable TaskInterrupted, so the observable fact is that
            # the runner released ownership and the UI survived.
            session.send(CTRL_C)
            wait_journal(
                workspace,
                lambda records: bool(events(records, "runner_released")),
                "the interrupted runner released ownership",
                session=session,
            )
            if session.child.poll() is not None:
                raise Failure("Ctrl-C closed the UI")
            session.wait_screen("run cancelled")

            # A second task proves the cancellation flag was reset.
            session.send(CTRL_N)
            session.send("second task")
            session.send(ENTER)
            session.wait_screen("second task")
            wait_journal(
                workspace,
                lambda records: len(events(records, "model_started")) == 2,
                "the second task reached the model",
                session=session,
            )
            records = journal(workspace)
            if len(events(records, "task_started")) != 2:
                raise Failure("the second task never started")
            report("interrupt_preserves_ui_and_resets_the_flag")

            # Ctrl-X is the durable path: a receipt, then the stop.
            session.send(CTRL_X)
            wait_journal(
                workspace,
                lambda records: bool(events(records, "task_cancel_requested")),
                "durable cancel receipt",
                session=session,
            )
            session.wait_screen("Cancellation queued")
            records = journal(workspace)
            requested = events(records, "task_cancel_requested")
            if len(requested) != 1:
                raise Failure(f"{len(requested)} cancel commands")
            if requested[0]["task_id"] != 2:
                raise Failure("cancellation hit the wrong task")
            report("durable_cancellation_reports_a_receipt")

            provider.release.set()
            session.send(CTRL_Q)
            if session.wait_exit(timeout=20) != 0:
                raise Failure("quit after cancellation hung")
            if not session.restored():
                raise Failure("termios was not restored after cancellation")
            report("quit_joins_the_local_worker")
        finally:
            session.close()
            provider.close()


def scenario_foreign_task(binary: str) -> None:
    provider = Provider([Turn([text_delta("detached")], hold=True)])
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        env = base_env(home, provider)
        detached = subprocess.run(
            [binary, "run", "outside work", "--detach"],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if detached.returncode != 0:
            raise Failure(f"detach failed: {detached.stderr}")
        session = Session(binary, workspace, env)
        try:
            # The sidebar shows a task this process did not create.
            screen = session.wait_screen("outside work")
            if "#1" not in screen:
                raise Failure(f"foreign task id not shown:\n{screen}")
            if "journal view" not in screen:
                raise Failure(f"foreign run not marked journal-only:\n{screen}")
            before = len(events(journal(workspace), "model_started"))
            time.sleep(0.6)
            after = len(events(journal(workspace), "model_started"))
            if after != before:
                raise Failure("browsing auto-resumed a foreign task")
            report("foreign_task_appears_without_resuming")

            session.send(CTRL_X)
            wait_journal(
                workspace,
                lambda records: bool(events(records, "task_cancel_requested")),
                "cancel of the foreign task",
                session=session,
            )
            requested = events(journal(workspace), "task_cancel_requested")
            if requested[0]["task_id"] != 1:
                raise Failure("cancelled the wrong task id")
            report("foreign_task_cancellation_targets_the_selection")

            provider.release.set()
            session.send(CTRL_Q)
            if session.wait_exit(timeout=20) != 0:
                raise Failure("quit with a foreign runner hung")
        finally:
            session.close()
            provider.close()
            # Never leave the detached runner behind.
            subprocess.run(
                [binary, "cancel", "1"],
                cwd=workspace,
                env=env,
                capture_output=True,
                timeout=15,
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                records = journal(workspace)
                if any(
                    event.get("status") in {"cancelled", "failed", "completed"}
                    for event in events(records, "task_status_changed")
                ) or events(records, "runner_released"):
                    break
                time.sleep(0.1)


def scenario_prelude(binary: str) -> None:
    for answer, expect_loaded in ((ENTER, False), ("y", True)):
        provider = Provider(
            [Turn(ptc_call('return finish({ summary: "ok", force: true });'))]
        )
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
            os.makedirs(os.path.join(workspace, ".mh"), exist_ok=True)
            with open(
                os.path.join(workspace, ".mh", "prelude.js"), "w", encoding="utf-8"
            ) as handle:
                handle.write("//! helper(): smoke tool\nfunction helper() { return 1; }\n")
            session = Session(binary, workspace, base_env(home, provider))
            try:
                session.wait_screen("New task")
                session.send("needs a prelude")
                session.send(ENTER)
                screen = session.wait_screen("Untrusted workspace prelude")
                if "mh>" in screen or "[y/N]" in screen:
                    raise Failure(f"readline prompt leaked:\n{screen}")
                if "sha256:" not in screen:
                    raise Failure(f"identity hash not shown:\n{screen}")
                session.send(answer)
                wait_journal(
                    workspace,
                    lambda records: bool(events(records, "task_completed")),
                    "task completed after the decision",
                    session=session,
                )
                loaded = bool(events(journal(workspace), "prelude_loaded"))
                if loaded != expect_loaded:
                    raise Failure(
                        f"answer {answer!r}: prelude_loaded={loaded}, expected {expect_loaded}"
                    )
                session.send(CTRL_Q)
                if session.wait_exit() != 0:
                    raise Failure("quit after the prelude decision failed")
            finally:
                session.close()
                provider.close()
    report("prelude_modal_decides_without_stdin")

    # Quitting while the modal is up must not hang on an unanswered callback.
    provider = Provider([Turn([text_delta("unused")], hold=True)])
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        os.makedirs(os.path.join(workspace, ".mh"), exist_ok=True)
        with open(
            os.path.join(workspace, ".mh", "prelude.js"), "w", encoding="utf-8"
        ) as handle:
            handle.write("//! helper(): smoke tool\nfunction helper() { return 2; }\n")
        session = Session(binary, workspace, base_env(home, provider))
        try:
            session.wait_screen("New task")
            session.send("prelude then quit")
            session.send(ENTER)
            session.wait_screen("Untrusted workspace prelude")
            session.send(CTRL_Q)
            if session.wait_exit(timeout=20) != 0:
                raise Failure("quitting at the modal hung")
        finally:
            session.close()
            provider.close()
    report("quit_at_the_prelude_modal_exits")


def scenario_resize_and_errors(binary: str) -> None:
    provider = Provider([Turn([], status=500)])
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        session = Session(binary, workspace, base_env(home, provider))
        try:
            session.wait_screen("Conversation")
            session.send("keep this draft")
            session.wait_screen("keep this draft")

            session.resize(80, 24)
            session.sigwinch()
            screen = session.wait_screen("Conversation")
            if "Tasks" in screen:
                raise Failure(f"sidebar shown below 90 cols:\n{screen}")

            session.resize(39, 11)
            session.sigwinch()
            session.wait_screen("Terminal too small")

            session.resize(COLS, ROWS)
            session.sigwinch()
            screen = session.wait_screen("Conversation")
            if "keep this draft" not in screen:
                raise Failure(f"resize discarded the draft:\n{screen}")
            if "Tasks" not in screen:
                raise Failure(f"sidebar missing after restore:\n{screen}")
            report("resize_preserves_the_draft_and_controls")

            # A provider error stays on screen and still exits cleanly. A 500
            # is rejected by the HTTP layer before the SSE body is parsed, so
            # the transport message is what the UI has to show.
            session.send(ENTER)
            session.wait_screen("model transport error")
            if session.child.poll() is not None:
                raise Failure("a provider error closed the UI")
            session.send(CTRL_Q)
            if session.wait_exit() != 0:
                raise Failure("quit after a provider error failed")
            if not session.restored():
                raise Failure("termios was not restored after an error")
            report("provider_error_stays_visible")
        finally:
            session.close()
            provider.close()


def scenario_non_tty(binary: str) -> None:
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        env = base_env(home, None)
        piped = subprocess.run(
            [binary],
            cwd=workspace,
            env=env,
            input="/quit\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        if piped.returncode != 0:
            raise Failure(f"piped stdin failed: {piped.stderr}")
        blended = piped.stdout + piped.stderr
        if CSI_ALT_ENTER in blended:
            raise Failure("a pipe entered the alternate screen")
        if "mh>" not in blended:
            raise Failure(f"the text REPL did not run:\n{blended}")

        for argv in (["--help"], ["--version"]):
            result = subprocess.run(
                [binary, *argv],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise Failure(f"mh {argv[0]} failed")
            if CSI_ALT_ENTER in result.stdout + result.stderr:
                raise Failure(f"mh {argv[0]} entered the alternate screen")
        report("non_tty_invocations_stay_textual")

    # A real PTY but TERM=dumb, and a PTY with stdout redirected: both textual.
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        env = base_env(home, None)
        env["TERM"] = "dumb"
        session = Session(binary, workspace, env)
        try:
            session.wait_for("mh>")
            if CSI_ALT_ENTER in session.buffer:
                raise Failure("TERM=dumb entered the alternate screen")
            session.send("/quit\n")
            if session.wait_exit() != 0:
                raise Failure("TERM=dumb exit failed")
        finally:
            session.close()

    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workspace:
        env = base_env(home, None)
        master, slave = pty.openpty()
        with tempfile.NamedTemporaryFile("w+", suffix=".log") as sink:
            child = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--child", binary],
                stdin=slave,
                stdout=sink,
                stderr=slave,
                cwd=workspace,
                env=env,
                start_new_session=True,
            )
            try:
                os.write(master, b"/quit\n")
                # The master must keep being drained: a session leader exiting
                # with an undrained controlling tty blocks in the kernel, which
                # would look like a hung binary rather than a full pipe.
                code = None
                deadline = time.monotonic() + DEADLINE
                while time.monotonic() < deadline:
                    readable, _, _ = select.select([master], [], [], 0.05)
                    if readable:
                        try:
                            os.read(master, 65536)
                        except OSError:
                            pass
                    code = child.poll()
                    if code is not None:
                        break
                if code is None:
                    raise Failure("the redirected-stdout run never exited")
                if code != 0:
                    raise Failure(f"redirected stdout exit status {code}")
                sink.seek(0)
                captured = sink.read()
                if CSI_ALT_ENTER in captured:
                    raise Failure("a redirected stdout entered the alternate screen")
                if "mh>" not in captured:
                    raise Failure(f"the text REPL did not run:\n{captured}")
            finally:
                if child.poll() is None:
                    os.killpg(os.getpgid(child.pid), 9)
                    child.wait(timeout=5)
                os.close(master)
                os.close(slave)
    report("dumb_term_and_redirects_stay_textual")


SCENARIOS = [
    scenario_empty,
    scenario_stream_and_steer,
    scenario_interrupt_and_cancel,
    scenario_foreign_task,
    scenario_prelude,
    scenario_resize_and_errors,
    scenario_non_tty,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", help="path to the mh binary under test")
    parser.add_argument("--child", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    parser.add_argument("--only", help="run one scenario by name substring")
    arguments = parser.parse_args()

    if arguments.child:
        run_child(arguments.child[0], arguments.child[1:])
        return 0

    if not arguments.binary:
        parser.error("--binary is required")
    binary = os.path.abspath(arguments.binary)
    if not os.access(binary, os.X_OK):
        print(f"FAIL not executable: {binary}", file=sys.stderr)
        return 2
    if shutil.which("git") is None:
        print("FAIL git is required", file=sys.stderr)
        return 2

    failures = 0
    for scenario in SCENARIOS:
        name = scenario.__name__.removeprefix("scenario_")
        if arguments.only and arguments.only not in name:
            continue
        try:
            scenario(binary)
        except Failure as failure:
            failures += 1
            print(f"FAIL {name}: {failure}", file=sys.stderr, flush=True)
        except Exception as error:  # noqa: BLE001 - report and keep going
            failures += 1
            print(f"FAIL {name}: unexpected {error!r}", file=sys.stderr, flush=True)
    print(f"\n{len(PASSES)} passed, {failures} scenario(s) failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
