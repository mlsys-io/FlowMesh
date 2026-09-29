"""How an SSH run reacts to its own cancel and graceful-stop signals."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from shared.schemas.result import SSHResult
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_live_worker_config
from worker.executors import ssh_executor as ssh_executor_module
from worker.executors.base_executor import (
    ExecutionError,
    RunControl,
    TaskCancelledError,
)
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.ssh_session import SessionRequest, SSHConfig, SSHSession
from worker.executors.ssh_session.backends import docker as docker_backend_module


@pytest.fixture(autouse=True)
def _docker_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_backend_module, "docker_available", lambda: True)


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    from_spec = SSHConfig.from_spec

    def fast(*args: Any, **kwargs: Any) -> SSHConfig:
        cfg = from_spec(*args, **kwargs)
        cfg.poll_interval_sec = 0.01
        return cfg

    monkeypatch.setattr(ssh_executor_module.SSHConfig, "from_spec", fast)


def _task_message() -> WorkerTaskMessage:
    return WorkerTaskMessage.model_validate(
        {
            "task_id": "tsk-ssh",
            "workflow_id": "wfl-1",
            "owner_id": "owner",
            "assigned_worker": "worker-1",
            "dispatched_at": "2026-03-22T00:00:00Z",
            "task": {
                "apiVersion": "flowmesh/v1",
                "kind": "Task",
                "metadata": {"name": "wf:shell"},
                "spec": {
                    "taskType": "ssh",
                    "accessMode": "proxy",
                    "authorizedKeys": ["ssh-ed25519 AAAA..."],
                },
            },
        }
    )


class _Session(SSHSession):
    """Session that runs until stopped, or exits with ``exit_code`` if set."""

    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.stop_threads: list[threading.Thread] = []
        self.on_collect: Callable[[], None] = lambda: None

    def login_user(self) -> str:
        return "flowmesh"

    def wait_ready(self, timeout_sec: float) -> int:
        if self.stopped.is_set():
            raise ExecutionError("Session exited before SSH became ready")
        self.ready.set()
        return 2222

    def poll(self) -> int | None:
        return self.exit_code

    def finish_requested(self) -> bool:
        return False

    def established_connections(self) -> int | None:
        return 1

    def output_size_bytes(self) -> int | None:
        return None

    def collect_output(self, destination: Path) -> None:
        self.on_collect()

    def stop(self, timeout_sec: float) -> None:
        self.stop_threads.append(threading.current_thread())
        self.stopped.set()

    def cleanup(self) -> None:
        return None


class _Backend:
    def __init__(
        self, session: _Session, on_start: Callable[[], None] = lambda: None
    ) -> None:
        self.session = session
        self.on_start = on_start
        self.started = 0

    def prepare(self) -> None:
        return None

    def start_session(self, request: SessionRequest) -> SSHSession:
        self.started += 1
        self.on_start()
        return self.session

    def session_address(self, access_mode: str) -> str:
        return "worker.local"

    def session_scope(self, access_mode: str) -> str:
        return "host"


def _executor(tmp_path: Path, backend: _Backend) -> SSHExecutor:
    executor = SSHExecutor(make_live_worker_config(tmp_path))
    executor._backend = cast(Any, backend)
    return executor


def _run(executor: SSHExecutor, tmp_path: Path, control: RunControl) -> SSHResult:
    return executor.run(_task_message(), tmp_path / "out", control)


class TestBeforeStart:
    def test_cancel_raises_without_starting_a_session(self, tmp_path: Path) -> None:
        backend = _Backend(_Session())
        control = RunControl("tsk-ssh")
        control.request_cancel()

        with pytest.raises(TaskCancelledError):
            _run(_executor(tmp_path, backend), tmp_path, control)
        assert backend.started == 0

    def test_stop_succeeds_without_starting_a_session(self, tmp_path: Path) -> None:
        backend = _Backend(_Session())
        control = RunControl("tsk-ssh")
        control.request_stop()

        result = _run(_executor(tmp_path, backend), tmp_path, control)
        assert result.exit_code == 0
        assert backend.started == 0


class TestDuringStart:
    """A signal landing while the session starts stops it before it is ready."""

    def test_cancel_is_reported_as_cancelled(self, tmp_path: Path) -> None:
        control = RunControl("tsk-ssh")
        session = _Session()
        backend = _Backend(session, on_start=control.request_cancel)

        with pytest.raises(TaskCancelledError):
            _run(_executor(tmp_path, backend), tmp_path, control)
        assert session.stopped.is_set()

    def test_stop_finishes_successfully(self, tmp_path: Path) -> None:
        control = RunControl("tsk-ssh")
        session = _Session()
        backend = _Backend(session, on_start=control.request_stop)

        result = _run(_executor(tmp_path, backend), tmp_path, control)
        assert result.exit_code == 0
        assert session.stopped.is_set()


class TestWhileRunning:
    def test_cancel_callback_stops_the_session(self, tmp_path: Path) -> None:
        session = _Session()
        executor = _executor(tmp_path, _Backend(session))
        control = RunControl("tsk-ssh")
        errors: list[BaseException] = []

        def run() -> None:
            try:
                _run(executor, tmp_path, control)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        assert session.ready.wait(5.0)
        control.request_cancel()
        thread.join(5.0)

        assert not thread.is_alive()
        assert threading.current_thread() in session.stop_threads
        assert len(errors) == 1 and isinstance(errors[0], TaskCancelledError)

    def test_stop_callback_finishes_the_session(self, tmp_path: Path) -> None:
        session = _Session()
        executor = _executor(tmp_path, _Backend(session))
        control = RunControl("tsk-ssh")
        results: list[SSHResult] = []

        thread = threading.Thread(
            target=lambda: results.append(_run(executor, tmp_path, control))
        )
        thread.start()
        assert session.ready.wait(5.0)
        control.request_stop()
        thread.join(5.0)

        assert not thread.is_alive()
        assert threading.current_thread() in session.stop_threads
        assert [result.exit_code for result in results] == [0]


class TestAfterRun:
    def test_callbacks_are_unregistered_when_the_run_ends(self, tmp_path: Path) -> None:
        session = _Session(exit_code=0)
        control = RunControl("tsk-ssh")

        _run(_executor(tmp_path, _Backend(session)), tmp_path, control)
        stops = len(session.stop_threads)
        control.request_cancel()
        control.request_stop()

        assert len(session.stop_threads) == stops

    def test_cancel_during_teardown_is_tolerated(self, tmp_path: Path) -> None:
        """The callback and the run's own teardown may both stop the session."""
        session = _Session(exit_code=0)
        control = RunControl("tsk-ssh")
        session.on_collect = control.request_cancel

        result = _run(_executor(tmp_path, _Backend(session)), tmp_path, control)

        assert result.exit_code == 0
        assert len(session.stop_threads) == 2
