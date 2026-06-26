"""MP executor lifecycle tests."""

import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import psutil
import pytest

from shared.schemas.result import BaseExecutorResult
from shared.tasks import TaskType
from shared.tasks.specs import EchoSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import (
    make_live_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.executors import mp_executor as mp_executor_module
from worker.executors.base_executor import ExecutionError, Executor
from worker.executors.mp_executor import MPExecutor


class _SimpleMPResult(BaseExecutorResult):
    ok: bool = True
    task_id: str


class _SimpleMPExecutor(Executor):
    name = "simple_mp"

    def run(self, task, out_dir: Path) -> _SimpleMPResult:
        return _SimpleMPResult(task_id=task.task_id)

    def cleanup_after_run(self) -> None:
        return None


class _HardCrashExecutor(Executor):
    """Exits the subprocess mid-run, mimicking an OOM kill or native crash."""

    name = "hard_crash"

    def run(self, task, out_dir: Path) -> _SimpleMPResult:
        os._exit(137)

    def cleanup_after_run(self) -> None:
        return None


class _SoftCrashExecutor(Executor):
    """Raises an unexpected (non-``ExecutionError``) exception."""

    name = "soft_crash"

    def run(self, task, out_dir: Path) -> _SimpleMPResult:
        raise RuntimeError("boom")

    def cleanup_after_run(self) -> None:
        return None


class _SoftCrashCleanupExecutor(Executor):
    """Raises an unexpected exception but records that cleanup ran."""

    name = "soft_crash_cleanup"
    _marker: Path | None = None

    def run(self, task, out_dir: Path) -> _SimpleMPResult:
        self._marker = Path(out_dir) / "cleanup_ran"
        raise RuntimeError("boom")

    def cleanup_after_run(self) -> None:
        if self._marker is not None:
            self._marker.write_text("ok")


class _OrphanSpawningCrashExecutor(Executor):
    """Spawns a long-lived grandchild, records its PID, then hard-crashes.

    Mimics vLLM, whose engine-core process holds the GPU and outlives a killed
    worker subprocess unless the whole process group is signalled.
    """

    name = "orphan_spawn_crash"

    def run(self, task, out_dir: Path) -> _SimpleMPResult:
        pid_file = Path(out_dir) / "grandchild.pid"
        code = (
            "import os, sys, time\n"
            "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
            "time.sleep(300)\n"
        )
        subprocess.Popen([sys.executable, "-c", code, str(pid_file)])
        while not pid_file.exists() or not pid_file.read_text():
            time.sleep(0.01)
        os._exit(137)

    def cleanup_after_run(self) -> None:
        return None


class _ControlledErrorExecutor(Executor):
    """Raises a controlled ``ExecutionError`` (e.g. bad spec)."""

    name = "controlled_error"

    def run(self, task, out_dir: Path) -> _SimpleMPResult:
        raise ExecutionError("bad input")

    def cleanup_after_run(self) -> None:
        return None


class _RetryableErrorExecutor(Executor):
    """Raises a controlled ``ExecutionError`` flagged retryable."""

    name = "retryable_error"

    def run(self, task, out_dir: Path) -> _SimpleMPResult:
        raise ExecutionError("transient", retryable=True)

    def cleanup_after_run(self) -> None:
        return None


def _simple_task_message() -> WorkerTaskMessage:
    return make_worker_task_message(
        EchoSpecStrict(taskType=TaskType.ECHO, data={"items": ["test"]}),
        api_version="mloc/v1",
        kind="EchoTask",
        task_id=str(uuid.uuid4()),
        workflow_id="test-workflow",
        owner_id="test-owner",
        assigned_worker="test-worker",
        dispatched_at="2026-03-01T00:00:00Z",
    )


def _pid_running(pid: int) -> bool:
    """True while ``pid`` is a live (non-zombie) process."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_mp_executor_does_not_start_subprocess_until_first_run(tmp_path: Path) -> None:
    mp = MPExecutor(
        _SimpleMPExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    assert mp._shutdown is True
    assert mp._proc is None
    assert mp._cmd_q is None
    assert mp._res_q is None

    with tempfile.TemporaryDirectory() as out_dir:
        result = mp.run(_simple_task_message(), Path(out_dir))

    assert isinstance(result, _SimpleMPResult)
    assert result.ok is True
    assert mp._shutdown is False
    assert mp._proc is not None
    assert mp._proc.is_alive()

    mp.cleanup_after_run()

    if mp._proc is not None:
        assert not mp._proc.is_alive()


def test_mp_executor_reports_dead_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mp_executor_module, "_RESULT_POLL_INTERVAL_SEC", 0.2)
    mp = MPExecutor(
        _HardCrashExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        with pytest.raises(ExecutionError) as exc_info:
            mp.run(_simple_task_message(), Path(out_dir))

    assert exc_info.value.retryable is True

    # State is reset so the next run() can start a fresh subprocess.
    assert mp._shutdown is True
    assert mp._proc is None
    assert mp._cmd_q is None
    assert mp._res_q is None


def test_mp_executor_restarts_subprocess_after_unexpected_failure(
    tmp_path: Path,
) -> None:
    mp = MPExecutor(
        _SoftCrashExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        with pytest.raises(RuntimeError) as exc_info:
            mp.run(_simple_task_message(), Path(out_dir))

    assert not isinstance(exc_info.value, ExecutionError)

    # The subprocess is torn down so the next run() starts a clean one.
    assert mp._shutdown is True
    assert mp._proc is None
    assert mp._cmd_q is None
    assert mp._res_q is None


def test_mp_executor_runs_inner_cleanup_on_unexpected_failure(tmp_path: Path) -> None:
    mp = MPExecutor(
        _SoftCrashCleanupExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        marker = Path(out_dir) / "cleanup_ran"
        with pytest.raises(RuntimeError):
            mp.run(_simple_task_message(), Path(out_dir))

        # The still-responsive subprocess is shut down gracefully, so the inner
        # executor's cleanup runs (releasing e.g. the GPU) instead of being
        # hard-killed.
        assert marker.exists()

    assert mp._shutdown is True
    assert mp._proc is None


def test_mp_executor_reaps_orphaned_descendants_on_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mp_executor_module, "_RESULT_POLL_INTERVAL_SEC", 0.2)
    mp = MPExecutor(
        _OrphanSpawningCrashExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        pid_file = Path(out_dir) / "grandchild.pid"
        with pytest.raises(ExecutionError):
            mp.run(_simple_task_message(), Path(out_dir))
        grandchild_pid = int(pid_file.read_text())

    # Tearing down the dead subprocess signals its whole process group, so the
    # orphaned grandchild does not survive to hold the GPU.
    deadline = time.time() + 5.0
    while time.time() < deadline and _pid_running(grandchild_pid):
        time.sleep(0.05)
    assert not _pid_running(grandchild_pid)


def test_mp_executor_keeps_subprocess_after_controlled_error(tmp_path: Path) -> None:
    mp = MPExecutor(
        _ControlledErrorExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        with pytest.raises(ExecutionError) as exc_info:
            mp.run(_simple_task_message(), Path(out_dir))

    assert exc_info.value.retryable is False

    # A controlled error leaves the warm subprocess in place for reuse.
    assert mp._shutdown is False
    assert mp._proc is not None
    assert mp._proc.is_alive()

    mp.cleanup_after_run()


def test_mp_executor_propagates_retryable_flag(tmp_path: Path) -> None:
    mp = MPExecutor(
        _RetryableErrorExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    with tempfile.TemporaryDirectory() as out_dir:
        with pytest.raises(ExecutionError) as exc_info:
            mp.run(_simple_task_message(), Path(out_dir))

    # The inner executor's retryable flag survives the subprocess boundary.
    assert exc_info.value.retryable is True

    mp.cleanup_after_run()


def test_mp_executor_cleanup_before_run_is_noop(tmp_path: Path) -> None:
    mp = MPExecutor(
        _SimpleMPExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    mp.cleanup_after_run()

    assert mp._shutdown is True
    assert mp._proc is None
    assert mp._cmd_q is None
    assert mp._res_q is None
