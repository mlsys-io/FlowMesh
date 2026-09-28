"""Which run a cancel or graceful-stop signal reaches, whenever it arrives."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from shared.schemas.result import EchoResult
from shared.tasks.specs import EchoSpecStrict
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import (
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.executors.base_executor import Executor, ExecutorTask
from worker.executors.run_control import RunControl
from worker.runner import _PENDING_SIGNAL_TTL_SEC, Runner


class _RecordingExecutor(Executor):
    """Records each run's control and what it saw on entry."""

    name = "recording"
    supported_task_types = frozenset({TaskType.ECHO})

    def __init__(self) -> None:
        super().__init__(make_worker_config())
        self.controls: dict[str, RunControl] = {}
        self.seen: dict[str, tuple[bool, bool]] = {}
        self.on_run: dict[str, Callable[[RunControl], None]] = {}

    def run(self, task: ExecutorTask, out_dir: Path, control: RunControl) -> EchoResult:
        self.controls[task.task_id] = control
        hook = self.on_run.get(task.task_id)
        if hook is not None:
            hook(control)
        self.seen[task.task_id] = (control.cancel_requested, control.stop_requested)
        control.raise_if_cancelled(f"{task.task_id} cancelled")
        return EchoResult()


def _message(task_id: str) -> WorkerTaskMessage:
    return make_worker_task_message(
        spec=EchoSpecStrict(taskType=TaskType.ECHO), task_id=task_id
    )


def _runner(
    tmp_path: Path, executor: _RecordingExecutor, task_ids: list[str]
) -> tuple[Runner, MagicMock]:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.live_gpu_availability.return_value = {}
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[_message(task_id) for task_id in task_ids],
        results_dir=tmp_path,
        hardware=make_worker_hardware(),
        executors={"echo": executor},
        default_executor=executor,
        logger=MagicMock(),
    )
    return runner, lifecycle


def _reported(lifecycle: MagicMock, method: str) -> list[Any]:
    return [c.args[0] for c in getattr(lifecycle, method).call_args_list]


class TestBeforeStart:
    def test_cancel_before_the_task_arrives_skips_the_executor(
        self, tmp_path: Path
    ) -> None:
        executor = _RecordingExecutor()
        runner, lifecycle = _runner(tmp_path, executor, ["tsk-a"])
        runner._handle_interrupt("tsk-a", "user")

        runner.start()

        assert executor.controls == {}
        assert _reported(lifecycle, "set_cancelled") == ["tsk-a"]
        assert runner._controls == {}

    def test_cancel_during_executor_selection_reaches_the_run(
        self, tmp_path: Path
    ) -> None:
        executor = _RecordingExecutor()
        runner, lifecycle = _runner(tmp_path, executor, ["tsk-a"])
        lifecycle.notify_task_started.side_effect = (
            lambda task_id, **_: runner._handle_interrupt(task_id, "user")
        )

        runner.start()

        assert executor.seen["tsk-a"] == (True, False)
        assert _reported(lifecycle, "set_cancelled") == ["tsk-a"]

    def test_stop_before_the_task_arrives_reaches_the_run(self, tmp_path: Path) -> None:
        executor = _RecordingExecutor()
        runner, lifecycle = _runner(tmp_path, executor, ["tsk-a"])
        runner._handle_stop("tsk-a", "user")

        runner.start()

        assert executor.seen["tsk-a"] == (False, True)
        assert _reported(lifecycle, "set_succeeded") == ["tsk-a"]


class TestAfterFinish:
    def test_late_cancel_for_a_finished_task_spares_the_next_one(
        self, tmp_path: Path
    ) -> None:
        executor = _RecordingExecutor()
        runner, lifecycle = _runner(tmp_path, executor, ["tsk-a", "tsk-b"])
        executor.on_run["tsk-b"] = lambda _: runner._handle_interrupt("tsk-a", "late")

        runner.start()

        assert executor.seen["tsk-b"] == (False, False)
        assert _reported(lifecycle, "set_succeeded") == ["tsk-a", "tsk-b"]
        assert not executor.controls["tsk-a"].cancel_requested
        pending = runner._controls["tsk-a"].control
        assert pending.cancel_requested
        assert pending is not executor.controls["tsk-a"]

    def test_a_finished_task_leaves_no_control(self, tmp_path: Path) -> None:
        executor = _RecordingExecutor()
        runner, _ = _runner(tmp_path, executor, ["tsk-a"])

        runner.start()

        assert runner._controls == {}
        assert runner._current_control is None


class TestShutdown:
    def test_stop_cancels_the_current_run(self, tmp_path: Path) -> None:
        """``stop()`` runs in a signal handler on the task-loop thread."""
        executor = _RecordingExecutor()
        runner, lifecycle = _runner(tmp_path, executor, ["tsk-a"])
        cancelled: list[bool] = []

        def shut_down(control: RunControl) -> None:
            runner.stop()
            cancelled.append(control.wait_for_cancel(5.0))

        executor.on_run["tsk-a"] = shut_down

        runner.start()

        assert cancelled == [True]
        assert _reported(lifecycle, "set_cancelled") == ["tsk-a"]


class TestRegistry:
    def test_sweep_drops_only_expired_pending_signals(self, tmp_path: Path) -> None:
        runner, _ = _runner(tmp_path, _RecordingExecutor(), [])
        runner._handle_interrupt("tsk-old", "user")
        runner._handle_stop("tsk-new", "user")
        running = runner._begin_run("tsk-running")
        for task_id in ("tsk-old", "tsk-running"):
            runner._controls[task_id].created_at -= _PENDING_SIGNAL_TTL_SEC + 1

        runner._sweep_pending_controls()

        assert set(runner._controls) == {"tsk-new", "tsk-running"}
        assert runner._controls["tsk-running"].control is running

    def test_signal_for_the_running_task_reuses_its_control(
        self, tmp_path: Path
    ) -> None:
        runner, _ = _runner(tmp_path, _RecordingExecutor(), [])
        running = runner._begin_run("tsk-a")

        runner._handle_interrupt("tsk-a", "user")

        assert running.cancel_requested
        runner._end_run("tsk-a")
        assert runner._controls == {}
