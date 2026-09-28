"""Which run a cancel or graceful-stop signal reaches, whenever it arrives."""

import queue
import signal
import threading
from collections.abc import Callable, Iterator
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
from worker.executors.base_executor import Executor, ExecutorTask, RunControl
from worker.runner import _PENDING_SIGNAL_TTL_SEC, Runner


class _SameThreadGuard:
    """A lock that fails instead of deadlocking when its holder re-acquires it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None

    def __enter__(self) -> None:
        if self._owner == threading.get_ident():
            raise AssertionError("lock re-acquired by the thread holding it")
        self._lock.acquire()
        self._owner = threading.get_ident()

    def __exit__(self, *_: object) -> None:
        self._owner = None
        self._lock.release()


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

    def test_stop_takes_no_lock_and_leaves_the_work_to_the_shutdown_thread(
        self, tmp_path: Path
    ) -> None:
        """A signal handler runs on the task-loop thread, which may hold any lock."""
        runner, lifecycle = _runner(tmp_path, _RecordingExecutor(), [])
        runner._cancel_lock = _SameThreadGuard()  # type: ignore[assignment]
        stopped_on: list[threading.Thread] = []
        stopped = threading.Event()

        def record_stop() -> None:
            stopped_on.append(threading.current_thread())
            stopped.set()

        lifecycle.stop.side_effect = record_stop
        logged_on: list[threading.Thread] = []
        for level in ("debug", "info", "warning", "error", "exception"):
            getattr(runner.logger, level).side_effect = lambda *_, **__: (
                logged_on.append(threading.current_thread())
            )
        runner._start_shutdown_thread()
        try:
            control = runner._begin_run("tsk-a")
            with runner._cancel_lock:
                runner.stop()
                assert not control.cancel_requested
            assert stopped.wait(5.0)
            assert control.wait_for_cancel(5.0)
        finally:
            runner._stop_shutdown_thread()

        assert stopped_on != [threading.current_thread()]
        assert logged_on and threading.current_thread() not in logged_on
        assert runner._shutdown_thread is None

    def test_stop_before_start_skips_every_task(self, tmp_path: Path) -> None:
        executor = _RecordingExecutor()
        runner, lifecycle = _runner(tmp_path, executor, ["tsk-a"])

        runner.stop()
        runner.start()

        assert executor.controls == {}
        lifecycle.stop.assert_called_once_with()

    def test_signal_handler_stop_ends_an_idle_runner(self, tmp_path: Path) -> None:
        """End to end: a signal handler calling stop() ends a runner awaiting tasks."""
        tasks: queue.Queue[WorkerTaskMessage | None] = queue.Queue()

        def task_stream() -> Iterator[WorkerTaskMessage]:
            while (msg := tasks.get()) is not None:
                yield msg

        runner, lifecycle = _runner(tmp_path, _RecordingExecutor(), [])
        runner.task_stream = task_stream()
        lifecycle.stop.side_effect = lambda: tasks.put(None)
        previous = signal.signal(signal.SIGUSR1, lambda *_: runner.stop())
        send = threading.Timer(
            0.2, signal.pthread_kill, (threading.get_ident(), signal.SIGUSR1)
        )
        # Ends the loop without stopping the lifecycle if the signal never does.
        give_up = threading.Timer(5.0, tasks.put, (None,))
        try:
            send.start()
            give_up.start()
            runner.start()
        finally:
            send.cancel()
            give_up.cancel()
            signal.signal(signal.SIGUSR1, previous)

        lifecycle.stop.assert_called_once_with()
        assert runner._shutdown_requested


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
