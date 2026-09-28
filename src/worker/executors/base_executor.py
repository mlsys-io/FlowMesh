"""
Executor base class, its per-run control, and a minimal example implementation.

Usage:
    from shared.schemas.result import BaseExecutorResult
    from worker.executors.base_executor import Executor, ExecutionError

    class MyResult(BaseExecutorResult):
        echo: str

    class MyExecutor(Executor):
        name = "my-executor"
        def run(
            self, task: ExecutorTask, out_dir: Path, control: RunControl
        ) -> MyResult:
            # ... your logic ...
            return MyResult(echo=task.task_id)

Contract:
- Implement `run(task: ExecutorTask, out_dir: Path, control: RunControl)
  -> BaseExecutorResult`.
  The runner writes the returned model to `out_dir/results.json` and
  injects the top-level `_artifacts` context — executors should not write
  that file themselves on the success path.
- Drop generated files under `out_dir/artifacts/` (uploaded to the server
  when the task has an HTTP destination) or `scratch_dir(out_dir)` for
  local-only scratch data.
- Optionally override `prepare()` and `teardown()` for lifecycle hooks.
- Raise `ExecutionError` for user-visible failures.
- Read `control` to end early. On cancel, raise `TaskCancelledError`. On a
  graceful stop, finish and return a result, or raise `TaskCancelledError` if
  the run has nothing to report yet. Register `control.on_cancel` /
  `control.on_stop` callbacks to interrupt blocking work.
"""

import itertools
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from shared.schemas.result import BaseExecutorResult
from shared.tasks import MergedChildTaskStrict
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerHardware, WorkerTaskMessage
from worker.config import WorkerConfig
from worker.lifecycle import Lifecycle

type ExecutorTask = WorkerTaskMessage
type TaskReference = WorkerTaskMessage | MergedChildTaskStrict

logger = logging.getLogger(__name__)

SpecT = TypeVar("SpecT", bound=TaskSpecStrictBase)


class ExecutionError(RuntimeError):
    """Raised when an executor fails in an expected / controlled way.

    ``retryable`` marks failures that may succeed on another worker (transient network
    or I/O errors). Deterministic failures (invalid spec, unsupported config) leave it
    ``False`` so they fail without retry.
    """

    def __init__(self, *args: object, retryable: bool = False) -> None:
        super().__init__(*args)
        self.retryable = retryable


class TaskCancelledError(RuntimeError):
    """Raised when a task is explicitly cancelled while running."""


type Callback = Callable[[], None]


class RunControl:
    """Cancel and graceful-stop signals for a single executor run.

    Both signals are latched and one-way. Every method is thread-safe. Only the
    runner raises signals; executors read them, wait on them, or register callbacks.
    """

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._stop = threading.Event()
        self._cancel_callbacks: dict[int, Callback] = {}
        self._stop_callbacks: dict[int, Callback] = {}
        self._tokens = itertools.count()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def wait_for_cancel(self, timeout: float) -> bool:
        """Block up to ``timeout`` seconds; return whether cancel was requested."""
        return self._cancel.wait(timeout)

    def raise_if_cancelled(self, message: str) -> None:
        """Raise ``TaskCancelledError(message)`` if cancel was requested."""
        if self._cancel.is_set():
            raise TaskCancelledError(message)

    def on_cancel(self, fn: Callback) -> Callback:
        """Run ``fn`` once when cancel is requested; return its unregister function.

        ``fn`` runs immediately on the calling thread if cancel was already requested.
        """
        return self._register(self._cancel, self._cancel_callbacks, fn)

    def on_stop(self, fn: Callback) -> Callback:
        """Run ``fn`` once when stop is requested; return its unregister function.

        ``fn`` runs immediately on the calling thread if stop was already requested.
        """
        return self._register(self._stop, self._stop_callbacks, fn)

    def request_cancel(self) -> None:
        self._raise(self._cancel, self._cancel_callbacks)

    def request_stop(self) -> None:
        self._raise(self._stop, self._stop_callbacks)

    def _register(
        self, signal: threading.Event, callbacks: dict[int, Callback], fn: Callback
    ) -> Callback:
        with self._lock:
            if not signal.is_set():
                token = next(self._tokens)
                callbacks[token] = fn

                def unregister() -> None:
                    with self._lock:
                        callbacks.pop(token, None)

                return unregister
        self._invoke(fn)
        return lambda: None

    def _raise(self, signal: threading.Event, callbacks: dict[int, Callback]) -> None:
        with self._lock:
            if signal.is_set():
                return
            signal.set()
            pending = list(callbacks.values())
            callbacks.clear()
        for fn in pending:
            self._invoke(fn)

    def _invoke(self, fn: Callback) -> None:
        try:
            fn()
        except Exception:
            logger.warning(
                "Run control callback for task %s raised",
                self._task_id,
                exc_info=True,
            )


class Executor(ABC):
    """Abstract task executor.

    Subclasses must implement `run` and may override `prepare` and `teardown`.
    """

    name: str = "executor"
    """Human-readable identifier for logging/telemetry"""
    supported_task_types: ClassVar[frozenset[TaskType]] = frozenset()
    """Types of tasks this executor can service"""

    def __init__(
        self,
        config: WorkerConfig,
        hardware: WorkerHardware | None = None,
        lifecycle: Lifecycle | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._hardware = hardware
        self._lifecycle = lifecycle

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        """Whether this executor can run on the worker.

        Override to declare a runtime dependency the worker must satisfy. An unavailable
        executor is skipped at startup, so it never registers and the worker advertises
        no capability for it.
        """
        return True

    def emit_update(self, task_id: str, payload: dict[str, Any]) -> None:
        """Emit a mid-task TASK_UPDATE event.

        Calls the lifecycle if one was injected; otherwise a no-op.
        Executors that produce interim results (e.g. SSHExecutor) should call
        this method; all other executors can ignore it.
        """
        if self._lifecycle is not None:
            self._lifecycle.notify_task_update(task_id, payload)

    def publish_endpoint(self, endpoint_id: str, port: int) -> None:
        """Offer a local port for the supervisor to relay to.

        Calls the lifecycle if one was injected; otherwise a no-op.
        """
        if self._lifecycle is not None:
            self._lifecycle.publish_endpoint(endpoint_id, port)

    def withdraw_endpoint(self, endpoint_id: str) -> None:
        """Stop offering a local port. Safe to call for an unpublished id."""
        if self._lifecycle is not None:
            self._lifecycle.withdraw_endpoint(endpoint_id)

    def prepare(self) -> None:
        """Optional: called once before the first `run`.
        Use for lazy initialization (e.g., loading models, warming caches).
        """
        return None

    @abstractmethod
    def run(
        self, task: ExecutorTask, out_dir: Path, control: RunControl
    ) -> BaseExecutorResult:
        """Execute a single task.

        Args:
            task: Parsed task payload.
            out_dir: Directory for any outputs. Implementations should create it
            if needed.
            control: Cancel and graceful-stop signals for this run. Executors that
            can end early read it; others may ignore it.

        Returns:
            A ``BaseExecutorResult`` subclass instance.

        Raises:
            ExecutionError: for expected, user-facing failures.
            Exception: for unexpected errors (will be logged by the caller).
        """
        raise NotImplementedError

    @staticmethod
    def require_spec(task: ExecutorTask, spec_type: type[SpecT]) -> SpecT:
        spec = task.spec
        if not isinstance(spec, spec_type):
            raise ExecutionError(
                f"{task.task_id} received unexpected spec type "
                f"{spec.__class__.__name__}; expected {spec_type.__name__}"
            )
        return spec

    def teardown(self) -> None:
        """Optional: called when the worker is shutting down."""
        return None

    def cleanup_after_run(self) -> None:
        """Optional: called after every `run` invocation (even on failure)."""
        return None

    # ---------- Convenience helpers ----------
    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


# -------- Minimal example implementation --------
class EchoResult(BaseExecutorResult):
    ok: bool = True
    executor: str
    task_id: str
    task_type: str


class EchoExecutor(Executor):
    name = "echo"
    supported_task_types = frozenset({TaskType.ECHO})

    def run(self, task: ExecutorTask, out_dir: Path, control: RunControl) -> EchoResult:
        return EchoResult(
            executor=self.name,
            task_id=task.task_id,
            task_type=task.spec.taskType,
        )
