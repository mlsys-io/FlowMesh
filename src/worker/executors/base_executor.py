"""
Executor base class and a minimal example implementation.

Usage:
    from executor_base import Executor, ExecutionError, EchoExecutor

    class MyExecutor(Executor):
        name = "my-executor"
        def run(self, task: ExecutorTask, out_dir: Path) -> dict:
            # ... your logic ...
            return {"ok": True, "echo": task.task_id}

Contract:
- Implement `run(task: ExecutorTask, out_dir: Path) -> dict`. The runner
  writes the returned dict to `out_dir/results.json` and injects the
  top-level `_artifacts` context — executors should not write that file
  themselves on the success path.
- Drop generated files under `out_dir/artifacts/` (uploaded to the server
  when the task has an HTTP destination) or `scratch_dir(out_dir)` for
  local-only scratch data.
- Optionally override `prepare()` and `teardown()` for lifecycle hooks.
- Raise `ExecutionError` for user-visible failures.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from shared.tasks import MergedChildTaskStrict
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.worker_message import WorkerHardware, WorkerTaskMessage
from worker.config import WorkerConfig
from worker.lifecycle import Lifecycle

type ExecutorTask = WorkerTaskMessage
type TaskReference = WorkerTaskMessage | MergedChildTaskStrict

SpecT = TypeVar("SpecT", bound=TaskSpecStrictBase)


class ExecutionError(RuntimeError):
    """Raised when an executor fails in an expected / controlled way."""


class TaskCancelledError(RuntimeError):
    """Raised when a task is explicitly cancelled while running."""


class Executor(ABC):
    """Abstract task executor.

    Subclasses must implement `run` and may override `prepare` and `teardown`.
    """

    #: Human-readable identifier for logging/telemetry
    name: str = "executor"

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

    def emit_update(self, task_id: str, payload: dict[str, Any]) -> None:
        """Emit a mid-task TASK_UPDATE event.

        Calls the lifecycle if one was injected; otherwise a no-op.
        Executors that produce interim results (e.g. SSHExecutor) should call
        this method; all other executors can ignore it.
        """
        if self._lifecycle is not None:
            self._lifecycle.notify_task_update(task_id, payload)

    def prepare(self) -> None:
        """Optional: called once before the first `run`.
        Use for lazy initialization (e.g., loading models, warming caches).
        """
        return None

    @abstractmethod
    def run(self, task: ExecutorTask, out_dir: Path) -> dict[str, Any]:
        """Execute a single task.

        Args:
            task: Parsed task payload.
            out_dir: Directory for any outputs. Implementations should create it
            if needed.

        Returns:
            A JSON-serializable dictionary summarizing the result.

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

    def cancel(self, task_id: str) -> None:
        """Signal the executor to abort the current task."""
        return None

    def stop(self, task_id: str) -> None:
        """Signal the executor to finish the current task successfully."""
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
class EchoExecutor(Executor):
    name = "echo"

    def run(self, task: ExecutorTask, out_dir: Path) -> dict[str, Any]:
        return {
            "ok": True,
            "executor": self.name,
            "task_id": task.task_id,
            "task_type": task.spec.taskType,
        }
