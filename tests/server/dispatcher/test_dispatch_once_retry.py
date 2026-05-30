"""dispatch_once retry/exhaustion/grace branches against a real task record."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from server.dispatcher import Dispatcher
from server.task.runtime import TaskRuntime

_ECHO_WORKFLOW = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: retry-branches
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
"""


class _WorkflowRegistryStub:
    async def register_workflow_async(self, workflow_id: str, tasks: list[Any]) -> None:
        return None

    def mark_task_dispatched(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_done(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_failed(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_pending(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_cancelled(self, workflow_id: str, *task_ids: str) -> None: ...


class _CapturingDispatcher(Dispatcher):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failed: list[tuple[str, str, dict[str, Any]]] = []
        self.requeued: list[tuple[str, dict[str, Any]]] = []

    def _fail_task(self, task_id: str, error_message: str, **kwargs: Any) -> None:
        self.failed.append((task_id, error_message, kwargs))

    def _requeue_task(self, task_id: str, **kwargs: Any) -> None:
        self.requeued.append((task_id, kwargs))


def _setup(
    idle_ids: list[str],
    satisfying_ids: list[str],
    failed: list[str] | None = None,
) -> tuple[_CapturingDispatcher, str]:
    runtime = TaskRuntime(
        cast(Any, _WorkflowRegistryStub()),
        cast(Any, mock.Mock()),
        logging.getLogger("test_dispatch_once_retry"),
    )
    _, results = asyncio.run(
        runtime.register("owner", "org", _ECHO_WORKFLOW, format="native")
    )
    task_id = results[0].task_id
    record = runtime.get_record(task_id)
    assert record is not None
    record.failed_workers = list(failed or [])

    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [
        SimpleNamespace(id=wid) for wid in idle_ids
    ]
    registry.satisfying_workers.return_value = [
        SimpleNamespace(id=wid) for wid in satisfying_ids
    ]
    disp = _CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        results_dir=Path("/tmp/flowmesh-test"),  # noqa: S108 - unused in these paths
        logger=logging.getLogger("test_dispatch_once_retry"),
        no_worker_grace_sec=60,
    )
    return disp, task_id


def test_waits_when_untried_eligible_worker_is_busy() -> None:
    # Only idle candidate already failed the task, but another eligible worker
    # exists (busy) — wait rather than fail.
    disp, tid = _setup(idle_ids=["w-1"], satisfying_ids=["w-1", "w-2"], failed=["w-1"])
    disp.dispatch_once(tid)
    assert disp.failed == []
    assert len(disp.requeued) == 1
    _, kwargs = disp.requeued[0]
    assert kwargs["reason"] == "untried_workers_busy"
    assert kwargs["count_retry"] is False


def test_fails_when_all_eligible_workers_already_failed() -> None:
    disp, tid = _setup(idle_ids=["w-1"], satisfying_ids=["w-1"], failed=["w-1"])
    record = disp._runtime.get_record(tid)
    assert record is not None
    record.last_error = "boom"
    disp.dispatch_once(tid)
    assert disp.requeued == []
    assert len(disp.failed) == 1
    _, message, kwargs = disp.failed[0]
    assert message == "boom"
    assert kwargs["payload"]["reason"] == "eligible_workers_exhausted"


def test_zero_eligible_enters_grace_then_waits() -> None:
    disp, tid = _setup(idle_ids=[], satisfying_ids=[])
    with mock.patch("server.dispatcher.base.time.time", return_value=1000.0):
        disp.dispatch_once(tid)
    record = disp._runtime.get_record(tid)
    assert record is not None and record.no_eligible_since == 1000.0
    assert disp.failed == []
    assert len(disp.requeued) == 1
    assert disp.requeued[0][1]["reason"] == "no_eligible_worker"
