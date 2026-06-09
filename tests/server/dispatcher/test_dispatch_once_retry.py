"""dispatch_once retry/exhaustion/grace branches against a real task record."""

import asyncio
import logging
from typing import Any, cast
from unittest import mock

from server.task.runtime import TaskRuntime
from tests.server.dispatcher.helpers import (
    CapturingDispatcher,
    WorkflowRegistryStub,
    make_capturing_dispatcher,
)

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


def _setup(
    idle_ids: list[str],
    satisfying_ids: list[str],
    failed: list[str] | None = None,
) -> tuple[CapturingDispatcher, str]:
    runtime = TaskRuntime(
        cast(Any, WorkflowRegistryStub()),
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

    disp = make_capturing_dispatcher(
        runtime=runtime, idle_ids=idle_ids, satisfying_ids=satisfying_ids
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


def test_exhausted_requeues_within_grace() -> None:
    # Every eligible worker has failed it: requeue and wait out the grace.
    disp, tid = _setup(idle_ids=["w-1"], satisfying_ids=["w-1"], failed=["w-1"])
    with mock.patch("server.dispatcher.base.time.time", return_value=1000.0):
        disp.dispatch_once(tid)
    record = disp._runtime.get_record(tid)
    assert record is not None and record.no_eligible_since == 1000.0
    assert disp.failed == []
    assert len(disp.requeued) == 1
    assert disp.requeued[0][1]["reason"] == "eligible_workers_exhausted"


def test_all_eligible_failed_but_busy_grace_then_fails() -> None:
    # No idle worker (the only eligible one is busy) and it already failed the
    # task: grace-then-fail rather than waiting on a worker that would refail.
    disp, tid = _setup(idle_ids=[], satisfying_ids=["w-1"], failed=["w-1"])
    with mock.patch("server.dispatcher.base.time.time", return_value=1000.0):
        disp.dispatch_once(tid)
    record = disp._runtime.get_record(tid)
    assert record is not None and record.no_eligible_since == 1000.0
    assert disp.failed == []
    assert len(disp.requeued) == 1
    assert disp.requeued[0][1]["reason"] == "eligible_workers_exhausted"


def test_exhausted_fails_after_grace() -> None:
    disp, tid = _setup(idle_ids=["w-1"], satisfying_ids=["w-1"], failed=["w-1"])
    record = disp._runtime.get_record(tid)
    assert record is not None
    record.last_error = "boom"
    record.no_eligible_since = 1000.0
    with mock.patch("server.dispatcher.base.time.time", return_value=1100.0):
        disp.dispatch_once(tid)
    assert disp.requeued == []
    assert len(disp.failed) == 1
    _, message, kwargs = disp.failed[0]
    assert message == "boom"
    assert kwargs["payload"]["reason"] == "eligible_workers_exhausted"
    assert kwargs["payload"]["failed_workers"] == ["w-1"]


def test_zero_eligible_enters_grace_then_waits() -> None:
    disp, tid = _setup(idle_ids=[], satisfying_ids=[])
    with mock.patch("server.dispatcher.base.time.time", return_value=1000.0):
        disp.dispatch_once(tid)
    record = disp._runtime.get_record(tid)
    assert record is not None and record.no_eligible_since == 1000.0
    assert disp.failed == []
    assert len(disp.requeued) == 1
    assert disp.requeued[0][1]["reason"] == "no_eligible_worker"
