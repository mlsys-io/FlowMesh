"""Eligible-worker computation and the no-eligible-worker grace in the dispatcher."""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from server.dispatcher import Dispatcher
from server.task.models import TaskRecord


def _dispatcher(
    satisfying_ids: list[str], grace_sec: int = 60
) -> "_CapturingDispatcher":
    registry = mock.Mock()
    registry.satisfying_workers.return_value = [
        SimpleNamespace(id=wid) for wid in satisfying_ids
    ]
    return _CapturingDispatcher(
        runtime=mock.Mock(),
        worker_registry=registry,
        results_dir=Path("/tmp/flowmesh-test"),  # noqa: S108 - unused in these paths
        logger=logging.getLogger("test_retry_eligibility"),
        no_worker_grace_sec=grace_sec,
    )


class _CapturingDispatcher(Dispatcher):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failed: list[tuple[str, str, dict[str, Any]]] = []
        self.requeued: list[tuple[str, dict[str, Any]]] = []

    def _fail_task(self, task_id: str, error_message: str, **kwargs: Any) -> None:
        self.failed.append((task_id, error_message, kwargs))

    def _requeue_task(self, task_id: str, **kwargs: Any) -> None:
        self.requeued.append((task_id, kwargs))


def _record(
    selected_worker: list[str] | None = None,
    no_eligible_since: float | None = None,
    last_error: str | None = None,
) -> TaskRecord:
    rec = SimpleNamespace(
        task=SimpleNamespace(),
        selected_worker=selected_worker,
        no_eligible_since=no_eligible_since,
        last_error=last_error,
    )
    return cast(TaskRecord, cast(Any, rec))


def test_eligible_worker_ids_returns_satisfying_set() -> None:
    disp = _dispatcher(["w-1", "w-2", "w-3"])
    assert disp.eligible_worker_ids(_record()) == {"w-1", "w-2", "w-3"}


def test_eligible_worker_ids_intersects_selected_worker() -> None:
    disp = _dispatcher(["w-1", "w-2", "w-3"])
    record = _record(selected_worker=["w-2", "w-9"])
    assert disp.eligible_worker_ids(record) == {"w-2"}


def test_no_eligible_worker_waits_within_grace() -> None:
    disp = _dispatcher([], grace_sec=60)
    record = _record()
    with mock.patch("server.dispatcher.base.time.time", return_value=1000.0):
        result = disp._handle_no_eligible_worker("tsk-1", record)
    assert result is False
    assert record.no_eligible_since == 1000.0
    assert disp.failed == []
    assert disp.requeued and disp.requeued[0][1]["count_retry"] is False


def test_no_eligible_worker_fails_after_grace() -> None:
    disp = _dispatcher([], grace_sec=60)
    record = _record(no_eligible_since=1000.0, last_error="bad spec")
    with mock.patch("server.dispatcher.base.time.time", return_value=1100.0):
        disp._handle_no_eligible_worker("tsk-1", record)
    assert disp.requeued == []
    assert len(disp.failed) == 1
    task_id, message, kwargs = disp.failed[0]
    assert task_id == "tsk-1"
    assert message == "bad spec"
    assert kwargs["payload"]["reason"] == "no_eligible_worker"
