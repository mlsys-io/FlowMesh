"""Eligible-worker computation and the no-worker grace in the dispatcher."""

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from server.task.models import TaskRecord
from tests.server.dispatcher.helpers import make_capturing_dispatcher


def _record(
    selected_worker: list[str] | None = None,
    no_eligible_since: float | None = None,
    last_error: str | None = None,
    last_failed_worker: str | None = None,
) -> TaskRecord:
    rec = SimpleNamespace(
        task=SimpleNamespace(),
        selected_worker=selected_worker,
        no_eligible_since=no_eligible_since,
        last_error=last_error,
        last_failed_worker=last_failed_worker,
    )
    return cast(TaskRecord, cast(Any, rec))


def test_eligible_worker_ids_returns_satisfying_set() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1", "w-2", "w-3"])
    assert disp.eligible_worker_ids(_record()) == {"w-1", "w-2", "w-3"}


def test_eligible_worker_ids_intersects_selected_worker() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1", "w-2", "w-3"])
    record = _record(selected_worker=["w-2", "w-9"])
    assert disp.eligible_worker_ids(record) == {"w-2"}


def test_grace_then_fail_waits_within_grace() -> None:
    disp = make_capturing_dispatcher(grace_sec=60)
    record = _record()
    with mock.patch("server.dispatcher.base.time.time", return_value=1000.0):
        result = disp._grace_then_fail(
            "tsk-1", record, reason="no_eligible_worker", message="no worker"
        )
    assert result is False
    assert record.no_eligible_since == 1000.0
    assert disp.failed == []
    assert disp.requeued and disp.requeued[0][1]["count_retry"] is False


def test_grace_then_fail_fails_after_grace() -> None:
    disp = make_capturing_dispatcher(grace_sec=60)
    record = _record(no_eligible_since=1000.0, last_error="bad spec")
    with mock.patch("server.dispatcher.base.time.time", return_value=1100.0):
        disp._grace_then_fail(
            "tsk-1", record, reason="no_eligible_worker", message="no worker"
        )
    assert disp.requeued == []
    assert len(disp.failed) == 1
    task_id, message, kwargs = disp.failed[0]
    assert task_id == "tsk-1"
    assert message == "bad spec"
    assert kwargs["payload"]["reason"] == "no_eligible_worker"
