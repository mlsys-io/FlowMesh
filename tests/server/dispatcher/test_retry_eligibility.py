"""Eligible-worker computation and the no-worker grace in the dispatcher."""

from typing import cast
from unittest import mock

from server.task.models import TaskRecord
from server.task.worker_pins import effective_selected_worker
from shared.tasks import TaskEnvelopeTemplate
from tests.server.dispatcher.helpers import make_capturing_dispatcher


def _record(
    selected_worker: list[str] | None = None,
    no_eligible_since: float | None = None,
    last_error: str | None = None,
    last_failed_worker: str | None = None,
    allow_worker_fallback: bool = False,
) -> TaskRecord:
    return TaskRecord(
        task_id="tsk-1",
        workflow_id="wfl-1",
        owner_id="owner",
        source="",
        task=TaskEnvelopeTemplate.model_validate(
            {
                "apiVersion": "flowmesh/v1",
                "kind": "EchoTask",
                "spec": {"taskType": "echo"},
            }
        ),
        selected_worker=selected_worker,
        allow_worker_fallback=allow_worker_fallback,
        no_eligible_since=no_eligible_since,
        last_error=last_error,
        failed_workers=[last_failed_worker] if last_failed_worker else [],
    )


def test_eligible_worker_ids_returns_satisfying_set() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1", "w-2", "w-3"])
    assert disp.eligible_worker_ids(_record(), None) == {"w-1", "w-2", "w-3"}


def test_eligible_worker_ids_intersects_selected_worker() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1", "w-2", "w-3"])
    record = _record(selected_worker=["w-2", "w-9"])
    assert disp.eligible_worker_ids(record, ["w-2", "w-9"]) == {"w-2"}


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


def test_an_opted_in_pin_to_unregistered_workers_is_ignored() -> None:
    record = _record(selected_worker=["wkr-8"], allow_worker_fallback=True)
    assert effective_selected_worker(record, lambda wid: False) is None
    assert record.selected_worker == ["wkr-8"]


def test_a_pin_with_a_present_worker_is_still_strict() -> None:
    record = _record(selected_worker=["w-2", "wkr-8"], allow_worker_fallback=True)
    assert effective_selected_worker(record, lambda wid: wid == "w-2") == [
        "w-2",
        "wkr-8",
    ]


def test_an_absent_pin_is_strict_without_fallback() -> None:
    record = _record(selected_worker=["wkr-8"])
    assert effective_selected_worker(record, lambda wid: False) == ["wkr-8"]


def test_no_pin_is_no_pin() -> None:
    assert effective_selected_worker(_record(), lambda wid: True) is None


def test_unavailable_pin_fails_with_its_own_reason() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1"], grace_sec=0)
    record = _record(selected_worker=["wkr-8"])
    cast(mock.Mock, disp._runtime).get_record.return_value = record
    disp._attempt_results = {}
    assert disp._dispatch_attempt("tsk-1") is False
    _, message, kwargs = disp.failed[0]
    assert kwargs["payload"]["reason"] == "selected_worker_unavailable"
    assert kwargs["payload"]["selected_worker"] == ["wkr-8"]
    assert "wkr-8" in message


def test_unpinned_task_keeps_the_hardware_reason() -> None:
    disp = make_capturing_dispatcher(grace_sec=0)
    cast(mock.Mock, disp._runtime).get_record.return_value = _record()
    disp._attempt_results = {}
    disp._dispatch_attempt("tsk-1")
    assert disp.failed[0][2]["payload"]["reason"] == "no_eligible_worker"
