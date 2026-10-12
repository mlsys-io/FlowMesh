"""Eligible-worker computation and the no-worker grace in the dispatcher."""

from unittest import mock

from server.task.models import TaskRecord
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


def test_an_opted_in_pin_to_unregistered_workers_is_ignored() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1", "w-2"])
    record = _record(selected_worker=["wkr-8"], allow_worker_fallback=True)
    with mock.patch.object(disp._worker_registry, "worker_exists", return_value=False):
        assert disp.effective_selected_worker(record) is None
        assert disp.eligible_worker_ids(record) == {"w-1", "w-2"}
    assert record.selected_worker == ["wkr-8"]


def test_a_pin_with_a_registered_worker_is_still_strict() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1", "w-2"])
    record = _record(selected_worker=["w-2", "wkr-8"], allow_worker_fallback=True)
    with mock.patch.object(
        disp._worker_registry, "worker_exists", side_effect=lambda wid: wid == "w-2"
    ):
        assert disp.effective_selected_worker(record) == ["w-2", "wkr-8"]
        assert disp.eligible_worker_ids(record) == {"w-2"}


def test_an_unregistered_pin_is_strict_by_default() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1"])
    record = _record(selected_worker=["wkr-8"])
    with mock.patch.object(disp._worker_registry, "worker_exists", return_value=False):
        assert disp.effective_selected_worker(record) == ["wkr-8"]
        assert disp.eligible_worker_ids(record) == set()


def test_no_pin_is_no_pin() -> None:
    disp = make_capturing_dispatcher(satisfying_ids=["w-1"])
    assert disp.effective_selected_worker(_record()) is None
