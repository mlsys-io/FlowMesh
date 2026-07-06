"""Undeliverable-dispatch grace behavior.

When a worker is selected but the task cannot be delivered to it (its node has
no live dispatch subscriber, or publish raises), the dispatcher must treat this
as an infrastructure fault: retry without spending the task's ``max_attempts``
budget, then fail with a descriptive error after the no-worker grace.
"""

import time
from types import SimpleNamespace

from .helpers import make_capturing_dispatcher


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        no_dispatch_since=None,
        last_error=None,
        last_failed_worker=None,
    )


def test_undeliverable_within_grace_requeues_without_retry() -> None:
    disp = make_capturing_dispatcher(grace_sec=60)
    disp._runtime.release_merge = lambda task_id: None
    record = _record()

    result = disp._grace_then_fail_undeliverable(
        "tsk-1", record, reason="no_dispatch_subscriber", message="orphaned"
    )

    assert result is False
    assert not disp.failed
    assert len(disp.requeued) == 1
    _, kwargs = disp.requeued[0]
    # Infra fault must NOT consume the execution retry budget.
    assert kwargs["count_retry"] is False
    assert kwargs["reason"] == "no_dispatch_subscriber"
    assert record.no_dispatch_since is not None
    assert record.last_error == "orphaned"


def test_undeliverable_after_grace_fails_with_message() -> None:
    disp = make_capturing_dispatcher(grace_sec=60)
    disp._runtime.release_merge = lambda task_id: None
    record = _record()
    # Pretend the condition has persisted longer than the grace window.
    record.no_dispatch_since = time.time() - 120

    result = disp._grace_then_fail_undeliverable(
        "tsk-1",
        record,
        reason="no_dispatch_subscriber",
        message="orphaned worker w1 on node n1",
        extra_payload={"worker_id": "w1", "node_id": "n1"},
    )

    assert result is False
    assert not disp.requeued
    assert len(disp.failed) == 1
    task_id, error_message, kwargs = disp.failed[0]
    assert task_id == "tsk-1"
    assert error_message == "orphaned worker w1 on node n1"
    payload = kwargs["payload"]
    assert payload["reason"] == "no_dispatch_subscriber"
    assert payload["worker_id"] == "w1"
    assert payload["node_id"] == "n1"
