"""Retry-decision logic for failed tasks (taxonomy + eligible-worker budget)."""

from types import SimpleNamespace
from typing import Any, cast

from server.services.monitoring import failed_task_can_retry
from server.task.models import TaskRecord, TaskStatus


def _record(
    status: str = TaskStatus.DISPATCHED,
    attempts: int = 0,
    max_attempts: int = 3,
) -> TaskRecord:
    rec = SimpleNamespace(status=status, attempts=attempts, max_attempts=max_attempts)
    return cast(TaskRecord, cast(Any, rec))


def test_no_record_is_not_retryable() -> None:
    assert failed_task_can_retry(None, True, 5) is False


def test_non_retryable_failure_never_retries() -> None:
    # Controlled ExecutionError: deterministic, fails identically everywhere.
    assert failed_task_can_retry(_record(), False, 5) is False


def test_retries_while_untried_eligible_workers_remain() -> None:
    assert failed_task_can_retry(_record(attempts=1), True, 2) is True


def test_no_untried_eligible_worker_stops_retry() -> None:
    # Every eligible worker has already failed this task.
    assert failed_task_can_retry(_record(attempts=1), True, 0) is False


def test_attempt_budget_caps_retries() -> None:
    assert failed_task_can_retry(_record(attempts=3, max_attempts=3), True, 5) is False


def test_unspecified_retryable_is_treated_as_retryable() -> None:
    assert failed_task_can_retry(_record(), None, 1) is True


def test_already_terminal_record_does_not_retry() -> None:
    # A dispatcher-originated terminal failure re-enters the handler; don't retry.
    assert failed_task_can_retry(_record(status=TaskStatus.FAILED), True, 5) is False


def test_unbounded_max_attempts_still_needs_untried_worker() -> None:
    assert failed_task_can_retry(_record(max_attempts=-1), True, 1) is True
    assert failed_task_can_retry(_record(max_attempts=-1), True, 0) is False
