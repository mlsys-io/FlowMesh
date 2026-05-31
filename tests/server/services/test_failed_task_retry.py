"""Retry-decision logic for failed tasks (taxonomy + attempt budget)."""

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
    assert failed_task_can_retry(None, True) is False


def test_non_retryable_failure_never_retries() -> None:
    # Controlled ExecutionError: deterministic, fails identically everywhere.
    assert failed_task_can_retry(_record(), False) is False


def test_retries_within_budget() -> None:
    assert failed_task_can_retry(_record(attempts=1), True) is True


def test_attempt_budget_caps_retries() -> None:
    assert failed_task_can_retry(_record(attempts=3, max_attempts=3), True) is False


def test_unspecified_retryable_is_treated_as_retryable() -> None:
    assert failed_task_can_retry(_record(), None) is True


def test_already_terminal_record_does_not_retry() -> None:
    # A dispatcher-originated terminal failure re-enters the handler; don't retry.
    assert failed_task_can_retry(_record(status=TaskStatus.FAILED), True) is False


def test_unbounded_max_attempts_always_within_budget() -> None:
    assert failed_task_can_retry(_record(max_attempts=-1, attempts=99), True) is True
