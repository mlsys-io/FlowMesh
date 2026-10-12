from collections.abc import Callable

from shared.tasks import TaskEnvelopeTemplate

from .models import TaskRecord


class WorkerPinMismatch(ValueError):
    """Raised when a selected worker is outside the task's effective pin."""


def allows_worker_fallback(task: TaskEnvelopeTemplate) -> bool:
    annotations = task.metadata.annotations if task.metadata else None
    hint = annotations.schedule_hint if annotations else None
    return hint is not None and hint.allow_worker_fallback


def effective_selected_worker(
    record: TaskRecord, worker_exists: Callable[[str], bool]
) -> list[str] | None:
    """Resolve a worker pin, relaxing opted-in pins with no registered IDs."""
    pins = record.selected_worker
    if not pins:
        return None
    if not allows_worker_fallback(record.task) or any(
        worker_exists(wid) for wid in pins
    ):
        return list(pins)
    return None
