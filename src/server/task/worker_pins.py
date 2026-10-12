from collections.abc import Callable

from shared.tasks import TaskEnvelopeTemplate

from .models import TaskRecord


class WorkerPinMismatch(ValueError):
    """Raised when a selected worker is outside the task's effective pin."""


def resolve_worker_fallback(task: TaskEnvelopeTemplate, default: bool) -> bool:
    """Return the task's ``allow_worker_fallback`` hint, or ``default`` when unset."""
    annotations = task.metadata.annotations if task.metadata else None
    hint = annotations.schedule_hint if annotations else None
    if hint is None or hint.allow_worker_fallback is None:
        return default
    return hint.allow_worker_fallback


def effective_selected_worker(
    record: TaskRecord, is_present: Callable[[str], bool]
) -> list[str] | None:
    """Resolve a worker pin, relaxing a fallback-enabled pin once no ID is present."""
    pins = record.selected_worker
    if not pins:
        return None
    if not record.allow_worker_fallback or any(is_present(wid) for wid in pins):
        return list(pins)
    return None
