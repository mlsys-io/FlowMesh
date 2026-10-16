"""Worker capabilities tolerate task types newer than this SDK."""

from flowmesh.models.common import TaskType
from flowmesh.models.workers import WorkerCapabilities


def test_unknown_task_type_is_dropped_not_rejected() -> None:
    caps = WorkerCapabilities.model_validate(
        {"supported_task_types": ["echo", "python", "omni_text2future"]}
    )
    assert caps.supported_task_types == frozenset({TaskType.ECHO, TaskType.PYTHON})


def test_known_task_types_are_kept() -> None:
    caps = WorkerCapabilities.model_validate(
        {"supported_task_types": [t.value for t in TaskType]}
    )
    assert caps.supported_task_types == frozenset(TaskType)


def test_missing_task_types_default_to_empty() -> None:
    assert WorkerCapabilities.model_validate({}).supported_task_types == frozenset()
