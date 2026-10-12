from typing import Any

from pydantic import Field

from .._base import StrictBaseModel


class SelectedWorkerHint(StrictBaseModel):
    """
    Mirrors the structure supported by `src/server/task/parser.py` under:
      metadata.annotations.schedule_hint.selected_worker
    """

    global_: list[str] | None = Field(default=None, alias="global")
    selected: dict[str, str | list[str]] | None = None


class ScheduleHint(StrictBaseModel):
    node_execution_order: list[str] | list[list[str]] | None = None
    node_schedule_in_epoch_order: bool | None = True
    selected_worker: str | list[str] | SelectedWorkerHint | None = None
    allow_worker_fallback: bool | None = Field(default=None, strict=True)


class TaskAnnotations(StrictBaseModel):
    schedule_hint: ScheduleHint | None = None
    description: str | None = None
    custom: dict[str, Any] | None = None


class TaskMetadata(StrictBaseModel):
    name: str | None = None
    owner: str | None = None
    annotations: TaskAnnotations | None = None
