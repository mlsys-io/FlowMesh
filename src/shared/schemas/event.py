from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.utils.json import normalize_numbers
from shared.utils.time import now_iso

from .worker import WorkerStatus


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(
        ..., description="Event type, expected to be an uppercase enum value."
    )
    ts: str = Field(default_factory=now_iso, description="Event timestamp (ISO8601).")

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        value = (value or "").strip().upper()
        if not value:
            raise ValueError("event type must not be empty")
        return value


class TaskEvent(BaseEvent):
    worker_id: str | None = Field(
        default=None, description="Associated worker identifier."
    )
    task_id: str = Field(..., description="Associated task identifier.")
    status: str | None = Field(default=None, description="Task status.")
    error: str | None = Field(default=None, description="Error message if any.")
    retryable: bool | None = Field(
        default=None,
        description=(
            "Whether a failure may be retried on another worker. None leaves the "
            "decision to the server, which treats it as retryable."
        ),
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Additional event payload."
    )

    @field_validator("task_id")
    @classmethod
    def _trim_task_id(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("task_id must not be empty")
        return value


class WorkerEvent(BaseEvent):
    worker_id: str = Field(..., description="Associated worker identifier.")
    status: WorkerStatus | None = Field(
        default=None, description="Worker status (IDLE/RUNNING/etc)."
    )
    tags: list[str] | None = Field(default=None, description="Worker tags.")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Metrics reported in heartbeat."
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Additional context."
    )
    actor: dict[str, Any] | None = Field(default=None, description="Actor information")


class NodeEvent(BaseEvent):
    node_id: str = Field(..., description="Associated node identifier.")
    tags: list[str] | None = Field(default=None, description="Node tags.")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Additional context."
    )
    actor: dict[str, Any] | None = Field(default=None, description="Actor information")


Event = TaskEvent | WorkerEvent | NodeEvent


def parse_event(data: dict[str, Any]) -> Event:
    data = normalize_numbers(data)
    event_type = str(data.get("type", "")).upper()
    if event_type.startswith("TASK_"):
        return TaskEvent.model_validate(data)
    if event_type.startswith("SV_"):
        return NodeEvent.model_validate(data)
    return WorkerEvent.model_validate(data)


def serialize_event(event: Event) -> dict[str, Any]:
    return event.model_dump(mode="python")


__all__ = [
    "BaseEvent",
    "Event",
    "NodeEvent",
    "TaskEvent",
    "WorkerEvent",
    "parse_event",
    "serialize_event",
]
