from typing import Annotated, Any, Literal

from pydantic import Field

from ..task_type import TaskType
from .common import TaskSpecStrictBase, TaskSpecTemplateBase


class ServeSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.SERVE]
    model: str
    vllmArgs: dict[str, Any] = Field(default_factory=dict)
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    readinessTimeoutSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["direct", "forward"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None


class ServeSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.SERVE]
    model: str
    vllmArgs: dict[str, Any] = Field(default_factory=dict)
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    readinessTimeoutSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["direct", "forward"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None
