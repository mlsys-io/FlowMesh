from typing import Annotated, Literal

from pydantic import Field

from ..task_type import TaskType
from .common import ModelSpecStrict, ModelSpecTemplate


class ServeSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.SERVE]
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    readinessTimeoutSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["direct", "forward"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None


class ServeSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.SERVE]
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    readinessTimeoutSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["direct", "forward"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None
