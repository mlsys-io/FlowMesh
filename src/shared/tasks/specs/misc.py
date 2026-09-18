from typing import Any, Literal

from ..task_type import TaskType
from .common import (
    ModelSpecStrict,
    ModelSpecTemplate,
    TaskSpecStrictBase,
    TaskSpecTemplateBase,
)


class ApiSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.API]
    api: dict[str, Any] | None = None
    data: dict[str, Any] | None = None


class ApiSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.API]
    api: dict[str, Any] | None = None
    data: dict[str, Any] | None = None


class EchoSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.ECHO]
    data: dict[str, Any] | None = None


class EchoSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.ECHO]
    data: dict[str, Any] | None = None


class AgentSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.AGENT]

    configName: str | None = None
    task: str | None = None
    agent: dict[str, Any] | None = None
    data: dict[str, Any] | None = None


class AgentSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.AGENT]

    configName: str | None = None
    task: str | None = None
    agent: dict[str, Any] | None = None
    data: dict[str, Any] | None = None


class DataProfilingSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.DATA_PROFILING]
    data: dict[str, Any] | None = None


class DataProfilingSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_PROFILING]
    data: dict[str, Any] | None = None


class DataRetrievalSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None


class DataRetrievalSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None


class EmbeddingSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None


class EmbeddingSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None
