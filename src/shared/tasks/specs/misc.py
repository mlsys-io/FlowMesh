from typing import Any, Literal, Self

from ...utils.redact import contains_redacted_credential, redact_value
from ..task_type import TaskType
from .common import (
    ModelSpecStrict,
    ModelSpecTemplate,
    TaskSpecStrictBase,
    TaskSpecTemplateBase,
)


def redact_api(api: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a copy of an API spec with credential values replaced."""
    if not isinstance(api, dict):
        return api
    return redact_value(api)


class ApiSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.API]
    api: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        return self.model_copy(update={"api": redact_api(self.api)})

    def has_redacted_credentials(self) -> bool:
        return contains_redacted_credential({"api": self.api})


class ApiSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.API]
    api: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        return self.model_copy(update={"api": redact_api(self.api)})

    def has_redacted_credentials(self) -> bool:
        return contains_redacted_credential({"api": self.api})


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

    def redact_credentials(self) -> Self:
        return self.model_copy(update={"data": redact_value(self.data)})

    def has_redacted_credentials(self) -> bool:
        return contains_redacted_credential({"data": self.data})


class DataRetrievalSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        return self.model_copy(update={"data": redact_value(self.data)})

    def has_redacted_credentials(self) -> bool:
        return contains_redacted_credential({"data": self.data})


class EmbeddingSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None


class EmbeddingSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None
