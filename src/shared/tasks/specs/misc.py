from typing import Any, Literal, Self

from ...utils.pydantic_utils import copy_preserving_fields_set
from ...utils.redact import has_redacted_credential_fields, redact_credential_fields
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

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"api": redact_credential_fields(spec.api)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.api
        )


class ApiSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.API]
    api: dict[str, Any] | None = None
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"api": redact_credential_fields(spec.api)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.api
        )


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

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )


class DataProfilingSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_PROFILING]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )


class DataRetrievalSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )


class DataRetrievalSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )


class EmbeddingSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )

    def uses_gpu(self) -> bool:
        model = self.model
        if model is not None and model.vllm is not None:
            return True
        return self.model_uses_gpu()


class EmbeddingSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )

    def uses_gpu(self) -> bool:
        model = self.model
        if model is not None and model.vllm is not None:
            return True
        return self.model_uses_gpu()
