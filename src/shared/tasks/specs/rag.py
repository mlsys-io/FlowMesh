from typing import Any, Literal, Self

from ...utils.redact import has_redacted_credential_fields, redact_credential_fields
from ..task_type import TaskType
from .common import TaskSpecStrictBase, TaskSpecTemplateBase, copy_preserving_fields_set


class RagSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.RAG]

    qdrant: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None
    search: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    query: str | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec,
            {
                "qdrant": redact_credential_fields(spec.qdrant),
                "embedding": redact_credential_fields(spec.embedding),
                "search": redact_credential_fields(spec.search),
                "data": redact_credential_fields(spec.data),
            },
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {
                "qdrant": self.qdrant,
                "embedding": self.embedding,
                "search": self.search,
                "data": self.data,
            }
        )


class RagSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.RAG]

    qdrant: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None
    search: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    query: str | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec,
            {
                "qdrant": redact_credential_fields(spec.qdrant),
                "embedding": redact_credential_fields(spec.embedding),
                "search": redact_credential_fields(spec.search),
                "data": redact_credential_fields(spec.data),
            },
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {
                "qdrant": self.qdrant,
                "embedding": self.embedding,
                "search": self.search,
                "data": self.data,
            }
        )
