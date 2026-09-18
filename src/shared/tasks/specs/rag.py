from typing import Any, Literal, Self

from ...utils.redact import contains_redacted_credential, redact_value
from ..task_type import TaskType
from .common import TaskSpecStrictBase, TaskSpecTemplateBase


class RagSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.RAG]

    qdrant: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None
    search: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    query: str | None = None

    def redact_credentials(self) -> Self:
        return self.model_copy(
            update={
                "qdrant": redact_value(self.qdrant),
                "embedding": redact_value(self.embedding),
                "search": redact_value(self.search),
                "data": redact_value(self.data),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return contains_redacted_credential(
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
        return self.model_copy(
            update={
                "qdrant": redact_value(self.qdrant),
                "embedding": redact_value(self.embedding),
                "search": redact_value(self.search),
                "data": redact_value(self.data),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return contains_redacted_credential(
            {
                "qdrant": self.qdrant,
                "embedding": self.embedding,
                "search": self.search,
                "data": self.data,
            }
        )
