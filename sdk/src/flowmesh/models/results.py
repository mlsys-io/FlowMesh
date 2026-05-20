"""Result-related models."""

# This is necessary to allow for the recursive type hint of `children` in
# `BaseExecutorResult`.
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from .artifacts import ArtifactContext


class PathResponse(BaseModel):
    ok: bool
    path: str


class BaseExecutorResult(BaseModel):
    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    ok: bool = True
    children: dict[str, SerializeAsAny[BaseExecutorResult]] = Field(
        default_factory=dict, exclude_if=lambda v: not v
    )
    artifacts_: ArtifactContext | None = Field(default=None, alias="_artifacts")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "artifacts_" in cls.__annotations__:
            raise TypeError(
                f"{cls.__name__} may not redefine the internal "
                "BaseExecutorResult.artifacts_ field"
            )


class ResultEnvelope(BaseModel):
    """Canonical on-disk shape of ``results.json`` (mirrors the server)."""

    task_id: str
    result: SerializeAsAny[BaseExecutorResult]
    worker_id: str | None = None
    metadata: dict[str, Any] | None = None
    received_at: str | None = Field(default=None)
