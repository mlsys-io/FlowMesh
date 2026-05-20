"""Result envelope schema shared by server and worker."""

# This is necessary to allow for the recursive type hint of `children` in
# `BaseExecutorResult`.
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from shared.utils.atomic import atomic_write_text
from shared.utils.manifest import prepare_output_dir
from shared.utils.time import now_iso

from .artifact import ArtifactContext


class BaseExecutorResult(BaseModel):
    """Common shape for every executor's result payload.

    ``extra="allow"`` lets the server round-trip subclass payloads through
    this base class without losing executor-specific fields.
    """

    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    ok: bool = Field(default=True, description="Whether task execution succeeded.")
    children: dict[str, SerializeAsAny[BaseExecutorResult]] = Field(
        default_factory=dict,
        exclude_if=lambda v: not v,
        description="Per-child result payloads for task merging.",
    )
    artifacts_: ArtifactContext | None = Field(
        default=None,
        alias="_artifacts",
        description="Resolution context for relative artifact refs.",
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "artifacts_" in cls.__annotations__:
            raise TypeError(
                f"{cls.__name__} may not redefine the internal "
                "BaseExecutorResult.artifacts_ field"
            )


class ResultEnvelope(BaseModel):
    task_id: str = Field(description="Task identifier.")
    result: SerializeAsAny[BaseExecutorResult] = Field(
        description="Result payload data."
    )
    worker_id: str | None = Field(
        default=None, description="Worker identifier submitting the result."
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional result metadata."
    )
    received_at: str = Field(
        default_factory=now_iso, description="Result receipt timestamp."
    )


def _sanitize_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)


def result_file_path(base_dir: Path, task_id: str) -> Path:
    return base_dir / _sanitize_task_id(task_id) / "results.json"


def write_result(base_dir: Path, envelope: ResultEnvelope) -> Path:
    path = result_file_path(base_dir, envelope.task_id)
    prepare_output_dir(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, envelope.model_dump_json(indent=2))
    return path


def read_result(base_dir: Path, task_id: str) -> str:
    path = result_file_path(base_dir, task_id)
    return path.read_text(encoding="utf-8")
