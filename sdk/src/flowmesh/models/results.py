"""Result-related models."""

from typing import Any

from pydantic import BaseModel, Field


class PathResponse(BaseModel):
    ok: bool
    path: str


class ResultEnvelope(BaseModel):
    """Canonical on-disk shape of ``results.json`` (mirrors the server)."""

    task_id: str
    result: dict[str, Any]
    worker_id: str | None = None
    metadata: dict[str, Any] | None = None
    received_at: str | None = Field(default=None)
