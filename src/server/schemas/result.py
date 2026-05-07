import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from shared.utils.atomic import atomic_write_text
from shared.utils.manifest import prepare_output_dir

from ..utils.time import now_iso


class ResultPayload(BaseModel):
    task_id: str = Field(description="Task identifier.")
    result: dict[str, Any] = Field(description="Result payload data.")
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


def write_result(base_dir: Path, task_id: str, content: dict[str, Any]) -> Path:
    path = result_file_path(base_dir, task_id)
    prepare_output_dir(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(content, ensure_ascii=False, indent=2))
    return path


def read_result(base_dir: Path, task_id: str) -> str:
    path = result_file_path(base_dir, task_id)
    return path.read_text(encoding="utf-8")
