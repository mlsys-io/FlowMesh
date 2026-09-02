"""On-disk ``results.json`` read/write helpers."""

from pathlib import Path

from shared.utils.atomic import atomic_write_text
from shared.utils.manifest import prepare_output_dir

from .catalog import ResultEnvelope


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
