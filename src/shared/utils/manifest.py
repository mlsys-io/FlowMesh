"""Manifest helpers used by the worker pipeline when persisting outputs."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text
from .time import now_iso

MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.json"
LOGS_DIR = "logs"
ARTIFACTS_DIR = "artifacts"
SCRATCH_DIR = "scratch"

_SHARED_DIR_MODE = 0o0777


def prepare_output_dir(base_dir: Path) -> None:
    """Ensure the base directory and standard sub-directories exist."""
    for d in (base_dir, base_dir / LOGS_DIR, base_dir / ARTIFACTS_DIR):
        if not d.exists():
            d.mkdir(parents=True)
            d.chmod(_SHARED_DIR_MODE)


def scratch_dir(base_dir: Path) -> Path:
    """Return `out_dir/scratch/`, creating it if needed."""
    path = base_dir / SCRATCH_DIR
    if not path.exists():
        path.mkdir(parents=True)
        path.chmod(_SHARED_DIR_MODE)
    return path


def sync_manifest(
    base_dir: Path, task_id: str, expected: Iterable[str]
) -> dict[str, Any]:
    """
    Build a manifest by reconciling expected versus actual files.
    """
    prepare_output_dir(base_dir)
    expected_set = {_normalize_artifact_name(item) for item in expected or [] if item}
    expected_set.update({RESULTS_NAME, LOGS_DIR, ARTIFACTS_DIR})

    entries: list[dict[str, Any]] = []
    added: set[str] = set()

    for name in sorted(expected_set):
        rel_path = Path(name)
        entry = _describe_path(base_dir, rel_path, required=True)
        entries.append(entry)
        added.add(rel_path.as_posix())

    # Capture additional files/directories that exist but were not declared.
    for item in base_dir.iterdir():
        key = item.relative_to(base_dir).as_posix()
        if key in added or item.name == MANIFEST_NAME:
            continue
        entry = _describe_path(base_dir, item.relative_to(base_dir), required=False)
        entries.append(entry)

    manifest = {
        "task_id": task_id,
        "generated_at": now_iso(),
        "entries": entries,
    }
    atomic_write_text(
        base_dir / MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    return manifest


# -------------------------
# Helpers
# -------------------------


def _infer_type(rel_path: Path) -> str:
    normalized = rel_path.as_posix()
    if normalized == RESULTS_NAME:
        return "result"
    if normalized.startswith(f"{LOGS_DIR}/") or normalized == LOGS_DIR:
        return "logs"
    if normalized.startswith(f"{ARTIFACTS_DIR}/") or normalized == ARTIFACTS_DIR:
        return "artifact"
    if normalized.startswith(f"{SCRATCH_DIR}/") or normalized == SCRATCH_DIR:
        return "scratch"
    if rel_path.suffix:
        return "artifact"
    return "directory"


def _describe_path(base_dir: Path, rel_path: Path, *, required: bool) -> dict[str, Any]:
    target = base_dir / rel_path
    entry_type = _infer_type(rel_path)
    entry: dict[str, Any] = {
        "name": rel_path.as_posix(),
        "path": rel_path.as_posix(),
        "type": entry_type,
        "required": required,
    }

    if target.exists():
        entry["status"] = "present"
        entry["updated_at"] = now_iso()
        if target.is_file():
            stat = target.stat()
            entry["size"] = stat.st_size
            entry["sha256"] = _sha256_file(target)
        else:
            size, count = _directory_stats(target)
            entry["size"] = size
            entry["file_count"] = count
    else:
        entry["status"] = "missing"
    return entry


def _normalize_artifact_name(name: str) -> str:
    value = name.strip()
    if value.endswith("/"):
        value = value.rstrip("/")
    if value.startswith("./"):
        value = value[2:]
    return value or name


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _directory_stats(path: Path) -> tuple[int, int]:
    total_size = 0
    file_count = 0
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            total_size += stat.st_size
            file_count += 1
    return total_size, file_count
