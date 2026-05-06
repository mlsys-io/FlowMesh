import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .time import now_iso

MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.json"
LOGS_DIR = "logs"
ARTIFACTS_DIR = "artifacts"

# Single-node deployments share the results volume between the server (root)
# and supervisor-spawned workers (appuser); both call sync_manifest, so each
# directory and the manifest itself must be writable from either UID.
_SHARED_DIR_MODE = 0o0777
_SHARED_FILE_MODE = 0o0666


def prepare_output_dir(base_dir: Path) -> None:
    for d in (base_dir, base_dir / LOGS_DIR, base_dir / ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
        _shared_chmod(d, _SHARED_DIR_MODE)


def sync_manifest(
    base_dir: Path, task_id: str, expected: Iterable[str]
) -> dict[str, Any]:
    prepare_output_dir(base_dir)
    expected_set = {_normalize_artifact_name(item) for item in expected or [] if item}
    expected_set.update({RESULTS_NAME, LOGS_DIR, ARTIFACTS_DIR})

    entries: list[dict[str, Any]] = []
    added: set[str] = set()

    for name in sorted(expected_set):
        rel_path = Path(name)
        entry = _describe_path(base_dir, rel_path, required=True)
        entries.append(entry)
        added.add(_path_key(rel_path))

    for item in base_dir.iterdir():
        relative = item.relative_to(base_dir)
        key = _path_key(relative)
        if key in added or item.name == MANIFEST_NAME:
            continue
        entry = _describe_path(base_dir, relative, required=False)
        entries.append(entry)

    manifest = {
        "task_id": task_id,
        "generated_at": now_iso(),
        "entries": entries,
    }
    _atomic_write_json(base_dir / MANIFEST_NAME, manifest)
    return manifest


def _shared_chmod(path: Path, mode: int) -> None:
    """Best-effort chmod tolerant of cross-UID ownership."""
    try:
        path.chmod(mode)
    except PermissionError:
        pass


def _atomic_write_json(target: Path, payload: Any) -> None:
    """Replace ``target`` with a JSON dump of ``payload`` atomically.

    Uses tempfile + os.replace so the writer only needs write permission on
    the parent directory, not on any pre-existing file (which may be owned
    by a different UID under a shared results volume).
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        _shared_chmod(tmp_path, _SHARED_FILE_MODE)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _path_key(path: Path) -> str:
    return path.as_posix()


def _infer_type(rel_path: Path) -> str:
    normalized = rel_path.as_posix()
    if normalized == RESULTS_NAME:
        return "result"
    if normalized.startswith(f"{LOGS_DIR}/") or normalized == LOGS_DIR:
        return "logs"
    if normalized.startswith(f"{ARTIFACTS_DIR}/") or normalized == ARTIFACTS_DIR:
        return "artifact"
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
