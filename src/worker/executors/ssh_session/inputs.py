"""Upstream-result staging for SSH sessions.

Resolving an ``inputs[]`` entry to a local directory is the same work for every
session backend: locate the upstream task's results on disk, or download the
result bundle when this worker never ran that task. How the staged directory is
then exposed to the session is backend-specific.
"""

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests

from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.http import auth_headers

from ..base_executor import ExecutionError
from .config import (
    DEFAULT_INPUTS_ROOT,
    ResolvedSSHInput,
    SSHConfig,
    normalize_mount_path,
)

RESULT_BUNDLE_TIMEOUT_SEC = 300.0


def resolve_inputs(
    task: WorkerTaskMessage, cfg: SSHConfig, results_root: Path
) -> list[ResolvedSSHInput]:
    if not cfg.inputs:
        return []

    resolved: list[ResolvedSSHInput] = []
    upstream_task_ids = task.upstream_task_ids or {}
    for entry in cfg.inputs:
        stage = entry.stage.strip()
        if not stage:
            raise ExecutionError("SSH input stage names must be non-empty")
        task_id = upstream_task_ids.get(stage)
        if not task_id:
            raise ExecutionError(
                f"Missing resolved upstream task ID for SSH input stage '{stage}'"
            )
        resolved.append(
            ResolvedSSHInput(
                stage=stage,
                task_id=task_id,
                source_path=results_root / task_id,
                mount_path=normalize_mount_path(
                    entry.mountPath or f"{DEFAULT_INPUTS_ROOT}/{stage}",
                    field_name=f"inputs[{stage}].mountPath",
                ),
            )
        )
    return resolved


def stage_inputs_locally(
    resolved_inputs: list[ResolvedSSHInput], session_id: str
) -> Path:
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f"flowmesh-ssh-inputs-{session_id[:8]}-")
    )
    for resolved in resolved_inputs:
        destination = staging_dir / resolved.task_id
        if resolved.source_path.exists():
            shutil.copytree(resolved.source_path, destination, dirs_exist_ok=True)
            continue
        download_result_bundle(resolved.task_id, staging_dir)
        if not destination.exists():
            raise ExecutionError(
                "Downloaded SSH input bundle did not create expected directory "
                f"{destination} for upstream task {resolved.task_id}"
            )
    return staging_dir


def result_bundle_url(task_id: str) -> str:
    base_url = os.getenv("FLOWMESH_BASE_URL", "").strip()
    if not base_url:
        raise ExecutionError(
            "SSH input result hydration requires FLOWMESH_BASE_URL when "
            "upstream results are not available locally"
        )
    return (
        f"{base_url.rstrip('/')}/api/v1/results/{task_id}/bundle"
        "?include=results&include=artifacts"
    )


def download_result_bundle(task_id: str, destination_dir: Path) -> None:
    tmp_fd, tmp_str = tempfile.mkstemp(prefix="ssh_bundle_", suffix=".tar.gz")
    os.close(tmp_fd)
    tmp_path = Path(tmp_str)
    try:
        with requests.get(
            result_bundle_url(task_id),
            headers=auth_headers(),
            stream=True,
            timeout=RESULT_BUNDLE_TIMEOUT_SEC,
        ) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as sink:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        sink.write(chunk)
        extract_result_bundle(tmp_path, destination_dir)
    except requests.RequestException as exc:
        raise ExecutionError(
            f"Failed to download SSH input result bundle for {task_id}: {exc}",
            retryable=True,
        ) from exc
    except tarfile.TarError as exc:
        raise ExecutionError(
            f"Failed to unpack SSH input result bundle for {task_id}: {exc}"
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_result_bundle(bundle_path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest_root = destination_dir.resolve()
    with tarfile.open(bundle_path, mode="r:*") as archive:
        for member in archive:
            member_path = (dest_root / member.name).resolve()
            try:
                member_path.relative_to(dest_root)
            except ValueError as exc:
                raise ExecutionError(
                    f"Unsafe path in SSH input result bundle: {member.name}"
                ) from exc
            archive.extract(member, dest_root, filter="data")
