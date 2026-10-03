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
from urllib.parse import urlencode

import requests

from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.http import auth_headers
from shared.utils.result_delivery import (
    artifacts_ready,
    make_receipt,
    safe_relative,
    write_receipt,
)

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
                generation=task.upstream_result_generations.get(task_id),
                source_path=results_root / task_id,
                mount_path=normalize_mount_path(
                    entry.mountPath or f"{DEFAULT_INPUTS_ROOT}/{stage}",
                    field_name=f"inputs[{stage}].mountPath",
                ),
            )
        )
    by_task: dict[str, list[str]] = {}
    generations: dict[str, str | None] = {}
    for artifact_input in task.artifact_inputs.get(task.task_id, []):
        by_task.setdefault(artifact_input.task_id, []).append(artifact_input.path)
        generations[artifact_input.task_id] = artifact_input.generation
    for task_id, paths in by_task.items():
        resolved.append(
            ResolvedSSHInput(
                stage=task_id,
                task_id=task_id,
                source_path=results_root / task_id,
                mount_path=f"/mnt/flowmesh/references/{task_id}",
                generation=generations[task_id],
                artifact_paths=sorted(set(paths)),
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
        if artifacts_ready(
            resolved.source_path,
            resolved.task_id,
            resolved.artifact_paths,
            resolved.generation,
        ):
            # Links are copied as links: an upstream session's output could
            # otherwise point the copy at the worker's own files.
            if resolved.artifact_paths is None:
                shutil.copytree(
                    resolved.source_path, destination, symlinks=True, dirs_exist_ok=True
                )
            else:
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    resolved.source_path / "results.json", destination / "results.json"
                )
                for name in resolved.artifact_paths:
                    source = resolved.source_path / "artifacts" / safe_relative(name)
                    target = destination / "artifacts" / safe_relative(name)
                    if source.is_dir():
                        shutil.copytree(source, target, dirs_exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
                write_receipt(
                    destination,
                    make_receipt(
                        destination, resolved.task_id, resolved.artifact_paths
                    ),
                )
            continue
        download_result_bundle(
            resolved.task_id, staging_dir, resolved.artifact_paths, resolved.generation
        )
        if not artifacts_ready(
            destination, resolved.task_id, resolved.artifact_paths, resolved.generation
        ):
            raise ExecutionError(
                f"Incomplete SSH input result bundle for {resolved.task_id}"
            )
        if not destination.exists():
            raise ExecutionError(
                "Downloaded SSH input bundle did not create expected directory "
                f"{destination} for upstream task {resolved.task_id}"
            )
    return staging_dir


def result_bundle_url(
    task_id: str, paths: list[str] | None = None, generation: str | None = None
) -> str:
    base_url = os.getenv("FLOWMESH_BASE_URL", "").strip()
    if not base_url:
        raise ExecutionError(
            "SSH input result hydration requires FLOWMESH_BASE_URL when "
            "upstream results are not available locally"
        )
    query = [("include", "results"), ("include", "artifacts")]
    query.extend(("artifact_path", path) for path in paths or [])
    if generation:
        query.append(("generation", generation))
    return f"{base_url.rstrip('/')}/api/v1/results/{task_id}/bundle?{urlencode(query)}"


def download_result_bundle(
    task_id: str,
    destination_dir: Path,
    paths: list[str] | None = None,
    generation: str | None = None,
) -> None:
    tmp_fd, tmp_str = tempfile.mkstemp(prefix="ssh_bundle_", suffix=".tar.gz")
    os.close(tmp_fd)
    tmp_path = Path(tmp_str)
    try:
        with requests.get(
            result_bundle_url(task_id, paths, generation),
            headers=auth_headers(),
            stream=True,
            timeout=RESULT_BUNDLE_TIMEOUT_SEC,
        ) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as sink:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        sink.write(chunk)
        with tempfile.TemporaryDirectory(prefix="ssh-bundle-stage-") as temporary:
            stage_root = Path(temporary)
            extract_result_bundle(tmp_path, stage_root)
            staged = stage_root / task_id
            if not artifacts_ready(staged, task_id, paths, generation):
                raise ExecutionError(
                    f"Incomplete SSH input result bundle for {task_id}"
                )
            shutil.copytree(staged, destination_dir / task_id, dirs_exist_ok=True)
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
