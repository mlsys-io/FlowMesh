"""Upstream-result staging for SSH sessions.

Resolving an ``inputs[]`` entry to a local directory is the same work for every
session backend: locate the upstream task's results on disk, or download the
result bundle when this worker never ran that task. How the staged directory is
then exposed to the session is backend-specific.
"""

import os
import shutil
import tempfile
from pathlib import Path

from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.result_delivery import (
    artifacts_ready,
    make_receipt,
    relocated_links,
    safe_relative,
    selection_roots,
    write_receipt,
)
from worker.utils.result_delivery import hydrate_result

from ..base_executor import ExecutionError
from .config import (
    DEFAULT_INPUTS_ROOT,
    ResolvedSSHInput,
    SSHConfig,
    normalize_mount_path,
)


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
        if not artifacts_ready(
            resolved.source_path,
            resolved.task_id,
            resolved.artifact_paths,
            resolved.generation,
            verify_content=False,
        ):
            hydrate_result(
                resolved.task_id,
                resolved.source_path.parent,
                resolved.artifact_paths,
                resolved.generation,
            )
        destination = staging_dir / resolved.task_id
        # Links are copied as links: an upstream session's output could
        # otherwise point the copy at the worker's own files.
        if resolved.artifact_paths is None:
            shutil.copytree(
                resolved.source_path, destination, symlinks=True, dirs_exist_ok=True
            )
            _relocate_links(resolved, destination)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            resolved.source_path / "results.json", destination / "results.json"
        )
        for name in selection_roots(resolved.source_path, resolved.artifact_paths):
            source = resolved.source_path / "artifacts" / safe_relative(name)
            target = destination / "artifacts" / safe_relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                if not target.is_symlink():
                    target.symlink_to(os.readlink(source))
            elif source.is_dir():
                shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copyfile(source, target)
        _relocate_links(resolved, destination)
        write_receipt(
            destination,
            make_receipt(destination, resolved.task_id, resolved.artifact_paths),
        )
    return staging_dir


def _relocate_links(resolved: ResolvedSSHInput, destination: Path) -> None:
    for name, link in relocated_links(
        resolved.source_path, resolved.artifact_paths
    ).items():
        target = destination / name
        target.unlink()
        target.symlink_to(link)
