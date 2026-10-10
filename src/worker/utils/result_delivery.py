import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import requests
from pydantic import BaseModel

from shared.schemas.result import ResultEnvelope
from shared.schemas.result_delivery import ArtifactInput, ResultDeliveryRequest
from shared.tasks.placeholders import placeholder_fields
from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.http import auth_headers
from shared.utils.parsing import parse_float_env
from shared.utils.result_delivery import (
    artifact_path,
    artifacts_ready,
    commit_delivery,
    create_delivery_bundle,
    extract_delivery_bundle,
    result_field,
    result_generation,
    safe_relative,
)

from ..executors.base_executor import ExecutionError
from .upload_retry import send_with_retries


def _transfer_timeout() -> float:
    return parse_float_env("WORKER_RESULT_TRANSFER_TIMEOUT_SEC", 1800)


def publish_result(
    base_dir: Path, task_id: str, request: ResultDeliveryRequest, logger: logging.Logger
) -> int:
    """Publish the ``request`` selection of the result in ``base_dir`` to the server.

    Best-effort: a failure is logged and leaves the result local. Returns the
    number of bytes uploaded, which is 0 when nothing was sent.
    """
    base_url = os.getenv("FLOWMESH_BASE_URL", "").strip()
    if not base_url:
        logger.warning(
            "Task %s system result delivery requires FLOWMESH_BASE_URL", task_id
        )
        return 0
    bundle: Path | None = None
    try:
        envelope = ResultEnvelope.from_file(base_dir / "results.json")
        paths = (
            None
            if request.all_artifacts
            else sorted(
                {
                    path
                    for selector in request.artifact_fields
                    if (path := artifact_path(result_field(envelope.result, selector)))
                    is not None
                }
            )
        )
        with httpx.Client(timeout=_transfer_timeout()) as client:
            if _server_holds(
                client, base_url, task_id, paths, result_generation(base_dir)
            ):
                # The server shares this worker's results volume, or already
                # received this snapshot; uploading it again would only copy
                # the same files over themselves.
                logger.debug("Task %s result already held by the server", task_id)
                return 0
            bundle = create_delivery_bundle(
                base_dir, task_id, paths, include_traces=request.all_artifacts
            )
            size = bundle.stat().st_size
            delivery_bundle = bundle

            def send() -> httpx.Response:
                with delivery_bundle.open("rb") as source:
                    return client.post(
                        f"{base_url.rstrip('/')}/api/v1/results/{task_id}/delivery",
                        files={"file": ("delivery.tar", source, "application/x-tar")},
                        headers=auth_headers(),
                    )

            response = send_with_retries(
                send, what=f"Task {task_id} system result delivery", logger=logger
            )
            response.raise_for_status()
        return size
    except Exception as exc:
        logger.warning(
            "Task %s system result delivery failed: %s; result remains local",
            task_id,
            exc,
        )
        return 0
    finally:
        if bundle is not None:
            bundle.unlink(missing_ok=True)


def _server_holds(
    client: httpx.Client,
    base_url: str,
    task_id: str,
    paths: list[str] | None,
    generation: str,
) -> bool:
    """Whether the server already holds this exact snapshot selection.

    Any reply but 204, including one from a server without this check, means the
    snapshot is not held. A server that cannot be reached raises, so nothing is
    packed for an upload that would fail the same way.
    """
    query: list[tuple[str, str | int | float | bool | None]] = [
        ("generation", generation)
    ]
    if paths is None:
        query.append(("all_artifacts", "true"))
    else:
        query.extend(("artifact_path", path) for path in paths)
    response = client.get(
        f"{base_url.rstrip('/')}/api/v1/results/{task_id}/delivery",
        params=query,
        headers=auth_headers(),
    )
    return response.status_code == 204


def hydrate_result(
    task_id: str,
    destination_dir: Path,
    paths: list[str] | None = None,
    generation: str | None = None,
) -> None:
    """Make an upstream result available under ``destination_dir``.

    Downloads the result with the ``paths`` artifact selection (all artifacts when
    ``None``) from the server unless a complete local copy of that ``generation``
    is already there. Raises ``ExecutionError``, retryable when the transfer fails.
    """
    base_dir = destination_dir / task_id
    if artifacts_ready(base_dir, task_id, paths, generation, verify_content=False):
        return
    base_url = os.getenv("FLOWMESH_BASE_URL", "").strip()
    if not base_url:
        raise ExecutionError(
            f"Upstream task {task_id} hydration requires FLOWMESH_BASE_URL"
        )
    query = [("include", "results"), ("include", "artifacts")]
    if paths is not None:
        query.extend(("artifact_path", path) for path in paths)
    if generation:
        query.append(("generation", generation))
    timeout = _transfer_timeout()
    with tempfile.TemporaryDirectory(prefix="flowmesh-hydrate-") as temporary:
        root = Path(temporary)
        bundle = root / "bundle.tar"
        try:
            with requests.get(
                f"{base_url.rstrip('/')}/api/v1/results/{task_id}/bundle?{urlencode(query)}",
                headers=auth_headers(),
                stream=True,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                with bundle.open("wb") as sink:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            sink.write(chunk)
            staging = extract_delivery_bundle(bundle, root / "staging", task_id)
            if not artifacts_ready(
                staging, task_id, paths, generation, verify_content=False
            ):
                raise ValueError("Downloaded artifact selection is incomplete or stale")
            commit_delivery(staging, base_dir, task_id)
        except (requests.RequestException, OSError, ValueError) as exc:
            raise ExecutionError(
                f"Failed to hydrate upstream task {task_id}: {exc}", retryable=True
            ) from exc


def rewrite_artifact_inputs(value: Any, replacements: dict[str, str]) -> Any:
    """Replace each rendered upstream artifact reference in ``value`` with its local
    path, recursing through containers and placeholder-bearing models."""
    if isinstance(value, str):
        for source, destination in replacements.items():
            value = value.replace(source, destination)
        return value
    if isinstance(value, dict):
        return {
            key: rewrite_artifact_inputs(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_artifact_inputs(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(rewrite_artifact_inputs(item, replacements) for item in value)
    if isinstance(value, BaseModel):
        return value.model_copy(
            update={
                key: rewrite_artifact_inputs(item, replacements)
                for key, item in placeholder_fields(value)
            }
        )
    return value


def _hydrate_inputs(inputs: list[ArtifactInput], results_dir: Path) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for entry in inputs:
        hydrate_result(entry.task_id, results_dir, [entry.path], entry.generation)
        replacements[entry.source] = (
            (results_dir / entry.task_id / "artifacts" / safe_relative(entry.path))
            .absolute()
            .as_posix()
        )
    return replacements


def hydrate_task(task: WorkerTaskMessage, results_dir: Path) -> WorkerTaskMessage:
    """Fetch the upstream results ``task`` reads into ``results_dir`` and return a
    copy whose artifact references, merged children included, point at them."""
    for task_id in (task.upstream_task_ids or {}).values():
        hydrate_result(
            task_id,
            results_dir,
            generation=task.upstream_result_generations.get(task_id),
        )
    parent = _hydrate_inputs(task.artifact_inputs.get(task.task_id, []), results_dir)
    task = task.model_copy(update={"task": rewrite_artifact_inputs(task.task, parent)})
    if task.merged_children:
        task.merged_children = [
            child.model_copy(
                update={
                    "spec": rewrite_artifact_inputs(
                        child.spec,
                        _hydrate_inputs(
                            task.artifact_inputs.get(child.task_id, []), results_dir
                        ),
                    )
                }
            )
            for child in task.merged_children
        ]
    return task
