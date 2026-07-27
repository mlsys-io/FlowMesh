import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from shared.schemas.result import BaseExecutorResult
from shared.utils.http import auth_headers

from ..base_executor import ExecutionError


def artifact_to_source(
    ref: dict[str, Any], context: dict[str, BaseExecutorResult] | None, node: str | None
) -> str:
    """Translate a `{path: ...}` artifact ref into a URL or local path."""
    rel_path = ref.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        raise ExecutionError("Artifact ref must include a non-empty 'path' field")

    if (
        context
        and node
        and (node_result := context.get(node))
        and (ctx := node_result.artifacts_)
    ):
        base_url = ctx.base_url
        base_dir = ctx.base_dir
    else:
        base_url = base_dir = None

    # Check local filesystem first
    base_dir_path = Path(base_dir) if base_dir else None
    local_path = base_dir_path / "artifacts" / rel_path if base_dir_path else None
    if local_path is not None and local_path.is_file():
        return local_path.as_posix()

    # Fallback to a URL if base_url is provided
    if base_url:
        if not base_dir_path:
            raise ExecutionError(
                "Artifact ref with base_url requires upstream base_dir to "
                "derive task_id"
            )
        task_id = base_dir_path.name
        return f"{base_url.rstrip('/')}/api/v1/results/{task_id}/files/{rel_path}"

    # Return the local path anyway
    if local_path is not None:
        return local_path.as_posix()

    raise ExecutionError(
        "Cannot resolve artifact ref: upstream _artifacts context is missing"
    )


def is_flowmesh_origin_url(url: str) -> bool:
    """Whether `url` targets this worker's configured FlowMesh server.

    Gates whether the worker's bearer token (`FLOWMESH_API_KEY`, attached via
    `auth_headers()`) may be sent: only to the worker's own FlowMesh origin
    (`FLOWMESH_BASE_URL`), never to an arbitrary or public URL a workflow
    happens to reference.
    """
    base_url = os.getenv("FLOWMESH_BASE_URL", "").strip()
    if not base_url:
        return False
    base = urlparse(base_url)
    if not base.scheme or not base.netloc:
        return False
    target = urlparse(url)
    return (
        target.scheme.lower() == base.scheme.lower()
        and target.netloc.lower() == base.netloc.lower()
    )


def maybe_resolve_artifact_ref(
    value: Any, context: dict[str, BaseExecutorResult] | None, node: str | None
) -> Any:
    """Convert `{path: ...}` ref dicts to URL/path strings; pass others through."""
    if isinstance(value, dict) and "path" in value:
        return artifact_to_source(value, context, node)
    return value


def resolve_artifact(source: str, timeout: float = 1800) -> Path:
    """Download an artifact URL (or symlink a local path) into a tempfile.

    `source` is either an absolute HTTP/HTTPS URL or an absolute filesystem
    path. Callers normally get it by running `artifact_to_source` on a
    `{path: ...}` ref dict resolved via the task's upstream results.
    """
    if not isinstance(source, str) or not source:
        raise ExecutionError("Artifact source is missing for embedding resolution")

    parsed = urlparse(source)
    temp_dir = Path(tempfile.mkdtemp(prefix="flowmesh-artifact-"))
    filename = Path(parsed.path).name or "artifact"
    local_path = temp_dir / filename

    if not parsed.scheme:
        if not (src_path := Path(source)).is_file():
            raise ExecutionError(f"Artifact source path does not exist: {src_path}")
        try:
            os.symlink(src_path.resolve(), local_path)
        except Exception as exc:
            raise ExecutionError(
                f"Failed to link artifact from {src_path}: {exc}"
            ) from exc
        return local_path

    try:
        with requests.get(
            source, headers=auth_headers(), stream=True, timeout=timeout
        ) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_path
    except Exception as exc:
        raise ExecutionError(
            f"Failed to resolve embedding artifact: {exc}", retryable=True
        ) from exc
