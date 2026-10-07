"""Trace endpoints — per-task upload, workflow-level read + analyzer."""

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import TypeAdapter

from shared.schemas.result import result_file_path
from shared.utils.atomic import atomic_write_stream
from shared.utils.json import encode_jsonl_bytes

from ...app_state import get_logger, get_results_dir, get_workflow_registry
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...governance import ProfileSummary, analyze
from ...hooks import ResourceAction, ResourceKind
from ...registries.workflow import WorkflowRegistry
from ...schemas.common import PathResponse

router = APIRouter(prefix="/traces", tags=["Traces"])

_TYPE_TO_FILENAME: dict[str, str] = {
    "spans": "spans.jsonl",
    "assets": "assets.jsonl",
    "lineage": "lineage.jsonl",
}

# A trace file holds JSON lines our writers keep far shorter than this; a file with
# a longer line, or with this many lines in a row that hold no row, is not a trace,
# and the rest of it is not read.
_MAX_TRACE_LINE_BYTES = 4 << 20
_MAX_SKIPPED_TRACE_LINES = 100


def _logs_dir_for_task(results_dir: Path, task_id: str) -> Path:
    """Per-task ``logs/`` directory holding the trace JSONL artifacts."""
    return result_file_path(results_dir, task_id).parent / "logs"


def _iter_workflow_jsonl(
    results_dir: Path,
    task_ids: Iterable[str],
    filename: str,
    logger: logging.Logger,
) -> Iterator[dict[str, Any]]:
    for task_id in task_ids:
        path = _logs_dir_for_task(results_dir, task_id) / filename
        if not path.exists() or not path.is_file():
            continue
        with path.open("rb") as fh:
            yield from _rows(fh, filename, task_id, logger)


def _rows(
    fh: BinaryIO, filename: str, task_id: str, logger: logging.Logger
) -> Iterator[dict[str, Any]]:
    skipped = 0
    while line := fh.readline(_MAX_TRACE_LINE_BYTES + 1):
        if len(line) > _MAX_TRACE_LINE_BYTES and not line.endswith(b"\n"):
            _stop(
                task_id,
                filename,
                f"a line longer than {_MAX_TRACE_LINE_BYTES} bytes",
                logger,
            )
            return
        row = _parse_row(line)
        if row is None:
            skipped += 1
            if skipped >= _MAX_SKIPPED_TRACE_LINES:
                _stop(
                    task_id,
                    filename,
                    f"{skipped} lines in a row that hold no row",
                    logger,
                )
                return
            continue
        skipped = 0
        yield row


def _stop(task_id: str, filename: str, reason: str, logger: logging.Logger) -> None:
    logger.warning(
        "Not reading the rest of task %s's %s: %s", task_id, filename, reason
    )


def _parse_row(line: bytes) -> Any | None:
    """The JSON value on ``line``, or None when it holds none."""
    try:
        return json.loads(line) if line.strip() else None
    except (ValueError, RecursionError):
        return None


async def _resolve_task_ids(workflow_id: str, registry: WorkflowRegistry) -> list[str]:
    workflow = await registry.get_workflow_async(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )
    return workflow.task_ids


@router.get(
    "/workflows/analyze/{workflow_id}",
    summary="Run the trace analyzer; return ProfileSummary",
    response_model=ProfileSummary,
)
async def analyze_workflow_trace(
    workflow_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    body = await run_in_threadpool(
        _analyze_workflow, results_dir, task_ids, workflow_id, logger
    )
    return Response(body, media_type="application/json")


_PROFILE_SUMMARY = TypeAdapter(ProfileSummary)


def _analyze_workflow(
    results_dir: Path,
    task_ids: list[str],
    workflow_id: str,
    logger: logging.Logger,
) -> bytes:
    spans = list(_iter_workflow_jsonl(results_dir, task_ids, "spans.jsonl", logger))
    assets = list(_iter_workflow_jsonl(results_dir, task_ids, "assets.jsonl", logger))
    lineage = list(_iter_workflow_jsonl(results_dir, task_ids, "lineage.jsonl", logger))
    summary = analyze(spans, assets, lineage, workflow_id=workflow_id)
    return _PROFILE_SUMMARY.dump_json(summary, by_alias=True)


@router.get(
    "/workflows/{workflow_id}/{trace_type}",
    summary="Stream JSONL rows (spans / assets / lineage)",
)
async def get_workflow_trace(
    workflow_id: str,
    trace_type: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    filename = _TYPE_TO_FILENAME.get(trace_type)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown type '{trace_type}'; expected spans, assets, or lineage",
        )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    return StreamingResponse(
        encode_jsonl_bytes(
            _iter_workflow_jsonl(results_dir, task_ids, filename, logger)
        ),
        media_type="application/x-ndjson",
    )


@router.post(
    "/tasks/{task_id}/{trace_type}",
    summary="Upload a per-task trace JSONL file (spans / assets / lineage)",
)
async def upload_task_trace(
    task_id: str,
    trace_type: str,
    file: UploadFile = File(...),
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> PathResponse:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    filename = _TYPE_TO_FILENAME.get(trace_type)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown type '{trace_type}'; expected spans, assets, or lineage",
        )
    target_path = _logs_dir_for_task(results_dir, task_id) / filename
    try:
        await run_in_threadpool(atomic_write_stream, target_path, file.file)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store trace: {exc}",
        ) from exc
    return PathResponse(ok=True, path=target_path.as_posix())
