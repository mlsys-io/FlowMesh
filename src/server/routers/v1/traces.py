"""Trace endpoints — per-task upload, workflow-level read + analyzer."""

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from shared.utils.json import encode_jsonl_bytes, read_jsonl

from ...app_state import get_logger, get_results_dir, get_workflow_registry
from ...auth.security import (
    PrincipalContext,
    authenticate_request,
    require_permission,
)
from ...governance import ProfileSummary, analyze
from ...hooks import ResourceAction, ResourceType
from ...registries.workflow import WorkflowRegistry
from ...schemas.common import PathResponse
from ...schemas.result import result_file_path

router = APIRouter(prefix="/traces", tags=["Traces"])

_TYPE_TO_FILENAME: dict[str, str] = {
    "spans": "spans.jsonl",
    "assets": "assets.jsonl",
    "lineage": "lineage.jsonl",
}


def _logs_dir_for_task(results_dir: Path, task_id: str) -> Path:
    """Per-task ``logs/`` directory holding the trace JSONL artifacts."""
    return result_file_path(results_dir, task_id).parent / "logs"


def _iter_workflow_jsonl(
    results_dir: Path, task_ids: Iterable[str], filename: str
) -> Iterator[dict[str, Any]]:
    for task_id in task_ids:
        yield from read_jsonl(_logs_dir_for_task(results_dir, task_id) / filename)


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
    principal: PrincipalContext = Depends(authenticate_request),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> ProfileSummary:
    await require_permission(
        principal, ResourceType.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    spans = list(_iter_workflow_jsonl(results_dir, task_ids, "spans.jsonl"))
    assets = list(_iter_workflow_jsonl(results_dir, task_ids, "assets.jsonl"))
    lineage = list(_iter_workflow_jsonl(results_dir, task_ids, "lineage.jsonl"))
    return analyze(spans, assets, lineage, workflow_id=workflow_id)


@router.get(
    "/workflows/{workflow_id}/{trace_type}",
    summary="Stream JSONL rows (spans / assets / lineage)",
)
async def get_workflow_trace(
    workflow_id: str,
    trace_type: str,
    principal: PrincipalContext = Depends(authenticate_request),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    await require_permission(
        principal, ResourceType.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    filename = _TYPE_TO_FILENAME.get(trace_type)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown type '{trace_type}'; expected spans, assets, or lineage",
        )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    return StreamingResponse(
        encode_jsonl_bytes(_iter_workflow_jsonl(results_dir, task_ids, filename)),
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
    principal: PrincipalContext = Depends(authenticate_request),
    results_dir: Path = Depends(get_results_dir),
) -> PathResponse:
    del principal  # auth gate; the supplier-key principal isn't task-scoped
    filename = _TYPE_TO_FILENAME.get(trace_type)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown type '{trace_type}'; expected spans, assets, or lineage",
        )
    target_path = _logs_dir_for_task(results_dir, task_id) / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target_path.open("wb") as out:
            out.write(await file.read())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store trace: {exc}",
        ) from exc
    return PathResponse(ok=True, path=target_path.as_posix())
