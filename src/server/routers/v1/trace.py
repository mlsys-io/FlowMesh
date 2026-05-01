"""Workflow trace endpoints: stream rows or run the analyzer."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from shared.governance import ProfileSummary, analyze
from shared.utils.jsonl import encode_jsonl_bytes, read_jsonl

from ...app_state import get_results_dir, get_workflow_registry
from ...registries.workflow import WorkflowRegistry
from ...schemas.result import result_file_path

router = APIRouter(prefix="/workflows", tags=["Trace"])

_KIND_TO_FILENAME: dict[str, str] = {
    "spans": "spans.jsonl",
    "assets": "assets.jsonl",
    "lineage": "lineage.jsonl",
}


def _logs_dir_for_task(results_dir: Path, task_id: str) -> Path:
    return result_file_path(results_dir, task_id).parent / "artifacts" / "logs"


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
    "/{workflow_id}/trace/analyze",
    summary="Run the trace analyzer; return ProfileSummary",
    response_model=ProfileSummary,
)
async def analyze_workflow_trace(
    workflow_id: str,
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
) -> ProfileSummary:
    task_ids = await _resolve_task_ids(workflow_id, registry)
    spans = list(_iter_workflow_jsonl(results_dir, task_ids, "spans.jsonl"))
    assets = list(_iter_workflow_jsonl(results_dir, task_ids, "assets.jsonl"))
    lineage = list(_iter_workflow_jsonl(results_dir, task_ids, "lineage.jsonl"))
    return analyze(spans, assets, lineage, workflow_id=workflow_id)


@router.get(
    "/{workflow_id}/trace/{kind}",
    summary="Stream JSONL rows (spans / assets / lineage)",
)
async def get_workflow_trace(
    workflow_id: str,
    kind: str,
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
) -> StreamingResponse:
    filename = _KIND_TO_FILENAME.get(kind)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown kind '{kind}'; expected spans, assets, or lineage",
        )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    return StreamingResponse(
        encode_jsonl_bytes(_iter_workflow_jsonl(results_dir, task_ids, filename)),
        media_type="application/x-ndjson",
    )
