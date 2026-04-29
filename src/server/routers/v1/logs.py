"""Workflow-scoped lineage and profile endpoints.

The worker emits per-task `events.jsonl`, `assets.jsonl`, and `lineage.jsonl`
under `out_dir/artifacts/logs/`; the artifact upload pipeline lands them at
`{results_dir}/{task_id}/logs/<file>.jsonl` on the server. These endpoints
read each task in a workflow and concatenate the rows.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from shared.profile import ProfileSummary, analyze

from ...app_state import get_results_dir, get_workflow_registry
from ...registries.workflow import WorkflowRegistry
from ...schemas.result import result_file_path

router = APIRouter(prefix="/workflows", tags=["Logs"])

_KIND_TO_FILENAME: dict[str, str] = {
    "events": "events.jsonl",
    "assets": "assets.jsonl",
    "lineage": "lineage.jsonl",
}


def _logs_dir_for_task(results_dir: Path, task_id: str) -> Path:
    return result_file_path(results_dir, task_id).parent / "logs"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_workflow_jsonl(
    results_dir: Path, task_ids: Iterable[str], filename: str
) -> Iterator[dict[str, Any]]:
    for task_id in task_ids:
        path = _logs_dir_for_task(results_dir, task_id) / filename
        yield from _iter_jsonl(path)


async def _resolve_task_ids(
    workflow_id: str, registry: WorkflowRegistry
) -> list[str]:
    workflow = await registry.get_workflow_async(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )
    return list(workflow.task_ids)


def _stream_jsonl(rows: Iterator[dict[str, Any]]) -> Iterator[bytes]:
    for row in rows:
        yield (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")


@router.get(
    "/{workflow_id}/logs/{kind}",
    summary="Stream workflow lineage rows",
    description=(
        "Stream concatenated JSONL rows for `events`, `assets`, or `lineage` "
        "across every task in the workflow."
    ),
)
async def get_workflow_lineage(
    workflow_id: str,
    kind: str,
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
) -> StreamingResponse:
    filename = _KIND_TO_FILENAME.get(kind)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown kind '{kind}'; expected events, assets, or lineage",
        )
    task_ids = await _resolve_task_ids(workflow_id, registry)
    return StreamingResponse(
        _stream_jsonl(_iter_workflow_jsonl(results_dir, task_ids, filename)),
        media_type="application/x-ndjson",
    )


@router.get(
    "/{workflow_id}/profile",
    summary="Profile a workflow's lineage",
    description=(
        "Run the trace analyzer over the workflow's events / assets / lineage "
        "rows and return a structured `ProfileSummary`."
    ),
    response_model=ProfileSummary,
)
async def get_workflow_profile(
    workflow_id: str,
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    results_dir: Path = Depends(get_results_dir),
) -> ProfileSummary:
    task_ids = await _resolve_task_ids(workflow_id, registry)
    events = list(_iter_workflow_jsonl(results_dir, task_ids, "events.jsonl"))
    assets = list(_iter_workflow_jsonl(results_dir, task_ids, "assets.jsonl"))
    lineage = list(_iter_workflow_jsonl(results_dir, task_ids, "lineage.jsonl"))
    return analyze(events, assets, lineage)
