"""Tests for the workflow lineage / profile router."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.responses import StreamingResponse

from server.routers.v1 import logs as logs_router


def _seed_task_logs(
    base: Path,
    task_id: str,
    events: list[dict[str, Any]] | None = None,
    assets: list[dict[str, Any]] | None = None,
    lineage: list[dict[str, Any]] | None = None,
) -> None:
    logs_dir = base / task_id / "logs"
    logs_dir.mkdir(parents=True)
    if events is not None:
        (logs_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(row) for row in events) + "\n"
        )
    if assets is not None:
        (logs_dir / "assets.jsonl").write_text(
            "\n".join(json.dumps(row) for row in assets) + "\n"
        )
    if lineage is not None:
        (logs_dir / "lineage.jsonl").write_text(
            "\n".join(json.dumps(row) for row in lineage) + "\n"
        )


def _registry(task_ids: list[str]):
    workflow = type("WF", (), {"task_ids": task_ids})()
    registry = AsyncMock()
    registry.get_workflow_async.return_value = workflow
    return registry


async def _collect_streamed_lines(response: StreamingResponse) -> list[str]:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk)
        elif isinstance(chunk, memoryview):
            chunks.append(bytes(chunk))
        else:
            chunks.append(chunk.encode("utf-8"))
    text = b"".join(chunks).decode("utf-8")
    return [line for line in text.split("\n") if line]


@pytest.mark.anyio
async def test_get_workflow_lineage_concats_across_tasks(tmp_path: Path) -> None:
    _seed_task_logs(
        tmp_path,
        "tsk-a",
        events=[{"event_type": "write", "data_id": "tsk-a"}],
        assets=[{"data_id": "tsk-a", "asset_guid": "g-a", "version": 1}],
    )
    _seed_task_logs(
        tmp_path,
        "tsk-b",
        events=[{"event_type": "read", "data_id": "tsk-a"}],
        lineage=[{"data_id": "tsk-b", "source_data_id": "tsk-a"}],
    )

    response = await logs_router.get_workflow_lineage(
        workflow_id="wfl-1",
        kind="events",
        registry=_registry(["tsk-a", "tsk-b"]),
        results_dir=tmp_path,
    )
    lines = await _collect_streamed_lines(response)
    parsed = [json.loads(line) for line in lines]
    assert [row["event_type"] for row in parsed] == ["write", "read"]


@pytest.mark.anyio
async def test_get_workflow_lineage_skips_missing_files(tmp_path: Path) -> None:
    _seed_task_logs(
        tmp_path,
        "tsk-a",
        assets=[{"data_id": "tsk-a", "asset_guid": "g-a", "version": 1}],
    )
    response = await logs_router.get_workflow_lineage(
        workflow_id="wfl-1",
        kind="assets",
        registry=_registry(["tsk-a", "tsk-b-missing"]),
        results_dir=tmp_path,
    )
    parsed = [json.loads(line) for line in await _collect_streamed_lines(response)]
    assert len(parsed) == 1
    assert parsed[0]["data_id"] == "tsk-a"


@pytest.mark.anyio
async def test_get_workflow_lineage_unknown_kind(tmp_path: Path) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        await logs_router.get_workflow_lineage(
            workflow_id="wfl-1",
            kind="bogus",
            registry=_registry(["tsk-a"]),
            results_dir=tmp_path,
        )
    assert excinfo.value.status_code == 400


@pytest.mark.anyio
async def test_get_workflow_profile_runs_analyzer(tmp_path: Path) -> None:
    _seed_task_logs(
        tmp_path,
        "tsk-a",
        events=[
            {
                "timestamp": "2026-04-29T00:00:00+00:00",
                "event_type": "write request transfer",
                "data_id": "tsk-a",
            }
        ],
        assets=[
            {
                "data_id": "tsk-a",
                "asset_guid": "g-a",
                "version": 1,
                "user_id": "alice",
                "created_at": "2026-04-29T00:00:00+00:00",
            }
        ],
    )

    summary = await logs_router.get_workflow_profile(
        workflow_id="wfl-1",
        registry=_registry(["tsk-a"]),
        results_dir=tmp_path,
    )
    assert summary.total_events == 1
    assert summary.total_assets == 1
    assert summary.write_count == 1
    assert summary.assets[0].asset_guid == "g-a"


@pytest.mark.anyio
async def test_workflow_not_found_raises_404(tmp_path: Path) -> None:
    from fastapi import HTTPException

    registry = AsyncMock()
    registry.get_workflow_async.return_value = None

    with pytest.raises(HTTPException) as excinfo:
        await logs_router.get_workflow_lineage(
            workflow_id="wfl-missing",
            kind="events",
            registry=registry,
            results_dir=tmp_path,
        )
    assert excinfo.value.status_code == 404
