"""Tests for the workflow traces router."""

import json
import logging
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from lumid_hooks import PrincipalContext, ResourceRef

from server.hooks import PERMISSION_CHECKERS
from server.routers.v1 import traces as traces_router


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.traces_router")


def _principal() -> PrincipalContext:
    return PrincipalContext(
        principal_id="p-1",
        org_id="org",
        external_id="ext",
        principal_type="user",
        scopes=[],
    )


class _DenyAllChecker:
    name = "deny-all"

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="denied")

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return frozenset[str]()


@pytest.fixture
def deny_all_permissions() -> Iterator[None]:
    PERMISSION_CHECKERS.append(_DenyAllChecker())
    try:
        yield
    finally:
        PERMISSION_CHECKERS.clear()


def _otel_span(
    name: str,
    *,
    data_id: str,
    start: str,
    end: str,
    span_type: str,
    span_id: str = "0xa3f1e9d2c5b40678",
    parent_id: str | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {"data_id": data_id, "flowmesh.type": span_type}
    if batch_id:
        attributes["batch_id"] = batch_id
    return {
        "name": name,
        "context": {
            "trace_id": "0xfbad6be5c4434181a2d394eac830dea1",
            "span_id": span_id,
        },
        "parent_id": parent_id,
        "start_time": start,
        "end_time": end,
        "status": {"status_code": "OK"},
        "attributes": attributes,
    }


def _seed_task_logs(
    base: Path,
    task_id: str,
    spans: list[dict[str, Any]] | None = None,
    assets: list[dict[str, Any]] | None = None,
    lineage: list[dict[str, Any]] | None = None,
) -> None:
    logs_dir = base / task_id / "logs"
    logs_dir.mkdir(parents=True)
    if spans is not None:
        (logs_dir / "spans.jsonl").write_text(
            "\n".join(json.dumps(row) for row in spans) + "\n"
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
async def test_get_workflow_trace_concats_across_tasks(tmp_path: Path) -> None:
    _seed_task_logs(
        tmp_path,
        "tsk-a",
        spans=[
            _otel_span(
                "write",
                data_id="tsk-a",
                start="2026-04-29T00:00:00+00:00",
                end="2026-04-29T00:00:01+00:00",
                span_type="network",
                span_id="0xaaaa000000000001",
            )
        ],
        assets=[{"data_id": "tsk-a", "asset_guid": "g-a", "version": 1}],
    )
    _seed_task_logs(
        tmp_path,
        "tsk-b",
        spans=[
            _otel_span(
                "read",
                data_id="tsk-a",
                start="2026-04-29T00:00:01+00:00",
                end="2026-04-29T00:00:02+00:00",
                span_type="network",
                span_id="0xbbbb000000000001",
            )
        ],
        lineage=[{"data_id": "tsk-b", "source_data_id": "tsk-a"}],
    )

    response = await traces_router.get_workflow_trace(
        workflow_id="wfl-1",
        trace_type="spans",
        registry=_registry(["tsk-a", "tsk-b"]),
        results_dir=tmp_path,
    )
    lines = await _collect_streamed_lines(response)
    parsed = [json.loads(line) for line in lines]
    assert [row["name"] for row in parsed] == ["write", "read"]


@pytest.mark.anyio
async def test_get_workflow_trace_skips_missing_files(tmp_path: Path) -> None:
    _seed_task_logs(
        tmp_path,
        "tsk-a",
        assets=[{"data_id": "tsk-a", "asset_guid": "g-a", "version": 1}],
    )
    response = await traces_router.get_workflow_trace(
        workflow_id="wfl-1",
        trace_type="assets",
        registry=_registry(["tsk-a", "tsk-b-missing"]),
        results_dir=tmp_path,
    )
    parsed = [json.loads(line) for line in await _collect_streamed_lines(response)]
    assert len(parsed) == 1
    assert parsed[0]["data_id"] == "tsk-a"


@pytest.mark.anyio
async def test_get_workflow_trace_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as excinfo:
        await traces_router.get_workflow_trace(
            workflow_id="wfl-1",
            trace_type="bogus",
            registry=_registry(["tsk-a"]),
            results_dir=tmp_path,
        )
    assert excinfo.value.status_code == 400


@pytest.mark.anyio
async def test_analyze_workflow_trace_runs_analyzer(tmp_path: Path) -> None:
    _seed_task_logs(
        tmp_path,
        "tsk-a",
        spans=[
            _otel_span(
                "task",
                data_id="tsk-a",
                start="2026-04-29T00:00:00+00:00",
                end="2026-04-29T00:00:01+00:00",
                span_type="compute",
                span_id="0xaaaa000000000001",
            ),
            _otel_span(
                "write",
                data_id="tsk-a",
                start="2026-04-29T00:00:00.500000+00:00",
                end="2026-04-29T00:00:01+00:00",
                span_type="network",
                parent_id="0xaaaa000000000001",
                span_id="0xaaaa000000000002",
            ),
            _otel_span(
                "dump to storage",
                data_id="tsk-a",
                start="2026-04-29T00:00:01+00:00",
                end="2026-04-29T00:00:01+00:00",
                span_type="marker",
                parent_id="0xaaaa000000000001",
                span_id="0xaaaa000000000003",
            ),
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

    summary = await traces_router.analyze_workflow_trace(
        workflow_id="wfl-1",
        registry=_registry(["tsk-a"]),
        results_dir=tmp_path,
    )
    assert summary.event_count == 3
    assert len(summary.assets) == 1
    assert summary.workflow_id == "wfl-1"
    assert summary.critical_path is not None
    assert summary.critical_path.path == ["tsk-a"]
    assert summary.assets[0].asset_guid == "g-a"
    assert "write" in summary.e2e_breakdown.network_summary.event_type


@pytest.mark.anyio
async def test_workflow_not_found_raises_404(tmp_path: Path) -> None:
    registry = AsyncMock()
    registry.get_workflow_async.return_value = None

    with pytest.raises(HTTPException) as excinfo:
        await traces_router.get_workflow_trace(
            workflow_id="wfl-missing",
            trace_type="spans",
            registry=registry,
            results_dir=tmp_path,
        )
    assert excinfo.value.status_code == 404


def _upload(content: bytes, filename: str = "ignored.jsonl") -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


@pytest.mark.anyio
@pytest.mark.parametrize("trace_type", ["spans", "assets", "lineage"])
async def test_upload_task_trace_writes_named_file(
    tmp_path: Path, trace_type: str
) -> None:
    payload = b'{"name":"task"}\n'
    response = await traces_router.upload_task_trace(
        task_id="tsk-up",
        trace_type=trace_type,
        file=_upload(payload),
        results_dir=tmp_path,
    )
    target = tmp_path / "tsk-up" / "logs" / f"{trace_type}.jsonl"
    assert target.is_file()
    assert target.read_bytes() == payload
    assert response.path == target.as_posix()


@pytest.mark.anyio
async def test_upload_task_trace_ignores_client_filename(tmp_path: Path) -> None:
    """Server filename comes from {trace_type}, never from the multipart name."""
    await traces_router.upload_task_trace(
        task_id="tsk-x",
        trace_type="spans",
        file=_upload(b"row\n", filename="../../escape.jsonl"),
        results_dir=tmp_path,
    )
    assert (tmp_path / "tsk-x" / "logs" / "spans.jsonl").is_file()
    assert not (tmp_path / "escape.jsonl").exists()


@pytest.mark.anyio
async def test_upload_task_trace_unknown_type_400(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as excinfo:
        await traces_router.upload_task_trace(
            task_id="tsk-x",
            trace_type="bogus",
            file=_upload(b""),
            results_dir=tmp_path,
        )
    assert excinfo.value.status_code == 400


@pytest.mark.anyio
async def test_upload_task_trace_denied_without_permission(
    deny_all_permissions: None, logger: logging.Logger
) -> None:
    with pytest.raises(HTTPException) as exc:
        await traces_router.upload_task_trace(
            task_id="tsk-x",
            trace_type="spans",
            principal=_principal(),
            logger=logger,
        )
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN
