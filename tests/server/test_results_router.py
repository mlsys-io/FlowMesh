import asyncio
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from lumid_hooks import PrincipalContext, ResourceRef

from server.hooks import PERMISSION_CHECKERS
from server.routers.v1 import results as results_router
from server.task.runtime import TaskRuntime
from shared.schemas.result import BaseExecutorResult, ResultEnvelope, write_result
from shared.utils.result_delivery import delivery_lock


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.results_router")


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


def _runtime_without_records() -> TaskRuntime:
    runtime = Mock(spec=TaskRuntime)
    runtime.get_record.return_value = None
    return cast(TaskRuntime, runtime)


async def _body(response: StreamingResponse) -> bytes:
    chunks = [chunk async for chunk in response.body_iterator]
    return b"".join(c.encode() if isinstance(c, str) else bytes(c) for c in chunks)


def test_download_result_file_route_uses_path_converter() -> None:
    route = next(
        route
        for route in results_router.router.routes
        if isinstance(route, APIRoute)
        and route.path == "/results/{task_id}/files/{filename:path}"
    )
    route = cast(APIRoute, route)
    assert route.path == "/results/{task_id}/files/{filename:path}"


@pytest.mark.anyio
async def test_download_result_file_resolves_flat_name_under_artifacts(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task-1"
    artifacts_dir = task_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    artifact_path = artifacts_dir / "result.json"
    artifact_path.write_text('{"ok":true}', encoding="utf-8")

    response = await results_router.download_result_file(
        task_id="task-1",
        filename="result.json",
        results_dir=tmp_path,
        runtime=_runtime_without_records(),
    )

    assert isinstance(response, StreamingResponse)
    assert await _body(response) == artifact_path.read_bytes()
    assert response.headers["content-length"] == str(artifact_path.stat().st_size)


@pytest.mark.anyio
async def test_download_result_file_falls_back_to_task_root_for_flat_filename(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task-1"
    task_dir.mkdir(parents=True)
    root_file = task_dir / "result.json"
    root_file.write_text('{"ok":true}', encoding="utf-8")

    response = await results_router.download_result_file(
        task_id="task-1",
        filename="result.json",
        results_dir=tmp_path,
        runtime=_runtime_without_records(),
    )

    assert isinstance(response, StreamingResponse)
    assert await _body(response) == root_file.read_bytes()
    assert response.headers["content-length"] == str(root_file.stat().st_size)


def test_resolve_artifact_relative_path_scopes_nested_paths_to_artifacts() -> None:
    assert results_router._resolve_artifact_path("result.json") == Path(
        "artifacts/result.json"
    )
    assert results_router._resolve_artifact_path("nested/result.json") == Path(
        "artifacts/nested/result.json"
    )
    assert results_router._resolve_artifact_path("artifacts/result.json") == Path(
        "artifacts/artifacts/result.json"
    )
    assert results_router._resolve_artifact_path(
        "artifacts/nested/result.json"
    ) == Path("artifacts/artifacts/nested/result.json")


def test_resolve_artifact_relative_path_rejects_invalid_paths() -> None:
    with pytest.raises(Exception):
        results_router._resolve_artifact_path("../result.json")


@pytest.mark.anyio
async def test_ingest_result_denied_without_permission(
    deny_all_permissions: None, logger: logging.Logger
) -> None:
    envelope = ResultEnvelope(task_id="t-1", result=BaseExecutorResult())
    with pytest.raises(HTTPException) as exc:
        await results_router.ingest_result(
            envelope=envelope, principal=_principal(), logger=logger
        )
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.anyio
async def test_upload_result_file_denied_without_permission(
    deny_all_permissions: None, logger: logging.Logger
) -> None:
    with pytest.raises(HTTPException) as exc:
        await results_router.upload_result_file(
            task_id="t-1", principal=_principal(), logger=logger
        )
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.anyio
async def test_get_result_waits_for_the_snapshot_lock_off_the_event_loop(
    tmp_path: Path, logger: logging.Logger
) -> None:
    write_result(tmp_path, ResultEnvelope(task_id="t-1", result=BaseExecutorResult()))
    held, release = threading.Event(), threading.Event()

    def hold_lock() -> None:
        with delivery_lock(tmp_path / "t-1"):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    held.wait(5)
    try:
        pending = asyncio.ensure_future(
            results_router.get_result(
                task_id="t-1",
                principal=_principal(),
                results_dir=tmp_path,
                runtime=_runtime_without_records(),
                logger=logger,
            )
        )
        await asyncio.sleep(0.05)
        assert not pending.done()
    finally:
        release.set()
        holder.join()
    assert isinstance(await pending, BaseExecutorResult)
