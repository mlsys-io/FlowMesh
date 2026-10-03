import io
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from lumid_hooks import PrincipalContext, ResourceRef

from server.hooks import PERMISSION_CHECKERS
from server.routers.v1 import results as results_router
from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.utils import atomic


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
    )

    assert isinstance(response, FileResponse)
    assert Path(response.path) == artifact_path


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
    )

    assert isinstance(response, FileResponse)
    assert Path(response.path) == root_file


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
async def test_upload_result_file_copies_in_chunks_off_the_event_loop(
    tmp_path: Path, logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    copies: list[tuple[int, int]] = []
    copy = atomic.shutil.copyfileobj

    def _recording(source: Any, target: Any, length: int = 0) -> None:
        copies.append((threading.get_ident(), length))
        copy(source, target, length)

    async def _whole_read(*args: Any) -> bytes:
        raise AssertionError("the upload was read whole")

    monkeypatch.setattr(atomic.shutil, "copyfileobj", _recording)
    upload = UploadFile(file=io.BytesIO(b"x" * 10), filename="out.bin")
    monkeypatch.setattr(upload, "read", _whole_read)

    await results_router.upload_result_file(
        task_id="task-1",
        file=upload,
        runtime=cast(Any, SimpleNamespace(get_record=lambda _task_id: None)),
        principal=_principal(),
        results_dir=tmp_path,
        logger=logger,
    )

    assert copies == [(copies[0][0], 1 << 20)]
    assert copies[0][0] != threading.get_ident()
    assert (tmp_path / "task-1" / "artifacts" / "out.bin").read_bytes() == b"x" * 10


@pytest.mark.anyio
async def test_download_result_bundle_builds_off_the_event_loop(
    tmp_path: Path, logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []
    build = results_router._create_result_bundle_archive

    def _recording(*args: Any, **kwargs: Any) -> Path:
        threads.append(threading.get_ident())
        return build(*args, **kwargs)

    monkeypatch.setattr(results_router, "_create_result_bundle_archive", _recording)
    stub = SimpleNamespace(
        get_record=lambda _task_id: None,
        read_result_bytes=lambda _task_id: b"{}",
    )
    (tmp_path / "t-1").mkdir(parents=True, exist_ok=True)

    response = await results_router.download_result_bundle(
        task_id="t-1",
        background_tasks=BackgroundTasks(),
        include=[],
        principal=_principal(),
        runtime=cast(Any, stub),
        results_dir=tmp_path,
        logger=logger,
    )

    assert isinstance(response, FileResponse)
    assert threads and threads[0] != threading.get_ident()
    Path(response.path).unlink()
