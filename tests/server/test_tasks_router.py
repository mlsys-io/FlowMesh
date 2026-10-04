"""Router tests for `GET /api/v1/tasks` workflow-scoped, non-blocking listing."""

import asyncio
import logging
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from lumid_hooks import ResourceRef

from server.app_state import get_runtime
from server.auth.security import PrincipalContext, authenticate_connection
from server.hooks import PERMISSION_CHECKERS
from server.routers.v1 import tasks as tasks_router
from server.task.runtime import TaskRuntime
from tests.server.task.merge_harness import build_runtime

PREFIX = "/api/v1"

_PAYLOAD = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: list-tasks
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
      - name: b
        spec:
          taskType: echo
"""


class _ScopedChecker:
    """Scopes task access to a fixed set of task ids."""

    name = "scoped"

    def __init__(self, allowed: frozenset[str]) -> None:
        self._allowed = allowed

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        return None

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return self._allowed


@pytest.fixture
def no_checkers() -> Iterator[None]:
    PERMISSION_CHECKERS.clear()
    yield
    PERMISSION_CHECKERS.clear()


def _client(runtime: TaskRuntime) -> AsyncClient:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.tasks_router")
    app.include_router(tasks_router.router, prefix=PREFIX)
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.dependency_overrides[authenticate_connection] = lambda: MagicMock(
        spec=PrincipalContext
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def _task_ids(runtime: TaskRuntime, workflow_id: str) -> list[str]:
    return [
        task_id
        for task_id, record in runtime.tasks.items()
        if record.workflow_id == workflow_id
    ]


async def _register(runtime: TaskRuntime, payload: str) -> str:
    workflow_id, _ = await runtime.register("owner", "org", payload, format="native")
    return workflow_id


@pytest.mark.anyio
async def test_list_tasks_filters_by_workflow_id_with_permissions(
    no_checkers: None,
) -> None:
    runtime, _ = build_runtime()
    wf_a = await _register(runtime, _PAYLOAD)
    wf_b = await _register(runtime, _PAYLOAD)
    ids_a = _task_ids(runtime, wf_a)
    ids_b = _task_ids(runtime, wf_b)

    # Permission checker scopes access to only one task of workflow A.
    PERMISSION_CHECKERS.append(_ScopedChecker(frozenset({ids_a[0]})))
    try:
        async with _client(runtime) as ac:
            resp = await ac.get(f"{PREFIX}/tasks", params={"workflow_id": wf_a})
    finally:
        PERMISSION_CHECKERS.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert [t["task_id"] for t in body] == [ids_a[0]]
    assert all(t["workflow_id"] == wf_a for t in body)
    # No task from the other workflow leaks through.
    assert all(t["task_id"] not in ids_b for t in body)


@pytest.mark.anyio
async def test_list_tasks_repeated_workflow_id_returns_both(no_checkers: None) -> None:
    runtime, _ = build_runtime()
    wf_a = await _register(runtime, _PAYLOAD)
    wf_b = await _register(runtime, _PAYLOAD)
    ids_a = _task_ids(runtime, wf_a)
    ids_b = _task_ids(runtime, wf_b)

    async with _client(runtime) as ac:
        resp = await ac.get(
            f"{PREFIX}/tasks", params=[("workflow_id", wf_a), ("workflow_id", wf_b)]
        )

    assert resp.status_code == 200
    body = resp.json()
    assert {t["task_id"] for t in body} == set(ids_a) | set(ids_b)


@pytest.mark.anyio
async def test_list_tasks_repeated_status_returns_both(no_checkers: None) -> None:
    runtime, _ = build_runtime()
    wf_a = await _register(runtime, _PAYLOAD)
    ids_a = _task_ids(runtime, wf_a)

    async with _client(runtime) as ac:
        resp = await ac.get(
            f"{PREFIX}/tasks", params=[("status", "PENDING"), ("status", "DONE")]
        )

    assert resp.status_code == 200
    body = resp.json()
    assert {t["task_id"] for t in body} == set(ids_a)


@pytest.mark.anyio
async def test_list_tasks_response_body_is_unchanged(no_checkers: None) -> None:
    runtime, _ = build_runtime()
    wf_a = await _register(runtime, _PAYLOAD)
    wf_b = await _register(runtime, _PAYLOAD)
    ids_a = _task_ids(runtime, wf_a)
    ids_b = _task_ids(runtime, wf_b)

    # Give one task a latest_update holding private fields and ssh/serve sub-dicts
    # so _sanitize_latest_update has real work to do.
    target = ids_a[0]
    runtime.mark_updated(
        target,
        {
            "progress": 0.5,
            "_secret": "hidden",
            "ssh": {"host": "h", "_token": "t"},
            "serve": {"url": "u", "_key": "k"},
        },
    )

    def _expected(
        workflow_ids: list[str] | None = None,
        statuses: list[str] | None = None,
        allowed: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        tasks = runtime.list_tasks(workflow_ids=workflow_ids, statuses=statuses)
        if allowed is not None:
            tasks = [task for task in tasks if task.task_id in allowed]
        for task in tasks:
            tasks_router._sanitize_latest_update(task)
        return [task.model_dump(mode="json", by_alias=True) for task in tasks]

    async with _client(runtime) as ac:
        # Unfiltered call returns every task across both workflows.
        resp = await ac.get(f"{PREFIX}/tasks")
        assert resp.status_code == 200
        assert resp.json() == _expected()
        assert {t["task_id"] for t in resp.json()} == set(ids_a) | set(ids_b)

        # workflow_id filter narrows to that workflow.
        resp = await ac.get(f"{PREFIX}/tasks", params={"workflow_id": wf_a})
        assert resp.status_code == 200
        assert resp.json() == _expected(workflow_ids=[wf_a])

        # A caller with restricted access sees only its allowed tasks.
        PERMISSION_CHECKERS.append(_ScopedChecker(frozenset({target})))
        try:
            resp = await ac.get(f"{PREFIX}/tasks")
        finally:
            PERMISSION_CHECKERS.clear()
        assert resp.status_code == 200
        assert resp.json() == _expected(allowed=frozenset({target}))


@pytest.mark.anyio
async def test_event_loop_stays_responsive_during_list(
    no_checkers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _ = build_runtime()
    wf_a = await _register(runtime, _PAYLOAD)
    await _register(runtime, _PAYLOAD)

    # Sleep inside the real TaskInfo build path, as a slow store would. The work
    # happens inside the thread, so the event loop stays free.
    orig_build = runtime._build_task_info_locked

    def slow_build(task_id: str, record: Any) -> Any:
        time.sleep(0.5)
        return orig_build(task_id, record)

    monkeypatch.setattr(runtime, "_build_task_info_locked", slow_build)

    app = FastAPI()
    app.state.logger = logging.getLogger("test.tasks_router")
    app.include_router(tasks_router.router, prefix=PREFIX)
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.dependency_overrides[authenticate_connection] = lambda: MagicMock(
        spec=PrincipalContext
    )

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        t0 = time.monotonic()
        list_task = asyncio.create_task(
            ac.get(f"{PREFIX}/tasks", params={"workflow_id": wf_a})
        )
        await asyncio.sleep(0.05)  # let the list request start and block
        ping_resp = await ac.get("/ping")
        elapsed = time.monotonic() - t0
        list_resp = await list_task

    assert ping_resp.status_code == 200
    # The trivial request completes while the slow list is still in flight, so the
    # whole exchange is far shorter than the store call's 0.5s block.
    assert elapsed < 0.3
    assert list_resp.status_code == 200


@pytest.mark.anyio
async def test_event_loop_stays_responsive_during_list_serialization(
    no_checkers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _ = build_runtime()
    wf_a = await _register(runtime, _PAYLOAD)
    await _register(runtime, _PAYLOAD)

    # The slow step is the serialization itself: dump_json sleeps, and a
    # concurrent request must still complete while it runs.
    orig_dump = tasks_router._TASK_LIST.dump_json

    def slow_dump(*args: Any, **kwargs: Any) -> bytes:
        time.sleep(0.5)
        return orig_dump(*args, **kwargs)

    monkeypatch.setattr(tasks_router._TASK_LIST, "dump_json", slow_dump)

    app = FastAPI()
    app.state.logger = logging.getLogger("test.tasks_router")
    app.include_router(tasks_router.router, prefix=PREFIX)
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.dependency_overrides[authenticate_connection] = lambda: MagicMock(
        spec=PrincipalContext
    )

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        t0 = time.monotonic()
        list_task = asyncio.create_task(
            ac.get(f"{PREFIX}/tasks", params={"workflow_id": wf_a})
        )
        await asyncio.sleep(0.05)  # let the list request start and block
        ping_resp = await ac.get("/ping")
        elapsed = time.monotonic() - t0
        list_resp = await list_task

    assert ping_resp.status_code == 200
    assert elapsed < 0.3
    assert list_resp.status_code == 200
