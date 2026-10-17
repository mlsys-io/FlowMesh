"""Router tests for `GET /api/v1/workflows` status view, ordering and `limit`."""

import logging
from collections.abc import Iterator, Sequence
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from lumid_hooks import ResourceRef

from server.app_state import get_workflow_registry
from server.auth.security import PrincipalContext, authenticate_connection
from server.hooks import PERMISSION_CHECKERS
from server.registries.workflow import Workflow, WorkflowStatus
from server.routers.v1 import workflows as workflows_router
from server.schemas.workflow import WorkflowStatusView

PREFIX = "/api/v1"


class _ScopedChecker:
    """Scopes workflow access to a fixed set of workflow ids."""

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


class _StubRegistry:
    def __init__(self, workflows: Sequence[Workflow]) -> None:
        self._workflows = {wf.workflow_id: wf for wf in workflows}

    async def get_workflow_async(self, workflow_id: str) -> Workflow | None:
        return self._workflows.get(workflow_id)

    async def get_workflow_ids_async(self) -> set[str]:
        return set(self._workflows)

    async def get_workflows_async(self, workflow_ids: Sequence[str]) -> list[Workflow]:
        return [self._workflows[wid] for wid in workflow_ids if wid in self._workflows]


def _workflow(
    workflow_id: str,
    submitted_at: str = "2026-10-12T10:00:00+00:00",
    status: WorkflowStatus = WorkflowStatus.DISPATCHED,
) -> Workflow:
    return Workflow(
        workflow_id=workflow_id,
        task_ids=[f"tsk-{i}" for i in range(10)],
        submitted_at=submitted_at,
        updated_at="2026-10-12T10:05:00+00:00",
        status=status,
        dispatched_tasks=["tsk-0"],
        completed_tasks=["tsk-1", "tsk-2"],
        failed_tasks=["tsk-3", "tsk-4", "tsk-5"],
        cancelled_tasks=["tsk-6", "tsk-7", "tsk-8", "tsk-9"],
    )


@pytest.fixture
def no_checkers() -> Iterator[None]:
    PERMISSION_CHECKERS.clear()
    yield
    PERMISSION_CHECKERS.clear()


def _client(*workflows: Workflow) -> AsyncClient:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.workflows_router")
    app.include_router(workflows_router.router, prefix=PREFIX)
    registry = _StubRegistry(workflows)
    app.dependency_overrides[get_workflow_registry] = lambda: registry
    app.dependency_overrides[authenticate_connection] = lambda: MagicMock(
        spec=PrincipalContext
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def test_status_view_shared_fields_match_workflow() -> None:
    for name, field in WorkflowStatusView.model_fields.items():
        if info := Workflow.model_fields.get(name):
            served = (field.annotation, field.description)
            assert served == (info.annotation, info.description), name


@pytest.mark.anyio
async def test_get_workflow_status_view_counts_tasks(no_checkers: None) -> None:
    async with _client(_workflow("wfl-1")) as ac:
        full = await ac.get(f"{PREFIX}/workflows/wfl-1")
        slim = await ac.get(f"{PREFIX}/workflows/wfl-1", params={"view": "status"})

    assert full.status_code == 200 and slim.status_code == 200
    f, s = full.json(), slim.json()
    assert "task_ids" in f
    assert set(s) == set(WorkflowStatusView.model_fields)
    shared = set(s) & set(f)
    assert shared == {"workflow_id", "status", "submitted_at", "updated_at"}
    assert {key: f[key] for key in shared} == {key: s[key] for key in shared}
    assert (
        s["task_count"],
        s["dispatched_count"],
        s["completed_count"],
        s["failed_count"],
        s["cancelled_count"],
    ) == (10, 1, 2, 3, 4)


@pytest.mark.anyio
async def test_get_workflow_unknown_view_is_rejected(no_checkers: None) -> None:
    async with _client(_workflow("wfl-1")) as ac:
        resp = await ac.get(f"{PREFIX}/workflows/wfl-1", params={"view": "bogus"})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_list_workflows_orders_by_submission_and_limits_to_newest(
    no_checkers: None,
) -> None:
    old = _workflow("wfl-b", submitted_at="2026-10-12T09:00:00+00:00")
    new = _workflow("wfl-a", submitted_at="2026-10-12T11:00:00+00:00")
    garbage = _workflow("wfl-c", submitted_at="not-a-timestamp")

    async with _client(new, old, garbage) as ac:
        every = await ac.get(f"{PREFIX}/workflows")
        newest = await ac.get(f"{PREFIX}/workflows", params={"limit": 1})

    assert [w["workflow_id"] for w in every.json()] == ["wfl-c", "wfl-b", "wfl-a"]
    assert [w["workflow_id"] for w in newest.json()] == ["wfl-a"]


@pytest.mark.anyio
async def test_list_workflows_limit_applies_after_filters(no_checkers: None) -> None:
    done = _workflow(
        "wfl-a", submitted_at="2026-10-12T09:00:00+00:00", status=WorkflowStatus.DONE
    )
    running = _workflow("wfl-b", submitted_at="2026-10-12T11:00:00+00:00")

    async with _client(done, running) as ac:
        resp = await ac.get(
            f"{PREFIX}/workflows",
            params={"status": "DONE", "limit": 1, "view": "status"},
        )

    assert [w["workflow_id"] for w in resp.json()] == ["wfl-a"]


@pytest.mark.anyio
async def test_list_workflows_limit_applies_after_permissions(
    no_checkers: None,
) -> None:
    older = _workflow("wfl-a", submitted_at="2026-10-12T09:00:00+00:00")
    newer = _workflow("wfl-b", submitted_at="2026-10-12T10:00:00+00:00")
    hidden = _workflow("wfl-c", submitted_at="2026-10-12T11:00:00+00:00")

    PERMISSION_CHECKERS.append(_ScopedChecker(frozenset({"wfl-a", "wfl-b"})))
    async with _client(older, newer, hidden) as ac:
        resp = await ac.get(f"{PREFIX}/workflows", params={"limit": 1})

    assert [w["workflow_id"] for w in resp.json()] == ["wfl-b"]


@pytest.mark.anyio
async def test_list_workflows_status_view(no_checkers: None) -> None:
    async with _client(_workflow("wfl-1"), _workflow("wfl-2")) as ac:
        slim = await ac.get(f"{PREFIX}/workflows", params={"view": "status"})

    assert slim.status_code == 200 and len(slim.json()) == 2
    assert all(set(w) == set(WorkflowStatusView.model_fields) for w in slim.json())


@pytest.mark.anyio
@pytest.mark.parametrize("limit", [0, 10_001])
async def test_list_workflows_rejects_out_of_range_limit(
    no_checkers: None, limit: int
) -> None:
    async with _client() as ac:
        resp = await ac.get(f"{PREFIX}/workflows", params={"limit": limit})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_list_workflows_unknown_view_is_rejected(no_checkers: None) -> None:
    async with _client() as ac:
        resp = await ac.get(f"{PREFIX}/workflows", params={"view": "bogus"})
    assert resp.status_code == 422
