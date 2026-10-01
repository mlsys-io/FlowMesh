"""Tests for the workflow-level usage sum on GET /workflows/{id}.

Usage is captured once at result ingest (``ingest_result``) and stored per task
in the workflow registry's Redis; ``GET /workflows/{id}`` sums the stored
values without opening any result file.
"""

import logging
from collections.abc import Iterator
from typing import Any, cast
from unittest import mock

import fakeredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from lumid_hooks import PrincipalContext, ResourceRef

from server.app_state import get_logger, get_workflow_registry
from server.auth.security import authenticate_connection
from server.clients.redis import AsyncRedisClient, RedisClient, SyncRedisClient
from server.hooks import PERMISSION_CHECKERS
from server.registries.workflow import (
    UNKNOWN_USAGE,
    Workflow,
    WorkflowRegistry,
    WorkflowStatus,
)
from server.routers.v1 import results as results_router
from server.routers.v1 import workflows as workflows_router
from shared.schemas.result import (
    APIUsage,
    GenerationUsage,
    InferenceResult,
    ResultEnvelope,
)


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def registry(server: fakeredis.FakeServer) -> WorkflowRegistry:
    sync = SyncRedisClient.__new__(SyncRedisClient)
    sync._control = fakeredis.FakeRedis(server=server, decode_responses=True)
    async_client = AsyncRedisClient.__new__(AsyncRedisClient)
    cast(Any, async_client)._control = fakeredis.FakeAsyncRedis(
        server=server, decode_responses=True
    )
    client = RedisClient.__new__(RedisClient)
    client.sync = sync
    client.asyncio = async_client
    return WorkflowRegistry(client)


def _principal() -> PrincipalContext:
    return PrincipalContext(
        principal_id="p-1",
        org_id="org",
        external_id="ext",
        principal_type="user",
        scopes=[],
    )


class _AllowAllChecker:
    name = "allow-all"

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
        return None


@pytest.fixture
def allow_all_permissions() -> Iterator[None]:
    PERMISSION_CHECKERS.append(_AllowAllChecker())
    try:
        yield
    finally:
        PERMISSION_CHECKERS.clear()


def _api_usage(**overrides: Any) -> APIUsage:
    base: dict[str, Any] = dict(
        prompt_tokens=30,
        completion_tokens=12,
        reasoning_tokens=2,
        calls=3,
        failures=0,
        retries=1,
        truncated_calls=1,
        wall_sec=2.5,
    )
    base.update(overrides)
    return APIUsage(**base)


def _inference_usage() -> APIUsage:
    result = InferenceResult(
        ok=True,
        model="m",
        items=[],
        usage=GenerationUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            num_requests=4,
            latency_sec=1.0,
        ),
    )
    usage = results_router._task_usage_from_envelope(
        ResultEnvelope(task_id="tsk-inf", result=result)
    )
    assert isinstance(usage, APIUsage)
    return usage


def _merged_parent() -> InferenceResult:
    """A merged vLLM parent: batch total usage plus two children's shares."""
    child = lambda pt, ct, nr: InferenceResult(  # noqa: E731
        ok=True,
        model="m",
        items=[],
        usage=GenerationUsage(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            num_requests=nr,
            latency_sec=1.0,
        ),
    )
    return InferenceResult(
        ok=True,
        model="m",
        items=[],
        usage=GenerationUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            num_requests=4,
            latency_sec=1.0,
        ),
        children={
            "tsk-child-1": child(30, 20, 2),
            "tsk-child-2": child(10, 5, 1),
        },
    )


def _workflow(completed: list[str]) -> Workflow:
    return Workflow(
        workflow_id="wfl-1",
        task_ids=completed,
        submitted_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        status=WorkflowStatus.DONE,
        dispatched_tasks=[],
        completed_tasks=completed,
        failed_tasks=[],
        cancelled_tasks=[],
    )


async def _sum(registry: WorkflowRegistry, completed: list[str]) -> APIUsage | None:
    usages = await registry.load_task_usages_async(*completed)
    return workflows_router._sum_usage(usages, completed, logging.getLogger("test"))


@pytest.mark.anyio
async def test_usage_sums_api_and_inference_tasks(registry: WorkflowRegistry) -> None:
    """API and vLLM task usage sum exactly into one workflow figure."""
    await registry.save_task_usage_async("tsk-api", _api_usage())
    await registry.save_task_usage_async("tsk-inf", _inference_usage())

    usage = await _sum(registry, ["tsk-api", "tsk-inf"])
    assert usage is not None
    assert usage.prompt_tokens == 130
    assert usage.completion_tokens == 62
    assert usage.reasoning_tokens == 2
    assert usage.calls == 7
    assert usage.retries == 1
    assert usage.truncated_calls == 1
    assert usage.wall_sec == 3.5


@pytest.mark.anyio
async def test_no_model_task_contributes_nothing(registry: WorkflowRegistry) -> None:
    """A task that calls no model contributes nothing and is not a failure."""
    await registry.save_task_usage_async("tsk-api", _api_usage())
    await registry.save_task_usage_async("tsk-echo", None)

    usage = await _sum(registry, ["tsk-api", "tsk-echo"])
    assert usage is not None
    assert usage.prompt_tokens == 30
    assert usage.calls == 3


@pytest.mark.anyio
@pytest.mark.parametrize("unmappable", [False, True])
async def test_missing_or_unmappable_usage_makes_sum_null(
    registry: WorkflowRegistry, unmappable: bool
) -> None:
    """A completed model-calling task with missing or unmappable usage nulls the sum."""
    await registry.save_task_usage_async("tsk-api", _api_usage())
    if unmappable:
        await registry.save_task_usage_async("tsk-inf", UNKNOWN_USAGE)

    assert await _sum(registry, ["tsk-api", "tsk-inf"]) is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "order",
    [
        ["tsk-parent", "tsk-child-1", "tsk-child-2"],
        ["tsk-child-1", "tsk-child-2", "tsk-parent"],
    ],
)
async def test_merged_parent_plus_children_sums_to_batch_total(
    registry: WorkflowRegistry, order: list[str]
) -> None:
    """A merged parent and its children sum to the batch total in either order."""
    parent = _merged_parent()
    parent_usage = results_router._task_usage_from_envelope(
        ResultEnvelope(task_id="tsk-parent", result=parent)
    )
    assert isinstance(parent_usage, APIUsage)
    await registry.save_task_usage_async("tsk-parent", parent_usage)
    for child_id in ("tsk-child-1", "tsk-child-2"):
        child_usage = results_router._task_usage_from_envelope(
            ResultEnvelope(task_id=child_id, result=parent.children[child_id])
        )
        assert isinstance(child_usage, APIUsage)
        await registry.save_task_usage_async(child_id, child_usage)

    usage = await _sum(registry, order)
    assert usage is not None
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.calls == 4


@pytest.mark.anyio
async def test_mirrored_child_records_no_usage(
    registry: WorkflowRegistry, tmp_path: Any
) -> None:
    """A mirrored child records the no-usage marker (it made no calls)."""
    from server.services.monitoring import EventMonitor

    runtime = mock.Mock()
    runtime.get_record.return_value = None
    monitor = EventMonitor(
        redis_client=mock.Mock(),
        logger=logging.getLogger("test"),
        runtime=runtime,
        dispatcher=mock.Mock(),
        worker_registry=mock.Mock(),
        node_registry=mock.Mock(),
        metrics_recorder=mock.Mock(),
        watchdog=mock.Mock(),
        results_dir=tmp_path,
        workflow_registry=registry,
    )
    parent_dir = tmp_path / "tsk-parent"
    parent_dir.mkdir(parents=True)
    (parent_dir / "results.json").write_text("{}", encoding="utf-8")

    monitor.mirror_task_results("tsk-parent", ["tsk-clone"])

    usages = await registry.load_task_usages_async("tsk-clone")
    assert usages["tsk-clone"] is None


@pytest.mark.anyio
async def test_unregister_deletes_usage_keys(
    registry: WorkflowRegistry, server: fakeredis.FakeServer
) -> None:
    """Unregistering a workflow deletes its tasks' usage keys (no leak)."""
    rds = fakeredis.FakeRedis(server=server, decode_responses=True)
    rds.hset("workflow:wfl-1", "workflow_id", "wfl-1")
    rds.hset("workflow:wfl-1", "task_ids", '["tsk-1","tsk-2"]')
    await registry.save_task_usage_async("tsk-1", _api_usage())
    await registry.save_task_usage_async("tsk-2", None)

    await registry.unregister_workflows_async("wfl-1")

    assert rds.exists("task:tsk-1:usage") == 0
    assert rds.exists("task:tsk-2:usage") == 0


@pytest.mark.anyio
async def test_get_workflow_does_not_open_result_files(
    registry: WorkflowRegistry, allow_all_permissions: None
) -> None:
    """GET /workflows/{id} sums stored usage without reading any result file."""
    await registry.save_task_usage_async("tsk-api", _api_usage())
    await registry.save_task_usage_async("tsk-echo", None)

    app = FastAPI()
    app.state.logger = logging.getLogger("test.workflow_usage")
    app.include_router(workflows_router.router, prefix="/api/v1")
    app.dependency_overrides[get_workflow_registry] = lambda: registry
    app.dependency_overrides[get_logger] = lambda: logging.getLogger(
        "test.workflow_usage"
    )
    app.dependency_overrides[authenticate_connection] = lambda: _principal()

    with (
        mock.patch.object(
            registry,
            "get_workflow_async",
            return_value=_workflow(["tsk-api", "tsk-echo"]),
        ),
        mock.patch("shared.schemas.result.read_result") as read_result,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as ac:
            resp = await ac.get("/api/v1/workflows/wfl-1")
        read_result.assert_not_called()

    assert resp.status_code == 200
    assert resp.json()["usage"]["prompt_tokens"] == 30
