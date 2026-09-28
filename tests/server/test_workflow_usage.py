"""Tests for the workflow-level usage sum on GET /workflows/{id}.

Usage is captured once at result ingest (``ingest_result``) and stored per task
in the workflow registry's Redis; ``GET /workflows/{id}`` sums the stored
values without opening any result file.
"""

import logging
from collections.abc import Iterator
from typing import Any, cast
from unittest import mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from lumid_hooks import PrincipalContext, ResourceRef

from server.app_state import get_logger, get_workflow_registry
from server.auth.security import authenticate_connection
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
    EchoResult,
    GenerationUsage,
    InferenceResult,
    ResultEnvelope,
)


class _FakePipeline:
    """Records operations issued on a pipeline; execute() applies deletes."""

    def __init__(self, store: dict[str, str]) -> None:
        self._store = store
        self.deletes: list[str] = []

    async def __aenter__(self) -> "_FakePipeline":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def srem(self, key: str, *members: str) -> int:
        return 0

    def delete(self, *keys: str) -> int:
        self.deletes.extend(keys)
        return len(keys)

    async def execute(self) -> None:
        for key in self.deletes:
            self._store.pop(key, None)


class _FakeRedis:
    """In-memory stand-in for the registry's RedisClient asyncio surface."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.pipelines: list[_FakePipeline] = []

    async def set_value(self, key: str, value: str) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)

    async def mget(self, keys: list[str]) -> list[str | None]:
        return [self.store.get(k) for k in keys]

    async def hash_getall(self, key: str) -> dict[str, Any]:
        return {}

    def control_pipeline(self) -> _FakePipeline:
        pipe = _FakePipeline(self.store)
        self.pipelines.append(pipe)
        return pipe


class _FakeSyncRedis:
    def __init__(self, store: dict[str, str]) -> None:
        self.store = store
        self.records: dict[str, dict[str, Any]] = {}

    def hash_getall(self, key: str) -> dict[str, Any]:
        return self.records.get(key, {})

    def set_value(self, key: str, value: str) -> None:
        self.store[key] = value


class _FakeRedisClient:
    def __init__(self, fake: _FakeRedis | None = None) -> None:
        self.asyncio = fake if fake is not None else _FakeRedis()
        self.sync = _FakeSyncRedis(self.asyncio.store)


def _registry() -> WorkflowRegistry:
    return WorkflowRegistry(cast(Any, _FakeRedisClient()))


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


def _inference_result() -> InferenceResult:
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
    )


def _inference_usage() -> APIUsage:
    usage = results_router._task_usage_from_envelope(
        ResultEnvelope(task_id="tsk-inf", result=_inference_result())
    )
    assert isinstance(usage, APIUsage)
    return usage


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
async def test_workflow_usage_sums_api_and_inference_tasks() -> None:
    """Two API tasks plus one vLLM task sum exactly into one workflow figure."""
    registry = _registry()
    await registry.save_task_usage_async("tsk-api-1", _api_usage())
    await registry.save_task_usage_async(
        "tsk-api-2", _api_usage(prompt_tokens=5, completion_tokens=1, calls=1)
    )
    await registry.save_task_usage_async("tsk-inf", _inference_usage())

    usage = await _sum(registry, ["tsk-api-1", "tsk-api-2", "tsk-inf"])
    assert usage is not None
    assert usage.prompt_tokens == 135
    assert usage.completion_tokens == 63
    assert usage.reasoning_tokens == 4
    assert usage.calls == 8
    assert usage.retries == 2
    assert usage.truncated_calls == 2
    assert usage.wall_sec == 6.0


@pytest.mark.anyio
async def test_workflow_usage_echo_task_contributes_nothing() -> None:
    """An echo task (no usage) contributes nothing and is not a failure."""
    registry = _registry()
    await registry.save_task_usage_async(
        "tsk-api", _api_usage(prompt_tokens=5, calls=1)
    )
    await registry.save_task_usage_async("tsk-echo", None)

    usage = await _sum(registry, ["tsk-api", "tsk-echo"])
    assert usage is not None
    assert usage.prompt_tokens == 5
    assert usage.calls == 1


@pytest.mark.anyio
async def test_workflow_usage_fails_closed_on_missing_result() -> None:
    """A completed task with no recorded usage makes the workflow usage null."""
    registry = _registry()
    await registry.save_task_usage_async("tsk-api", _api_usage())

    assert await _sum(registry, ["tsk-api", "tsk-missing"]) is None


@pytest.mark.anyio
async def test_workflow_usage_all_zero_when_no_task_reports_usage() -> None:
    """A workflow of only no-usage tasks returns an all-zero usage object."""
    registry = _registry()
    await registry.save_task_usage_async("tsk-echo", None)

    usage = await _sum(registry, ["tsk-echo"])
    assert usage is not None
    assert usage.prompt_tokens == 0
    assert usage.calls == 0
    assert usage.wall_sec == 0.0


def test_task_usage_from_envelope_maps_vllm_latency() -> None:
    """A vLLM inference task maps wall_sec from latency_sec, not zero."""
    usage = _inference_usage()
    assert usage is not None
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.reasoning_tokens == 0
    assert usage.calls == 4
    assert usage.wall_sec == 1.0


@pytest.mark.anyio
async def test_ingest_result_persists_usage_once(
    allow_all_permissions: None, tmp_path: Any
) -> None:
    """ingest_result captures and stores a task's usage at ingest time."""
    registry = _registry()
    envelope = ResultEnvelope(
        task_id="tsk-inf",
        result=_inference_result(),
    )
    runtime = mock.Mock()
    runtime.get_record.return_value = None
    await results_router.ingest_result(
        envelope=envelope,
        principal=_principal(),
        runtime=runtime,
        event_monitor=mock.Mock(),
        results_dir=tmp_path,
        registry=registry,
        logger=logging.getLogger("test"),
    )

    usages = await registry.load_task_usages_async("tsk-inf")
    usage = usages["tsk-inf"]
    assert isinstance(usage, APIUsage)
    assert usage.wall_sec == 1.0


def test_task_usage_from_envelope_none_for_no_usage_type() -> None:
    """An echo task has no usage contribution."""
    envelope = ResultEnvelope(
        task_id="tsk-echo", result=EchoResult(ok=True, items=[], count=0)
    )
    assert results_router._task_usage_from_envelope(envelope) is None


@pytest.mark.anyio
async def test_get_workflow_does_not_open_result_files(
    allow_all_permissions: None,
) -> None:
    """GET /workflows/{id} sums stored usage without reading any result file."""
    registry = _registry()
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


@pytest.mark.anyio
async def test_unregister_workflows_deletes_usage_keys() -> None:
    """Unregistering a workflow deletes its tasks' usage keys (no leak)."""
    fake = _FakeRedis()
    client = _FakeRedisClient(fake)
    client.sync.records["workflow:wfl-1"] = {
        "workflow_id": "wfl-1",
        "task_ids": '["tsk-1","tsk-2"]',
    }
    registry = WorkflowRegistry(cast(Any, client))
    await registry.save_task_usage_async("tsk-1", _api_usage())
    await registry.save_task_usage_async("tsk-2", None)

    await registry.unregister_workflows_async("wfl-1")

    assert fake.pipelines
    deleted = set()
    for pipe in fake.pipelines:
        deleted.update(pipe.deletes)
    assert "task:tsk-1:usage" in deleted
    assert "task:tsk-2:usage" in deleted


@pytest.mark.anyio
async def test_workflow_usage_parent_plus_clone_sums_to_parent() -> None:
    """A parent plus a mirrored clone sums to exactly the parent's usage."""
    registry = _registry()
    await registry.save_task_usage_async("tsk-parent", _api_usage())
    await registry.save_task_usage_async("tsk-clone", None)

    usage = await _sum(registry, ["tsk-parent", "tsk-clone"])
    assert usage is not None
    assert usage.prompt_tokens == 30
    assert usage.calls == 3
    assert usage.wall_sec == 2.5


@pytest.mark.anyio
async def test_workflow_usage_fails_closed_on_unmappable_usage() -> None:
    """A model-calling task with unmappable usage makes the workflow null."""
    registry = _registry()
    await registry.save_task_usage_async("tsk-api", _api_usage())
    await registry.save_task_usage_async("tsk-inf", UNKNOWN_USAGE)

    assert await _sum(registry, ["tsk-api", "tsk-inf"]) is None


def test_task_usage_from_envelope_unknown_for_inference_without_usage() -> None:
    """An inference result with usage None maps to UNKNOWN_USAGE, not None."""
    envelope = ResultEnvelope(
        task_id="tsk-inf",
        result=InferenceResult(ok=True, model="m", items=[], usage=None),
    )
    assert results_router._task_usage_from_envelope(envelope) is UNKNOWN_USAGE


@pytest.mark.anyio
async def test_mirror_task_results_records_no_usage_for_clone(
    tmp_path: Any,
) -> None:
    """A mirrored clone records the no-usage marker (it made no calls)."""
    from server.services.monitoring import EventMonitor

    registry = _registry()
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
    # Seed the parent's result directory so the mirror clones it.
    parent_dir = tmp_path / "tsk-parent"
    parent_dir.mkdir(parents=True)
    (parent_dir / "results.json").write_text("{}", encoding="utf-8")

    monitor.mirror_task_results("tsk-parent", ["tsk-clone"])

    usages = await registry.load_task_usages_async("tsk-clone")
    assert "tsk-clone" in usages
    assert usages["tsk-clone"] is None
