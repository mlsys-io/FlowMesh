"""The workflow registry and the Redis clients keep sync and async twins, and each
twin does what its counterpart does."""

import asyncio
import inspect
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import AsyncRedisClient, RedisClient, SyncRedisClient
from server.registries.workflow import WorkflowRegistry
from server.task.models import TaskRecord, TaskStatus
from shared.tasks import TaskEnvelopeTemplate, TaskType

# Both clients hand out their pipeline synchronously; the caller awaits its execute.
_CLIENT_SYNC_ON_BOTH = {"control_pipeline"}
# The async client keeps xrevrange_telemetry for the log routers; the sync twin is
# unused and was dropped.
_ASYNC_ONLY = {"xrevrange_telemetry"}


def _public(cls: type) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def test_every_registry_method_has_a_sync_and_an_async_twin() -> None:
    names = _public(WorkflowRegistry)
    asyncs = {name for name in names if name.endswith("_async")}
    syncs = names - asyncs

    assert {f"{name}_async" for name in syncs} - asyncs == set()
    assert {name.removesuffix("_async") for name in asyncs} - syncs == set()
    for name in asyncs:
        assert inspect.iscoroutinefunction(getattr(WorkflowRegistry, name)), name
    for name in syncs:
        assert not inspect.iscoroutinefunction(getattr(WorkflowRegistry, name)), name


def test_the_redis_clients_share_their_method_names() -> None:
    syncs, asyncs = _public(SyncRedisClient), _public(AsyncRedisClient)

    assert syncs == asyncs - _ASYNC_ONLY
    for name in asyncs - _CLIENT_SYNC_ON_BOTH:
        assert inspect.iscoroutinefunction(getattr(AsyncRedisClient, name)), name


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


def _registry(server: fakeredis.FakeServer) -> WorkflowRegistry:
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


def _task(workflow_id: str, task_id: str) -> TaskRecord:
    template = TaskEnvelopeTemplate.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "wf:task"},
            "spec": {"taskType": TaskType.API.value},
        }
    )
    return TaskRecord(
        task_id=task_id,
        workflow_id=workflow_id,
        owner_id="owner",
        source="raw",
        task=template,
        status=TaskStatus.PENDING,
        task_type="api",
        local_name="stage",
    )


def test_the_workflow_reads_agree(server: fakeredis.FakeServer) -> None:
    registry = _registry(server)
    registry.register_workflow("wfl-1", [_task("wfl-1", "tsk-1")])
    asyncio.run(registry.register_workflow_async("wfl-2", [_task("wfl-2", "tsk-2")]))

    assert registry.get_workflow("wfl-1") == asyncio.run(
        registry.get_workflow_async("wfl-1")
    )
    assert registry.get_workflows(["wfl-1", "wfl-2"]) == asyncio.run(
        registry.get_workflows_async(["wfl-1", "wfl-2"])
    )


def test_the_workflow_writes_agree(server: fakeredis.FakeServer) -> None:
    registry = _registry(server)
    registry.register_workflow("wfl-1", [_task("wfl-1", "tsk-1")])

    registry.save_workflow_sched("wfl-1", True, 3)
    assert registry.load_workflow_sched("wfl-1") == asyncio.run(
        registry.load_workflow_sched_async("wfl-1")
    )

    asyncio.run(registry.save_workflow_sched_async("wfl-1", False, 5))
    assert registry.load_workflow_sched("wfl-1") == asyncio.run(
        registry.load_workflow_sched_async("wfl-1")
    )


def test_unregistering_reads_through_the_async_client(
    server: fakeredis.FakeServer,
) -> None:
    registry = _registry(server)
    registry.register_workflow("wfl-1", [_task("wfl-1", "tsk-1")])

    asyncio.run(registry.unregister_workflows_async("wfl-1"))

    assert registry.get_workflow("wfl-1") is None
    assert registry.get_workflow_ids() == set()
