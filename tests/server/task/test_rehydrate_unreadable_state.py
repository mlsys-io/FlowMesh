"""Rehydration of workflows whose durable records cannot be read, against the real
workflow registry over a decoding fakeredis client."""

import logging
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import fakeredis
import pytest
from redis.exceptions import RedisError

from server.clients.redis import (
    task_state_key,
    workflow_failed_tasks_key,
    workflow_key,
    workflow_sched_key,
    workflow_tasks_key,
)
from server.registries.workflow import WorkflowRegistry, WorkflowStatus
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from tests.server.registries.test_workflow_registry_parity import _registry
from tests.server.task.test_runtime_rehydrate import (
    EPOCH_GRAPH,
    GRAPH,
    _register,
    _WorkerRegistryStub,
)


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def raw(server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=server)


def _runtime(registry: WorkflowRegistry) -> TaskRuntime:
    return TaskRuntime(
        registry,
        cast(Any, _WorkerRegistryStub()),
        logging.getLogger("rehydrate-unreadable-test"),
    )


async def _two_workflows(
    server: fakeredis.FakeServer,
) -> tuple[TaskRuntime, str, dict[str, str], dict[str, str]]:
    """Register a workflow to corrupt and a healthy one; return the runtime that
    registered them, the corrupt workflow's id and both node-to-task maps."""
    runtime = _runtime(_registry(server))
    workflow_id, ids = await _register(runtime, GRAPH)
    _, healthy = await _register(runtime, GRAPH)
    return runtime, workflow_id, ids, healthy


def _corrupt(
    raw: fakeredis.FakeRedis, key: str, damage: Callable[[bytes], bytes]
) -> None:
    blob = cast(bytes, raw.get(key))
    damaged = damage(blob)
    assert damaged != blob
    raw.set(key, damaged)


def _flip_high_bit(blob: bytes) -> bytes:
    middle = len(blob) // 2
    return blob[:middle] + bytes([blob[middle] ^ 0x80]) + blob[middle + 1 :]


def _assert_healthy_restored(runtime: TaskRuntime, healthy: dict[str, str]) -> None:
    record = runtime.get_record(healthy["a"])
    assert record is not None and record.status == TaskStatus.PENDING
    assert runtime.next_ready(threading.Event(), timeout=0.01) == healthy["a"]
    assert runtime.ready_queue_length() == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(
            lambda b: b.replace(b'"task_id":', b'"task_ie"*', 1), id="flipped-key"
        ),
        pytest.param(
            lambda b: b[: len(b) // 2] + b"\x07" + b[len(b) // 2 :], id="control-char"
        ),
        pytest.param(lambda b: b[: len(b) // 2], id="truncated"),
        pytest.param(_flip_high_bit, id="invalid-utf8"),
        pytest.param(
            lambda b: b.replace(b'"status":"PENDING"', b'"status":"PENDIOG"', 1),
            id="flipped-status",
        ),
        pytest.param(
            lambda b: b.replace(b'"task_id":"tsk-', b'"task_id":"tsk/', 1),
            id="flipped-task-id",
        ),
        pytest.param(
            lambda b: b.replace(b'"workflow_id":"wfl-', b'"workflow_id":"wfl/', 1),
            id="flipped-workflow-id",
        ),
    ],
)
async def test_unreadable_task_state_fails_its_workflow(
    server: fakeredis.FakeServer,
    raw: fakeredis.FakeRedis,
    damage: Callable[[bytes], bytes],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, workflow_id, ids, healthy = await _two_workflows(server)
    _corrupt(raw, task_state_key(ids["a"]), damage)

    registry = _registry(server)
    runtime = _runtime(registry)
    with caplog.at_level(logging.ERROR):
        assert await runtime.rehydrate() == 1

    assert runtime.get_record(ids["a"]) is None
    record_b = runtime.get_record(ids["b"])
    assert record_b is not None
    assert record_b.status == TaskStatus.FAILED
    assert record_b.error == "state_unreadable"

    workflow = await registry.get_workflow_async(workflow_id)
    assert workflow is not None
    assert workflow.status == WorkflowStatus.FAILED
    assert set(workflow.failed_tasks) == {ids["a"], ids["b"]}
    assert task_state_key(ids["a"]) in caplog.text
    assert workflow_id in caplog.text

    _assert_healthy_restored(runtime, healthy)


@pytest.mark.anyio
async def test_unreadable_schedule_fails_its_workflow_and_lets_in_flight_finish(
    server: fakeredis.FakeServer, raw: fakeredis.FakeRedis
) -> None:
    previous, workflow_id, ids, healthy = await _two_workflows(server)
    previous.mark_dispatched(
        ids["a"], cast(Any, SimpleNamespace(id="wkr-1", node_id="nde-1"))
    )
    _corrupt(raw, workflow_sched_key(workflow_id), _flip_high_bit)

    registry = _registry(server)
    runtime = _runtime(registry)
    assert await runtime.rehydrate() == 1

    record_a = runtime.get_record(ids["a"])
    record_b = runtime.get_record(ids["b"])
    assert record_a is not None and record_a.status == TaskStatus.DISPATCHED
    assert record_b is not None and record_b.status == TaskStatus.FAILED
    workflow = await registry.get_workflow_async(workflow_id)
    assert workflow is not None and workflow.status == WorkflowStatus.FAILED
    _assert_healthy_restored(runtime, healthy)

    runtime.mark_succeeded(ids["a"], "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert runtime.ready_queue_length() == 0
    workflow = await registry.get_workflow_async(workflow_id)
    assert workflow is not None and workflow.status == WorkflowStatus.FAILED
    assert workflow.failed_tasks == [ids["b"]]


@pytest.mark.anyio
async def test_in_flight_task_of_failed_workflow_is_retried_as_usual(
    server: fakeredis.FakeServer, raw: fakeredis.FakeRedis
) -> None:
    previous, workflow_id, ids, _ = await _two_workflows(server)
    previous.mark_dispatched(
        ids["a"], cast(Any, SimpleNamespace(id="wkr-1", node_id="nde-1"))
    )
    _corrupt(raw, workflow_sched_key(workflow_id), _flip_high_bit)

    runtime = _runtime(_registry(server))
    await runtime.rehydrate()
    assert runtime.recover_tasks_for_worker("wkr-1") == [ids["a"]]

    runtime.mark_pending(ids["a"], increment_retry=True)
    assert runtime.requeue(ids["a"])
    record_b = runtime.get_record(ids["b"])
    assert record_b is not None and record_b.status == TaskStatus.FAILED


@pytest.mark.anyio
async def test_ordered_task_without_its_position_fails_its_workflow(
    server: fakeredis.FakeServer, raw: fakeredis.FakeRedis
) -> None:
    workflow_id, ids = await _register(_runtime(_registry(server)), EPOCH_GRAPH)
    _corrupt(
        raw,
        task_state_key(ids["a"]),
        lambda b: b.replace(b'"position_in_epoch":', b'"position_in_epocx":', 1),
    )

    registry = _registry(server)
    assert await _runtime(registry).rehydrate() == 0
    workflow = await registry.get_workflow_async(workflow_id)
    assert workflow is not None and workflow.status == WorkflowStatus.FAILED
    assert set(workflow.failed_tasks) == set(ids.values())


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("task_ids", b'["tsk-', id="invalid-task-ids"),
        pytest.param("updated_at", b"2026-\xb1", id="invalid-utf8"),
    ],
)
async def test_unreadable_workflow_record_fails_its_unfinished_tasks(
    server: fakeredis.FakeServer, raw: fakeredis.FakeRedis, field: str, value: bytes
) -> None:
    _, workflow_id, ids, healthy = await _two_workflows(server)
    raw.hset(workflow_key(workflow_id), mapping={field: value})

    runtime = _runtime(_registry(server))
    assert await runtime.rehydrate() == 1

    for task_id in ids.values():
        record = runtime.get_record(task_id)
        assert record is not None and record.status == TaskStatus.FAILED
    assert raw.smembers(workflow_tasks_key(workflow_id)) == set()
    assert raw.smembers(workflow_failed_tasks_key(workflow_id)) == {
        task_id.encode() for task_id in ids.values()
    }
    _assert_healthy_restored(runtime, healthy)

    restarted = _runtime(_registry(server))
    await restarted.rehydrate()
    for task_id in ids.values():
        record = restarted.get_record(task_id)
        assert record is not None and record.status == TaskStatus.FAILED
    _assert_healthy_restored(restarted, healthy)


@pytest.mark.anyio
async def test_redis_error_still_fails_rehydration(
    server: fakeredis.FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _two_workflows(server)
    registry = _registry(server)

    async def _unavailable(*task_ids: str) -> None:
        raise RedisError("connection refused")

    monkeypatch.setattr(registry, "load_task_states_async", _unavailable)
    with pytest.raises(RedisError):
        await _runtime(registry).rehydrate()
