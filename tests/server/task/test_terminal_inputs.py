"""A terminal task drops oversized inputs from memory and from its persisted state."""

import asyncio
import logging
import threading
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from redis.exceptions import RedisError

from server.registries.workflow import PersistedTask
from server.services.monitoring import EventMonitor
from server.task.models import (
    TERMINAL_INPUT_MAX_BYTES,
    TERMINAL_TASK_STATUSES,
    TaskRecord,
    TaskStatus,
)
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent
from shared.tasks.specs import EchoSpecTemplate
from tests.server.task.merge_harness import WorkerRegistryStub, register
from tests.server.task.test_runtime_rehydrate import FakeWorkflowRegistry

DROPPED_SOURCE = "# source dropped after the task finished"


def _payload(n_items: int) -> str:
    items = "\n".join(f"            - row-{i}-" + "x" * 200 for i in range(n_items))
    return f"""
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: bulky
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
          data:
            type: list
            dataset_name: corpus-x
            items:
{items}
"""


def _runtime(registry: FakeWorkflowRegistry) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, WorkerRegistryStub()),
        logging.getLogger("terminal-inputs"),
    )


def _ready_task(n_items: int) -> tuple[TaskRuntime, FakeWorkflowRegistry, str]:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = register(runtime, _payload(n_items))
    assert runtime.next_ready(threading.Event(), timeout=0.01) == ids["a"]
    return runtime, registry, ids["a"]


def _data(record: TaskRecord) -> Any:
    spec = record.task.spec
    assert isinstance(spec, EchoSpecTemplate)
    return spec.data


def _stored(registry: FakeWorkflowRegistry, task_id: str) -> TaskRecord:
    return PersistedTask.model_validate_json(registry.task_blobs[task_id]).record


def _assert_dropped(record: TaskRecord) -> None:
    assert _data(record) is None
    assert record.source.startswith(DROPPED_SOURCE)


def test_live_task_keeps_its_inputs() -> None:
    runtime, _, tid = _ready_task(600)
    record = runtime.get_record(tid)
    assert record is not None and record.status not in TERMINAL_TASK_STATUSES
    assert not record.drop_terminal_inputs()
    assert _data(record) is not None
    assert len(record.source.encode()) > TERMINAL_INPUT_MAX_BYTES


def test_failed_task_drops_oversized_inputs() -> None:
    runtime, registry, tid = _ready_task(600)
    runtime.mark_failed(tid, None, {"error": "boom"}, "2026-10-11T00:00:00Z")
    record = runtime.get_record(tid)
    assert record is not None and record.status == TaskStatus.FAILED
    _assert_dropped(record)
    _assert_dropped(_stored(registry, tid))


def test_cancelled_task_drops_oversized_inputs() -> None:
    runtime, registry, tid = _ready_task(600)
    record = runtime.get_record(tid)
    assert record is not None
    runtime.cancel_workflow(record.workflow_id)
    assert record.status == TaskStatus.CANCELLED
    _assert_dropped(record)
    _assert_dropped(_stored(registry, tid))


def test_terminal_task_keeps_small_inputs() -> None:
    runtime, registry, tid = _ready_task(3)
    runtime.mark_failed(tid, None, {"error": "boom"}, "2026-10-11T00:00:00Z")
    for record in (runtime.get_record(tid), _stored(registry, tid)):
        assert record is not None
        assert _data(record) is not None
        assert "row-0-" in record.source


def test_rehydrate_drops_and_saves_back_oversized_terminal_inputs() -> None:
    runtime, registry, tid = _ready_task(600)
    record = runtime.get_record(tid)
    assert record is not None
    record.status = TaskStatus.DONE
    registry.task_blobs[tid] = PersistedTask(record=record).model_dump_json()

    restored = _runtime(registry)
    assert asyncio.run(restored.rehydrate()) == 1
    loaded = restored.get_record(tid)
    assert loaded is not None and loaded.status == TaskStatus.DONE
    _assert_dropped(loaded)
    _assert_dropped(_stored(registry, tid))


def test_succeeded_task_records_worker_cache_before_dropping_data() -> None:
    runtime, _, tid = _ready_task(600)
    worker_registry = MagicMock()
    monitor = EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("terminal-inputs"),
        runtime=runtime,
        dispatcher=MagicMock(),
        worker_registry=worker_registry,
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
    )
    monitor._handle_task_event(
        TaskEvent(type="TASK_SUCCEEDED", task_id=tid, worker_id="wkr-1", payload={})
    )
    record = runtime.get_record(tid)
    assert record is not None and record.status == TaskStatus.DONE
    _assert_dropped(record)
    worker_registry.record_worker_cache.assert_called_once_with(
        "wkr-1", models=[], datasets=["corpus-x"]
    )


async def _refuse_writes(items: list[PersistedTask]) -> None:
    raise RedisError("OOM command not allowed when used memory > 'maxmemory'")


def test_rehydrate_survives_a_failed_save_back(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, registry, tid = _ready_task(600)
    record = runtime.get_record(tid)
    assert record is not None
    record.status = TaskStatus.DONE
    registry.task_blobs[tid] = PersistedTask(record=record).model_dump_json()
    monkeypatch.setattr(registry, "save_task_states_async", _refuse_writes)

    restored = _runtime(registry)
    assert asyncio.run(restored.rehydrate()) == 1
    loaded = restored.get_record(tid)
    assert loaded is not None
    _assert_dropped(loaded)
