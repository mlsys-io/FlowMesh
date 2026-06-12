"""Durable persistence and restart rehydration of TaskRuntime."""

import logging
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.registries.workflow import PersistedTask, WorkflowSched
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime


class FakeWorkflowRegistry:
    """In-memory registry that round-trips state through the real model JSON."""

    def __init__(self) -> None:
        self.task_blobs: dict[str, str] = {}
        self.sched: dict[str, str] = {}
        self.workflow_task_ids: dict[str, list[str]] = {}

    async def register_workflow_async(self, workflow_id: str, tasks: list[Any]) -> None:
        self.workflow_task_ids[workflow_id] = [t.task_id for t in tasks]

    def get_workflow_ids(self) -> set[str]:
        return set(self.workflow_task_ids)

    async def get_workflow_ids_async(self) -> set[str]:
        return self.get_workflow_ids()

    def get_workflow_record(self, workflow_id: str) -> Any:
        ids = self.workflow_task_ids.get(workflow_id)
        return SimpleNamespace(task_ids=list(ids)) if ids is not None else None

    async def get_workflow_record_async(self, workflow_id: str) -> Any:
        return self.get_workflow_record(workflow_id)

    def save_task_states(self, items: list[PersistedTask]) -> None:
        for item in items:
            self.task_blobs[item.record.task_id] = item.model_dump_json()

    async def save_task_states_async(self, items: list[PersistedTask]) -> None:
        self.save_task_states(items)

    def load_task_state(self, task_id: str) -> PersistedTask | None:
        blob = self.task_blobs.get(task_id)
        return PersistedTask.model_validate_json(blob) if blob else None

    async def load_task_state_async(self, task_id: str) -> PersistedTask | None:
        return self.load_task_state(task_id)

    def save_workflow_sched(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        self.sched[workflow_id] = WorkflowSched(
            in_epoch_order=in_epoch_order, epoch_frontier=frontier
        ).model_dump_json()

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        self.save_workflow_sched(workflow_id, in_epoch_order, frontier)

    def load_workflow_sched(self, workflow_id: str) -> WorkflowSched | None:
        blob = self.sched.get(workflow_id)
        return WorkflowSched.model_validate_json(blob) if blob else None

    async def load_workflow_sched_async(self, workflow_id: str) -> WorkflowSched | None:
        return self.load_workflow_sched(workflow_id)

    def mark_task_dispatched(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_done(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_failed(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_pending(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_cancelled(self, workflow_id: str, *task_ids: str) -> None: ...


class _WorkerRegistryStub:
    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(id=worker_id, node_id="nde-1")

    def publish_interrupt(self, *args: Any) -> int:
        return 0


def _runtime(registry: FakeWorkflowRegistry) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        logging.getLogger("rehydrate-test"),
    )


async def _register(runtime: TaskRuntime, payload: str) -> tuple[str, dict[str, str]]:
    workflow_id, results = await runtime.register(
        "owner", "org", payload, format="native"
    )
    return workflow_id, {str(r.graph_node_name): r.task_id for r in results}


GRAPH = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: graph
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
      - name: b
        dependsOn: [a]
        spec:
          taskType: echo
"""

EPOCH_GRAPH = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: graph
  annotations:
    schedule_hint:
      node_schedule_in_epoch_order: true
      node_execution_order:
        - [a, b]
        - [c]
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
      - name: b
        spec:
          taskType: echo
      - name: c
        dependsOn: [a]
        spec:
          taskType: echo
"""


@pytest.mark.anyio
async def test_persisted_task_round_trips_failed_workers_and_deps() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    record = runtime.get_record(b)
    assert record is not None
    record.failed_workers.append("wkr-dead")

    pt = PersistedTask(record=record, depends_on={a}, epoch_index=2)
    restored = PersistedTask.model_validate_json(pt.model_dump_json())

    # failed_workers is exclude=True on TaskRecord but must survive persistence;
    # depends_on round-trips through a JSON list back to a set.
    assert restored.record.failed_workers == ["wkr-dead"]
    assert restored.depends_on == {a}
    assert restored.epoch_index == 2
    assert restored.record.task_id == b


@pytest.mark.anyio
async def test_rehydrate_restores_completed_and_ready_state() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))
    runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    restored = _runtime(registry)
    assert await restored.rehydrate() == 1

    record_a = restored.get_record(a)
    record_b = restored.get_record(b)
    assert record_a is not None
    assert record_a.status == TaskStatus.DONE
    assert record_b is not None
    assert record_b.status == TaskStatus.PENDING

    # b's only dependency completed, so it is the sole ready task.
    assert restored.ready_queue_length() == 1
    stop = threading.Event()
    assert restored.next_ready(stop, timeout=0.01) == b
    assert restored.ready_queue_length() == 0


@pytest.mark.anyio
async def test_rehydrate_keeps_in_flight_task_dispatched() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-9", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))

    restored = _runtime(registry)
    await restored.rehydrate()

    record = restored.get_record(a)
    assert record is not None
    assert record.status == TaskStatus.DISPATCHED
    assert record.assigned_worker == "wkr-9"
    # An in-flight task is not re-queued; its completion arrives via the stream.
    assert restored.ready_queue_length() == 0


@pytest.mark.anyio
async def test_rehydrate_restores_epoch_frontier() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, EPOCH_GRAPH)

    runtime.mark_succeeded(ids["a"], None, {}, "2026-06-01T00:00:00Z")
    runtime.mark_succeeded(ids["b"], None, {}, "2026-06-01T00:00:01Z")
    assert runtime._workflow_epoch_frontier[workflow_id] == 1

    restored = _runtime(registry)
    await restored.rehydrate()

    assert restored._workflow_epoch_frontier[workflow_id] == 1
    stop = threading.Event()
    assert restored.next_ready(stop, timeout=0.01) == ids["c"]


@pytest.mark.anyio
async def test_mark_succeeded_is_idempotent_under_replay() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))
    runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    # A replayed completion must not re-apply.
    assert runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z") == []

    # b is enqueued exactly once despite the replay.
    assert runtime.ready_queue_length() == 1
    stop = threading.Event()
    assert runtime.next_ready(stop, timeout=0.01) == b
    assert runtime.ready_queue_length() == 0


@pytest.mark.anyio
async def test_rehydrated_in_flight_task_is_protected_then_released() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-7", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))

    restored = _runtime(registry)
    await restored.rehydrate()

    # Within the grace window the worker's rehydrated task is protected; with a
    # zero window it is not.
    assert restored.has_rehydrated_in_flight("wkr-7", 600.0) is True
    assert restored.has_rehydrated_in_flight("wkr-7", 0.0) is False
    assert restored.has_rehydrated_in_flight("wkr-other", 600.0) is False


@pytest.mark.anyio
async def test_rehydrated_protection_clears_on_completion() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-7", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))

    restored = _runtime(registry)
    await restored.rehydrate()
    assert restored.has_rehydrated_in_flight("wkr-7", 600.0) is True

    restored.mark_succeeded(a, "wkr-7", {}, "2026-06-01T00:00:00Z")
    assert restored.has_rehydrated_in_flight("wkr-7", 600.0) is False


@pytest.mark.anyio
async def test_recover_clears_rehydrated_protection() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-7", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))

    restored = _runtime(registry)
    await restored.rehydrate()
    assert restored.recover_tasks_for_worker("wkr-7") == [a]
    assert restored.has_rehydrated_in_flight("wkr-7", 600.0) is False


@pytest.mark.anyio
async def test_terminal_task_does_not_regress_on_replayed_dispatch_or_start() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))
    runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    # A replayed dispatch / start / progress update must not move a's status
    # back to DISPATCHED.
    runtime.mark_dispatched(a, cast(Any, worker))
    runtime.mark_started(a, "wkr-1", {}, "2026-06-01T00:00:01Z")
    runtime.mark_updated(a, {"note": "stale"})

    record = runtime.get_record(a)
    assert record is not None
    assert record.status == TaskStatus.DONE
    assert record.latest_update is None
