"""Durable persistence and restart rehydration of TaskRuntime."""

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import Any, cast

from server.registries.workflow import (
    PersistedTask,
    WorkflowSched,
    _dump_task_state,
    _load_task_state,
)
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime


class FakeWorkflowRegistry:
    """In-memory registry that exercises the real (de)serialization helpers."""

    def __init__(self) -> None:
        self.task_blobs: dict[str, str] = {}
        self.sched: dict[str, tuple[bool, int]] = {}
        self.workflow_task_ids: dict[str, list[str]] = {}

    async def register_workflow_async(self, workflow_id: str, tasks: list[Any]) -> None:
        self.workflow_task_ids[workflow_id] = [t.task_id for t in tasks]

    def get_workflow_ids(self) -> set[str]:
        return set(self.workflow_task_ids)

    def get_workflow_record(self, workflow_id: str) -> Any:
        ids = self.workflow_task_ids.get(workflow_id)
        return SimpleNamespace(task_ids=list(ids)) if ids is not None else None

    def save_task_states(self, items: list[PersistedTask]) -> None:
        for item in items:
            self.task_blobs[item.record.task_id] = _dump_task_state(
                item.record, item.depends_on, item.epoch_index
            )

    async def save_task_states_async(self, items: list[PersistedTask]) -> None:
        self.save_task_states(items)

    def load_task_state(self, task_id: str) -> PersistedTask | None:
        return _load_task_state(self.task_blobs.get(task_id))

    def save_workflow_sched(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        self.sched[workflow_id] = (in_epoch_order, frontier)

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        self.sched[workflow_id] = (in_epoch_order, frontier)

    def load_workflow_sched(self, workflow_id: str) -> WorkflowSched | None:
        value = self.sched.get(workflow_id)
        return WorkflowSched(*value) if value is not None else None

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


def _register(runtime: TaskRuntime, payload: str) -> tuple[str, dict[str, str]]:
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", payload, format="native")
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


def test_rehydrate_restores_completed_and_ready_state() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))
    runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    restored = _runtime(registry)
    assert restored.rehydrate() == 1

    assert restored.get_record(a).status == TaskStatus.DONE  # type: ignore[union-attr]
    assert restored.get_record(b).status == TaskStatus.PENDING  # type: ignore[union-attr]

    # b's only dependency completed, so it is the sole ready task.
    assert restored.ready_queue_length() == 1
    stop = threading.Event()
    assert restored.next_ready(stop, timeout=0.01) == b
    assert restored.ready_queue_length() == 0


def test_rehydrate_keeps_in_flight_task_dispatched() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-9", node_id="nde-1")
    runtime.mark_dispatched(a, cast(Any, worker))

    restored = _runtime(registry)
    restored.rehydrate()

    record = restored.get_record(a)
    assert record is not None
    assert record.status == TaskStatus.DISPATCHED
    assert record.assigned_worker == "wkr-9"
    # An in-flight task is not re-queued; its completion arrives via the stream.
    assert restored.ready_queue_length() == 0


def test_rehydrate_restores_epoch_frontier() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = _register(runtime, EPOCH_GRAPH)

    runtime.mark_succeeded(ids["a"], None, {}, "2026-06-01T00:00:00Z")
    runtime.mark_succeeded(ids["b"], None, {}, "2026-06-01T00:00:01Z")
    assert runtime._workflow_epoch_frontier[workflow_id] == 1

    restored = _runtime(registry)
    restored.rehydrate()

    assert restored._workflow_epoch_frontier[workflow_id] == 1
    stop = threading.Event()
    assert restored.next_ready(stop, timeout=0.01) == ids["c"]


def test_mark_succeeded_is_idempotent_under_replay() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = _register(runtime, GRAPH)
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
