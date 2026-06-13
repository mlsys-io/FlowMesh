"""Runtime scheduling behavior for epoch gating and in-epoch ordering."""

import asyncio
import logging
import threading
from typing import Any, cast

from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime


class _WorkflowRegistryStub:
    async def register_workflow_async(self, workflow_id: str, tasks: list[Any]) -> None:
        return None

    def commit_transition(self, transition: Any) -> None:
        return None

    async def save_task_states_async(self, items: Any) -> None:
        return None

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        return None


class _WorkerRegistryStub:
    def __init__(self) -> None:
        self.published_interrupts: list[tuple[Any, Any]] = []

    def get_worker(self, worker_id: str) -> Any:
        return {"id": worker_id, "node_id": "nde-1"}

    def publish_interrupt(self, worker: Any, payload: Any) -> int:
        self.published_interrupts.append((worker, payload))
        return 0


def _register(runtime: TaskRuntime, payload: str) -> tuple[str, dict[str, str]]:
    """Register a workflow and return workflow id and graph-node task ids."""
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", payload, format="native")
    )
    by_node = {str(item.graph_node_name): item.task_id for item in results}
    return workflow_id, by_node


def test_runtime_epoch_mode_enforces_frontier_order() -> None:
    """
    Validate ordered epoch behavior in runtime.
    - only frontier epoch tasks become runnable.
    - in-epoch ordering follows position_in_epoch.
    - frontier advances after all tasks in the epoch complete.
    """
    runtime = TaskRuntime(
        cast(Any, _WorkflowRegistryStub()),
        cast(Any, _WorkerRegistryStub()),
        logging.getLogger("runtime-test"),
    )
    payload = """
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
    workflow_id, node_ids = _register(runtime, payload)
    stop_event = threading.Event()

    assert runtime._workflow_epoch_frontier[workflow_id] == 0

    first = runtime.next_ready(stop_event, timeout=0.01)
    second = runtime.next_ready(stop_event, timeout=0.01)
    assert first == node_ids["a"]
    assert second == node_ids["b"]

    runtime.mark_succeeded(node_ids["a"], None, {}, "2026-02-19T00:00:00Z")
    assert runtime._workflow_epoch_frontier[workflow_id] == 0

    runtime.mark_succeeded(node_ids["b"], None, {}, "2026-02-19T00:00:01Z")
    assert runtime._workflow_epoch_frontier[workflow_id] == 1

    third = runtime.next_ready(stop_event, timeout=0.01)
    assert third == node_ids["c"]

    runtime.mark_succeeded(node_ids["c"], None, {}, "2026-02-19T00:00:02Z")
    assert workflow_id not in runtime._workflow_epoch_frontier


def test_runtime_unordered_in_epoch_allows_any_order() -> None:
    """
    Validate unordered epoch behavior in runtime.
    - frontier gating still applies.
    - tasks inside the epoch can be scheduled in any order.
    """
    runtime = TaskRuntime(
        cast(Any, _WorkflowRegistryStub()),
        cast(Any, _WorkerRegistryStub()),
        logging.getLogger("runtime-test"),
    )
    payload = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: graph
  annotations:
    schedule_hint:
      node_schedule_in_epoch_order: false
      node_execution_order:
        - [a, b, c]
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
        spec:
          taskType: echo
"""
    workflow_id, node_ids = _register(runtime, payload)
    stop_event = threading.Event()

    assert runtime._workflow_epoch_frontier[workflow_id] == 0
    assert workflow_id not in runtime._workflow_in_epoch_order

    first = runtime.next_ready(stop_event, timeout=0.01)
    second = runtime.next_ready(stop_event, timeout=0.01)
    third = runtime.next_ready(stop_event, timeout=0.01)
    assert {first, second, third} == {node_ids["a"], node_ids["b"], node_ids["c"]}


def test_runtime_epoch_mode_fails_later_epochs_when_frontier_task_fails() -> None:
    """
    Validate failure handling across epochs.
    - failure in a frontier epoch fails tasks in later epochs.
    - impacted tasks are marked FAILED with a blocking reason.
    """
    runtime = TaskRuntime(
        cast(Any, _WorkflowRegistryStub()),
        cast(Any, _WorkerRegistryStub()),
        logging.getLogger("runtime-test"),
    )
    payload = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: graph
  annotations:
    schedule_hint:
      node_schedule_in_epoch_order: true
      node_execution_order:
        - [a]
        - [b, c]
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
        spec:
          taskType: echo
"""
    _, node_ids = _register(runtime, payload)
    stop_event = threading.Event()
    ready = runtime.next_ready(stop_event, timeout=0.01)
    assert ready == node_ids["a"]

    impacted, _, _ = runtime.mark_failed(
        node_ids["a"],
        None,
        {"error": "boom"},
        "2026-02-19T00:00:00Z",
    )
    impacted_ids = {task_id for task_id, _reason in impacted}

    b_record = runtime.get_record(node_ids["b"])
    c_record = runtime.get_record(node_ids["c"])

    assert impacted_ids == {node_ids["b"], node_ids["c"]}
    assert b_record is not None
    assert c_record is not None
    assert b_record.status == TaskStatus.FAILED
    assert c_record.status == TaskStatus.FAILED
    assert "Blocked by failed task" in str(b_record.error)


def test_cancel_workflow_marks_dispatched_task_cancelling_and_publishes_interrupt() -> (
    None
):
    worker_registry = _WorkerRegistryStub()
    runtime = TaskRuntime(
        cast(Any, _WorkflowRegistryStub()),
        cast(Any, worker_registry),
        logging.getLogger("runtime-test"),
    )
    payload = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: cancel
spec:
  stages:
    - name: shell
      spec:
        taskType: ssh
"""
    workflow_id, _ = _register(runtime, payload)
    task_id = next(iter(runtime.tasks))
    runtime.mark_started(task_id, "wkr-1", {}, "2026-02-19T00:00:00Z")

    cancelled = runtime.cancel_workflow(workflow_id)

    assert cancelled == [task_id]
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.status == TaskStatus.CANCELLING
    assert len(worker_registry.published_interrupts) == 1
    worker, interrupt = worker_registry.published_interrupts[0]
    assert worker["id"] == "wkr-1"
    assert interrupt.task_id == task_id
    assert interrupt.worker_id == "wkr-1"


def test_cancel_workflow_skips_interruptive_cancellation_for_merged_tasks() -> None:
    worker_registry = _WorkerRegistryStub()
    runtime = TaskRuntime(
        cast(Any, _WorkflowRegistryStub()),
        cast(Any, worker_registry),
        logging.getLogger("runtime-test"),
    )
    payload = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: merge-cancel
spec:
  stages:
    - name: parent
      spec:
        taskType: inference
    - name: child
      spec:
        taskType: inference
"""
    workflow_id, _ = _register(runtime, payload)
    task_ids = [
        task_id
        for task_id, record in runtime.tasks.items()
        if record.workflow_id == workflow_id
    ]
    parent_id, child_id = task_ids
    parent = runtime.get_record(parent_id)
    child = runtime.get_record(child_id)
    assert parent is not None
    assert child is not None

    parent.status = TaskStatus.DISPATCHED
    parent.assigned_worker = "wkr-1"
    parent.merged_children = [child_id]
    child.status = TaskStatus.DISPATCHED
    child.merged_parent_id = parent_id
    runtime._merge_children_map[parent_id] = [child_id]
    runtime._merge_parent_map[child_id] = parent_id

    cancelled = runtime.cancel_workflow(workflow_id)

    updated_parent = runtime.get_record(parent_id)
    updated_child = runtime.get_record(child_id)
    assert updated_parent is not None
    assert updated_child is not None
    assert cancelled == []
    assert updated_parent.status == TaskStatus.DISPATCHED
    assert updated_child.status == TaskStatus.DISPATCHED
    assert updated_parent.merged_children == [child_id]
    assert updated_child.merged_parent_id == parent_id
    assert len(worker_registry.published_interrupts) == 0
