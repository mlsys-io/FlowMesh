"""Merge planning and single-child detachment.

Credential retention is gated at dispatch, not at merge time: ``plan_merge``
stays redaction-agnostic and a redacted child is failed individually during
merge resolution via ``drop_merged_child``, so a well-shaped task is never
sunk by a bad sibling it was batched with.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, cast

from server.registries.workflow import PersistedTask, WorkflowSched
from server.task.runtime import TaskRuntime

_WORKER = "wkr-1"

_PAYLOAD = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: merge-redaction
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
      - name: b
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
"""


class _WorkflowRegistryStub:
    async def register_workflow_async(self, workflow_id: str, tasks: list[Any]) -> None:
        return None

    def commit_transition(
        self,
        workflow_id: str,
        *,
        records: Sequence[PersistedTask] = (),
        dispatched: Sequence[str] = (),
        pending: Sequence[str] = (),
        done: Sequence[str] = (),
        failed: Sequence[str] = (),
        cancelled: Sequence[str] = (),
        sched: WorkflowSched | None = None,
    ) -> None:
        return None

    async def save_task_states_async(self, items: Any) -> None:
        return None

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        return None


class _WorkerRegistryStub:
    def get_worker(self, worker_id: str) -> Any:
        return {"id": worker_id, "node_id": "nde-1"}


def _make_runtime() -> tuple[TaskRuntime, dict[str, str]]:
    runtime = TaskRuntime(
        cast(Any, _WorkflowRegistryStub()),
        cast(Any, _WorkerRegistryStub()),
        logging.getLogger("runtime-merge-redaction-test"),
    )
    _, results = asyncio.run(
        runtime.register("owner", "org", _PAYLOAD, format="native")
    )
    node_ids = {str(item.graph_node_name): item.task_id for item in results}
    return runtime, node_ids


def test_clean_siblings_merge() -> None:
    runtime, node_ids = _make_runtime()
    merged = runtime.plan_merge(node_ids["a"], 2, _WORKER)
    assert merged == [node_ids["b"]]


def test_drop_merged_child_unlinks_without_requeue() -> None:
    runtime, node_ids = _make_runtime()
    parent, sibling = node_ids["a"], node_ids["b"]
    runtime.plan_merge(parent, 2, _WORKER)
    assert runtime._tasks[parent].merged_children == [sibling]
    assert runtime._merge_parent_map.get(sibling) == parent

    runtime.drop_merged_child(parent, sibling)

    assert runtime._tasks[parent].merged_children is None
    assert runtime._merge_parent_map.get(sibling) is None
    child = runtime._tasks[sibling]
    assert child.merged_parent_id is None
    # Not requeued: the caller terminates the child independently.
    assert sibling not in runtime._ready_index
