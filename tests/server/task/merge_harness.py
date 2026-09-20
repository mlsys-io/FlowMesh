"""Shared helpers for runtime task-merge tests.

A `TaskRuntime` wired to a capturing workflow registry that records every
`commit_transition` call, so tests can assert what was persisted, in which
workflow, and in what order.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, cast

from server.registries.workflow import PersistedTask, WorkflowSched
from server.task.runtime import TaskRuntime


class CapturingWorkflowRegistry:
    """Records each `commit_transition` as a dict of its arguments."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "workflow_id": workflow_id,
                "record_ids": [p.record.task_id for p in records],
                "dispatched": list(dispatched),
                "done": list(done),
                "failed": list(failed),
                "cancelled": list(cancelled),
            }
        )

    async def save_task_states_async(self, items: Any) -> None:
        return None

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        return None


class WorkerRegistryStub:
    def get_worker(self, worker_id: str) -> Any:
        return {"id": worker_id, "node_id": "nde-1"}


def build_runtime(
    name: str = "merge-test",
) -> tuple[TaskRuntime, CapturingWorkflowRegistry]:
    registry = CapturingWorkflowRegistry()
    runtime = TaskRuntime(
        cast(Any, registry), cast(Any, WorkerRegistryStub()), logging.getLogger(name)
    )
    return runtime, registry


def register(runtime: TaskRuntime, payload: str) -> tuple[str, dict[str, str]]:
    """Register a workflow; return its id and a {graph_node_name: task_id} map."""
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", payload, format="native")
    )
    return workflow_id, {str(r.graph_node_name): r.task_id for r in results}
