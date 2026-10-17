"""Workflow-related models."""

from typing import Any

from pydantic import BaseModel, model_validator

from .common import TaskStatus, WorkflowStatus


class WorkflowSubmitTaskEntry(BaseModel):
    task_id: str
    status: TaskStatus | None = None
    assigned_worker: str | None = None
    topic: str | None = None
    waiting_on: list[str] | None = None
    depends_on: list[str] | None = None
    attempts: int | None = None
    max_attempts: int | None = None
    load: int | None = None


class WorkflowSubmitResponse(BaseModel):
    ok: bool
    workflow_id: str
    count: int
    tasks: list[WorkflowSubmitTaskEntry]


class WorkflowValidateTaskEntry(BaseModel):
    task_id: str
    graph_node_name: str | None = None
    depends_on: list[str]


class WorkflowValidateResponse(BaseModel):
    ok: bool
    count: int
    tasks: list[WorkflowValidateTaskEntry]


class Workflow(BaseModel):
    workflow_id: str
    task_ids: list[str]
    submitted_at: str
    updated_at: str
    status: WorkflowStatus
    dispatched_tasks: list[str]
    completed_tasks: list[str]
    failed_tasks: list[str]
    cancelled_tasks: list[str]


_COUNTED_LISTS = {
    "task_count": "task_ids",
    "dispatched_count": "dispatched_tasks",
    "completed_count": "completed_tasks",
    "failed_count": "failed_tasks",
    "cancelled_count": "cancelled_tasks",
}


class WorkflowStatusView(BaseModel):
    """A workflow's status, timestamps and per-state task counts, without its task
    ID lists."""

    workflow_id: str
    status: WorkflowStatus
    submitted_at: str
    updated_at: str
    task_count: int
    dispatched_count: int
    completed_count: int
    failed_count: int
    cancelled_count: int

    @model_validator(mode="before")
    @classmethod
    def _count_task_lists(cls, data: Any) -> Any:
        # A server that ignores `view=status` returns the full workflow; count its
        # task ID lists instead.
        if not isinstance(data, dict):
            return data
        counts = {
            count: len(data[ids])
            for count, ids in _COUNTED_LISTS.items()
            if count not in data and isinstance(data.get(ids), list)
        }
        return {**data, **counts}
