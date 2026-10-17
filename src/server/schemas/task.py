from typing import Any, Self

from pydantic import BaseModel, Field

from ..task.models import TaskInfo


class TaskStatusView(BaseModel):
    """A task's status, placement, timing, errors and latest update, without its
    spec or source."""

    task_id: str = Field(description="Task identifier.")
    workflow_id: str = Field(description="Workflow identifier.")
    status: str = Field(description="Task status.")
    task_type: str | None = Field(default=None, description="Task type.")
    category: str | None = Field(default=None, description="Task category.")
    assigned_worker: str | None = Field(
        default=None, description="Assigned worker identifier."
    )
    submitted_at: str = Field(description="Submission timestamp.")
    submitted_ts: float = Field(description="Submission epoch seconds.")
    dispatched_ts: float | None = Field(
        default=None, description="Dispatch timestamp (epoch seconds)."
    )
    started_ts: float | None = Field(
        default=None, description="Start timestamp (epoch seconds)."
    )
    finished_ts: float | None = Field(
        default=None, description="Finish timestamp (epoch seconds)."
    )
    error: str | None = Field(default=None, description="Error message, if any.")
    last_error: str | None = Field(
        default=None, description="Most recent executor error message."
    )
    attempts: int = Field(description="Attempt count.")
    max_attempts: int = Field(description="Max retry count.")
    next_retry_at: str | None = Field(default=None, description="Next retry timestamp.")
    last_failed_worker: str | None = Field(
        default=None, description="The most recent worker to have failed this task."
    )
    parent_task_id: str | None = Field(
        default=None, description="Parent task identifier."
    )
    merged_parent_id: str | None = Field(
        default=None, description="Merged parent task identifier."
    )
    local_name: str | None = Field(default=None, description="Workflow stage name.")
    graph_node_name: str | None = Field(default=None, description="Graph node name.")
    latest_update: dict[str, Any] | None = Field(
        default=None, description="Latest mid-task update payload."
    )
    completed: bool = Field(description="Whether the task completed successfully.")
    failed: bool = Field(description="Whether the task failed.")

    @classmethod
    def from_info(cls, info: TaskInfo) -> Self:
        return cls.model_validate(info, from_attributes=True)
