from typing import Self

from pydantic import BaseModel, Field

from ..registries.workflow import Workflow, WorkflowStatus


class WorkflowSubmitTaskEntry(BaseModel):
    task_id: str = Field(description="Task identifier.")
    status: str | None = Field(default=None, description="Current task status.")
    assigned_worker: str | None = Field(
        default=None, description="Worker assigned to the task."
    )
    topic: str | None = Field(default=None, description="Dispatch topic.")
    waiting_on: list[str] | None = Field(
        default=None, description="Pending dependency task IDs."
    )
    depends_on: list[str] | None = Field(
        default=None, description="Dependency task IDs."
    )
    attempts: int | None = Field(default=None, description="Attempt count.")
    max_attempts: int | None = Field(default=None, description="Max retry count.")
    load: int | None = Field(default=None, description="Task load score.")


class WorkflowSubmitResponse(BaseModel):
    ok: bool = Field(description="Whether the submission succeeded.")
    workflow_id: str = Field(description="Workflow identifier.")
    count: int = Field(description="Number of tasks in the workflow.")
    tasks: list[WorkflowSubmitTaskEntry] = Field(description="Submitted task entries.")


class WorkflowValidateTaskEntry(BaseModel):
    task_id: str = Field(
        description="Mock task identifier, not associated with any real task."
    )
    graph_node_name: str | None = Field(description="Original graph node name.")
    depends_on: list[str] = Field(description="Dependency task IDs.")


class WorkflowValidateResponse(BaseModel):
    ok: bool = Field(description="Whether the validation succeeded.")
    count: int = Field(description="Number of validated tasks.")
    tasks: list[WorkflowValidateTaskEntry] = Field(
        description="Validated task entries."
    )


class WorkflowStatusView(BaseModel):
    """A workflow's status, timestamps and per-state task counts, without its task
    ID lists."""

    workflow_id: str = Field(description="Workflow identifier.")
    status: WorkflowStatus = Field(description="Workflow status.")
    submitted_at: str = Field(description="Submission timestamp.")
    updated_at: str = Field(description="Last update timestamp.")
    task_count: int = Field(description="Number of tasks in the workflow.")
    dispatched_count: int = Field(description="Number of dispatched tasks.")
    completed_count: int = Field(description="Number of completed tasks.")
    failed_count: int = Field(description="Number of failed tasks.")
    cancelled_count: int = Field(description="Number of cancelled tasks.")

    @classmethod
    def from_workflow(cls, workflow: Workflow) -> Self:
        return cls(
            workflow_id=workflow.workflow_id,
            status=workflow.status,
            submitted_at=workflow.submitted_at,
            updated_at=workflow.updated_at,
            task_count=len(workflow.task_ids),
            dispatched_count=len(workflow.dispatched_tasks),
            completed_count=len(workflow.completed_tasks),
            failed_count=len(workflow.failed_tasks),
            cancelled_count=len(workflow.cancelled_tasks),
        )
