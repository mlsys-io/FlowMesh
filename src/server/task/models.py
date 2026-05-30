import time
from typing import Any

from pydantic import BaseModel, Field, computed_field

from shared.tasks import TaskEnvelopeTemplate
from shared.tasks.worker_message import HardwareUsage

from ..utils.time import now_iso

TRAINING_TASK_TYPES = {"sft", "lora_sft", "ppo", "dpo", "training"}


def categorize_task_type(task_type: str | None) -> str:
    if not task_type:
        return "other"
    normalized = task_type.strip().lower()
    if normalized == "inference":
        return "inference"
    if normalized in TRAINING_TASK_TYPES:
        return "training"
    return "other"


class TaskStatus(str):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    CANCELLING = "CANCELLING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DONE = "DONE"


TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DONE}
)


class TaskUsage(BaseModel):
    started_at: str = Field(description="Start timestamp.")
    finished_at: str = Field(description="Finish timestamp.")
    runtime_sec: float = Field(description="Runtime in seconds.")
    hardware: HardwareUsage = Field(description="Hardware usage details.")
    cost_per_hour: float = Field(description="Cost per hour in USD.")
    total_cost: float = Field(description="Total cost in USD.")
    status: str = Field(description="Task status at completion.")

    @classmethod
    def from_payload(cls, payload: dict[str, Any], status: str) -> "TaskUsage | None":
        try:
            return cls(
                started_at=payload["started_at"],
                finished_at=payload["finished_at"],
                runtime_sec=payload["runtime_sec"],
                hardware=HardwareUsage.model_validate(payload["hardware"]),
                cost_per_hour=payload["cost_per_hour"],
                total_cost=payload["total_cost"],
                status=status,
            )
        except (KeyError, TypeError, ValueError):
            return None


class TaskRecord(BaseModel):
    task_id: str = Field(description="Task identifier.")
    workflow_id: str = Field(description="Workflow identifier.")
    owner_id: str = Field(description="Owner principal identifier.")
    org_id: str = Field(default="", description="Owner organization identifier.")
    supplier_id: str = Field(default="", description="Supplier identifier.")
    raw_yaml: str = Field(description="Original workflow YAML.")
    task: TaskEnvelopeTemplate = Field(description="Task template.")
    status: str = Field(default=TaskStatus.PENDING, description="Task status.")
    task_type: str | None = Field(default=None, description="Task type.")
    category: str | None = Field(default=None, description="Task category.")
    assigned_worker: str | None = Field(
        default=None, description="Assigned worker identifier."
    )
    topic: str | None = Field(default=None, description="Dispatch topic.")
    submitted_at: str = Field(
        default_factory=now_iso, description="Submission timestamp."
    )
    submitted_ts: float = Field(
        default_factory=time.time, description="Submission epoch seconds."
    )
    last_queue_ts: float = Field(
        default_factory=time.time, description="Last queue timestamp (epoch seconds)."
    )
    dispatched_ts: float | None = Field(
        default=None, description="Dispatch timestamp (epoch seconds)."
    )
    started_ts: float | None = Field(
        default=None, description="Start timestamp (epoch seconds)."
    )
    finished_ts: float | None = Field(
        default=None, description="Finish timestamp (epoch seconds)."
    )
    usages: list[TaskUsage] = Field(
        default_factory=list, description="Resource usage records."
    )
    error: str | None = Field(default=None, description="Error message, if any.")
    attempts: int = Field(default=0, description="Attempt count.")
    max_attempts: int = Field(default=3, description="Max retry count.")
    parent_task_id: str | None = Field(
        default=None, description="Parent task identifier."
    )
    shard_index: int | None = Field(default=None, description="Shard index.")
    shard_total: int | None = Field(default=None, description="Total shard count.")
    next_retry_at: str | None = Field(default=None, description="Next retry timestamp.")
    failed_workers: list[str] = Field(
        default_factory=list,
        description="Distinct workers that have failed this task.",
    )
    last_error: str | None = Field(
        default=None, description="Most recent executor error message."
    )
    no_eligible_since: float | None = Field(
        default=None,
        description="Epoch seconds when no eligible worker was first observed.",
    )
    local_name: str | None = Field(default=None, description="Workflow stage name.")
    graph_node_name: str | None = Field(default=None, description="Graph node name.")
    load: int = Field(default=0, description="Load score.")
    position_in_epoch: int | None = Field(
        default=None, description="Position within the scheduled epoch."
    )
    selected_worker: list[str] | None = Field(
        default=None, description="Selected worker identifiers."
    )
    merged_children: list[str] | None = Field(
        default=None, description="Merged child task identifiers."
    )
    merged_parent_id: str | None = Field(
        default=None, description="Merged parent task identifier."
    )
    merge_slice: dict[str, int] | None = Field(
        default=None, description="Merge slice information."
    )
    merge_key: str | None = Field(default=None, description="Merge grouping key.")
    latest_update: dict[str, Any] | None = Field(
        default=None, description="Latest mid-task update payload."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def last_failed_worker(self) -> str | None:
        """The most recent worker to have failed this task."""
        return self.failed_workers[-1] if self.failed_workers else None


class TaskInfo(TaskRecord):
    depends_on: list[str] = Field(description="Dependency task IDs.")
    pending_dependencies: list[str] = Field(
        description="Unresolved dependency task IDs."
    )
    dependents: list[str] = Field(description="Dependent task IDs.")
    completed: bool = Field(description="Whether the task completed successfully.")
    failed: bool = Field(description="Whether the task failed.")


class TaskParsingResult(BaseModel):
    task_id: str = Field(description="Task identifier.")
    graph_node_name: str | None = Field(description="Original graph node name.")
    depends_on: list[str] = Field(description="Dependency task IDs.")
