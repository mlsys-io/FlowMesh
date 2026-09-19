import time
from typing import Any, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    computed_field,
    model_serializer,
)
from pydantic.main import IncEx
from pydantic_core.core_schema import IncExCall

from shared.tasks import TaskEnvelopeTemplate, TaskSpecTemplate
from shared.tasks.worker_message import HardwareUsage
from shared.utils.redact import redact_raw_yaml

from ..utils.time import now_iso


def _descend_field_filter(selector: IncExCall, *path: str) -> IncExCall:
    """Descend a pydantic exclude selector to a nested field path.

    Return the sub-selector at ``path`` only when it is a set or mapping that
    ``model_dump`` accepts; otherwise None — the field is unfiltered, or
    excluded wholesale (handled by the caller's presence check) and never a
    bare bool, which ``model_dump`` rejects as a selector.
    """
    for key in path:
        if not isinstance(selector, dict):
            return None
        selector = selector.get(key)
    return selector if isinstance(selector, (set, dict)) else None


TRAINING_TASK_TYPES = {
    "sft",
    "lora_sft",
    "ppo",
    "dpo",
    "training",
    "image_classification_training",
}


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
    model_config = ConfigDict(validate_by_alias=True)

    task_id: str = Field(description="Task identifier.")
    workflow_id: str = Field(description="Workflow identifier.")
    owner_id: str = Field(description="Owner principal identifier.")
    org_id: str = Field(default="", description="Owner organization identifier.")
    supplier_id: str = Field(default="", description="Supplier identifier.")
    source: str = Field(
        validation_alias=AliasChoices("source", "raw_yaml"),
        description="Original workflow source (YAML or JSON).",
    )
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
        exclude=True,
    )
    last_error: str | None = Field(
        default=None, description="Most recent executor error message."
    )
    no_eligible_since: float | None = Field(
        default=None,
        description="Epoch seconds when no eligible worker was first observed.",
        exclude=True,
    )
    no_dispatch_since: float | None = Field(
        default=None,
        description="Epoch seconds when a selected worker was first found "
        "undeliverable.",
        exclude=True,
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

    _redacted_source: str | None = PrivateAttr(default=None)
    _redacted_spec: TaskSpecTemplate | None = PrivateAttr(default=None)

    def _redact_source(self) -> str:
        if self._redacted_source is None:
            self._redacted_source = redact_raw_yaml(self.source)
        return self._redacted_source

    def _redact_spec(self) -> TaskSpecTemplate:
        if self._redacted_spec is None:
            self._redacted_spec = self.task.spec.redact_credentials()
        return self._redacted_spec

    @model_serializer(mode="wrap")
    def _serialize(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        data = handler(self)
        if "source" in data:
            data["source"] = self._redact_source()
        task = data.get("task")
        if isinstance(task, dict) and "spec" in task:
            task["spec"] = self._redact_spec().model_dump(
                mode=info.mode,
                by_alias=info.by_alias,
                exclude_none=info.exclude_none,
                # info.exclude is typed IncExCall; model_dump expects the
                # equivalent IncEx — same runtime shapes, distinct aliases.
                exclude=cast(
                    IncEx | None,
                    _descend_field_filter(info.exclude, "task", "spec"),
                ),
            )
        return data


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
