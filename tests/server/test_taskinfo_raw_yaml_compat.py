"""TaskInfo responses carry a redacted ``raw_yaml`` copy of ``source`` that the 0.1.9
SDK TaskInfo parses."""

import json
from typing import Any

from pydantic import BaseModel

from server.task.models import TaskInfo
from shared.tasks import TaskEnvelopeTemplate
from shared.utils.redact import REDACTED, redact_raw_yaml

# TODO(deprecate): remove this module with the `raw_yaml` shim in
# server/task/models.py.


# Top-level fields of flowmesh-sdk 0.1.9's TaskInfo; frozen, do not update.
class _SdkV019TaskInfo(BaseModel):
    task_id: str
    workflow_id: str
    owner_id: str
    org_id: str
    supplier_id: str
    raw_yaml: str
    task: dict[str, Any]
    status: str
    task_type: str | None = None
    category: str | None = None
    assigned_worker: str | None = None
    topic: str | None = None
    submitted_at: str
    submitted_ts: float
    dispatched_ts: float | None = None
    started_ts: float | None = None
    finished_ts: float | None = None
    usages: list[dict[str, Any]]
    error: str | None = None
    attempts: int
    max_attempts: int
    parent_task_id: str | None = None
    shard_index: int | None = None
    shard_total: int | None = None
    next_retry_at: str | None = None
    last_failed_worker: str | None = None
    last_error: str | None = None
    local_name: str | None = None
    graph_node_name: str | None = None
    load: int
    position_in_epoch: int | None = None
    selected_worker: list[str] | None = None
    merged_children: list[str] | None = None
    merged_parent_id: str | None = None
    merge_slice: dict[str, int] | None = None
    merge_key: str | None = None
    latest_update: dict[str, Any] | None = None
    depends_on: list[str]
    pending_dependencies: list[str]
    dependents: list[str]
    completed: bool
    failed: bool


def _info(source: str) -> TaskInfo:
    task = TaskEnvelopeTemplate.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "t"},
            "spec": {"taskType": "echo", "data": {"type": "list", "items": ["x"]}},
        }
    )
    return TaskInfo(
        task_id="tsk-1",
        workflow_id="wfl-1",
        owner_id="owner",
        source=source,
        task=task,
        depends_on=[],
        pending_dependencies=[],
        dependents=[],
        completed=False,
        failed=False,
    )


def test_dump_carries_both_keys() -> None:
    dumped = _info("spec:\n  taskType: echo\n").model_dump(mode="json")
    assert dumped["source"] == "spec:\n  taskType: echo\n"
    assert dumped["raw_yaml"] == dumped["source"]


def test_raw_yaml_is_redacted_like_source() -> None:
    # ``source`` is redacted once at registration; ``raw_yaml`` mirrors it.
    dumped = _info(
        redact_raw_yaml("api:\n  headers:\n    Authorization: Bearer SECRET\n")
    ).model_dump(mode="json")
    assert "SECRET" not in json.dumps(dumped)
    assert dumped["raw_yaml"] == dumped["source"]
    assert REDACTED in dumped["raw_yaml"]


def test_round_trip_through_dump_validates() -> None:
    info = _info("spec: {}\n")
    again = TaskInfo.model_validate(info.model_dump(mode="json"))
    assert again.source == info.source


def test_sdk_0_1_9_task_info_parses_response() -> None:
    dumped = _info("spec: {}\n").model_dump(mode="json")
    old = _SdkV019TaskInfo.model_validate(dumped)
    assert old.raw_yaml == dumped["source"]
