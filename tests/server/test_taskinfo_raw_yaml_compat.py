"""GET /tasks responses keep the pre-#137 ``raw_yaml`` key for old SDK clients.

flowmesh-sdk<=0.1.9 (every released Lumilake image) declares ``raw_yaml`` as a
required field of TaskInfo. When the server stopped emitting it, every
``tasks.retrieve()`` raised and every Lumilake job failed with "Failed to fetch
FlowMesh task description".
"""

import json

from server.task.models import TaskInfo
from shared.tasks import TaskEnvelopeTemplate
from shared.utils.redact import REDACTED


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
    dumped = _info("api:\n  headers:\n    Authorization: Bearer SECRET\n").model_dump(
        mode="json"
    )
    assert "SECRET" not in json.dumps(dumped)
    assert dumped["raw_yaml"] == dumped["source"]
    assert REDACTED in dumped["raw_yaml"]


def test_round_trip_through_dump_validates() -> None:
    info = _info("spec: {}\n")
    again = TaskInfo.model_validate(info.model_dump(mode="json"))
    assert again.source == info.source


def test_old_sdk_shape_parses_response() -> None:
    """A client model that REQUIRES raw_yaml (sdk<=0.1.9) accepts the dump."""
    from pydantic import BaseModel

    class OldSdkTaskInfo(BaseModel):
        task_id: str
        raw_yaml: str

    OldSdkTaskInfo.model_validate(_info("spec: {}\n").model_dump(mode="json"))
