"""The SDK TaskInfo accepts both the new ``source`` and the old ``raw_yaml`` key."""

import pytest

from flowmesh.models.tasks import TaskInfo


def _payload(key: str) -> dict:
    return {
        "task_id": "tsk-1",
        "workflow_id": "wfl-1",
        "owner_id": "o",
        "org_id": "",
        "supplier_id": "",
        key: "spec: {}\n",
        "task": {},
        "status": "DONE",
        "submitted_at": "2026-09-28T00:00:00+00:00",
        "submitted_ts": 0.0,
        "usages": [],
        "attempts": 0,
        "max_attempts": 3,
        "load": 0,
        "depends_on": [],
        "pending_dependencies": [],
        "dependents": [],
        "completed": True,
        "failed": False,
    }


@pytest.mark.parametrize("key", ["source", "raw_yaml"])
def test_source_accepts_both_keys(key: str) -> None:
    try:
        info = TaskInfo.model_validate(_payload(key))
    except Exception as exc:  # pragma: no cover - surfaced as a failure below
        pytest.fail(f"TaskInfo rejected a {key!r} payload: {exc}")
    assert info.source == "spec: {}\n"
