"""A finished task drops bulky inputs from memory and from what it reloads."""

import threading

from server.registries.workflow import PersistedTask, _load_compacted
from server.task.models import TERMINAL_PAYLOAD_MAX_BYTES, TaskStatus
from tests.server.task.merge_harness import build_runtime, register


def _payload(n_items: int) -> str:
    items = "\n".join(f"            - row-{i}-" + "x" * 200 for i in range(n_items))
    return f"""
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: bulky
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
          data:
            type: list
            items:
{items}
"""


def _fail_only_task(n_items: int):
    runtime, _ = build_runtime("compact")
    _, ids = register(runtime, _payload(n_items))
    assert runtime.next_ready(threading.Event(), timeout=0.01) == ids["a"]
    return runtime, ids["a"]


def test_live_task_keeps_its_payload() -> None:
    runtime, tid = _fail_only_task(600)
    record = runtime.get_record(tid)
    assert record is not None and record.status != TaskStatus.FAILED
    assert not record.compact_terminal()
    assert record.task.spec.data is not None
    assert len(record.source) > TERMINAL_PAYLOAD_MAX_BYTES


def test_finished_task_drops_a_bulky_payload() -> None:
    runtime, tid = _fail_only_task(600)
    runtime.mark_failed(tid, None, {"error": "boom"}, "2026-10-11T00:00:00Z")
    record = runtime.get_record(tid)
    assert record is not None and record.status == TaskStatus.FAILED
    assert record.task.spec.data is None
    assert record.source.startswith("# source dropped after the task finished")
    assert record.model_dump()["task"]["spec"].get("data") is None


def test_finished_task_keeps_a_small_payload() -> None:
    runtime, tid = _fail_only_task(3)
    runtime.mark_failed(tid, None, {"error": "boom"}, "2026-10-11T00:00:00Z")
    record = runtime.get_record(tid)
    assert record is not None
    assert record.task.spec.data is not None
    assert "row-0-" in record.source


def test_reload_compacts_a_bulky_finished_task() -> None:
    runtime, tid = _fail_only_task(600)
    record = runtime.get_record(tid)
    assert record is not None
    record.status = TaskStatus.DONE  # persisted before compaction existed
    blob = PersistedTask(record=record).model_dump_json()
    assert len(blob) > TERMINAL_PAYLOAD_MAX_BYTES * 2

    loaded = _load_compacted(blob)
    assert loaded.record.task.spec.data is None
    assert len(loaded.record.source) < 200
