"""A failed task fails every PENDING task downstream of it, at any depth."""

import pytest

from server.task.models import TaskStatus
from tests.server.task.merge_harness import build_runtime, register
from tests.server.task.test_runtime_rehydrate import FakeWorkflowRegistry, _runtime

_CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: chain
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
      - name: b
        dependsOn: [a]
        spec:
          taskType: echo
      - name: c
        dependsOn: [b]
        spec:
          taskType: echo
      - name: d
        dependsOn: [c]
        spec:
          taskType: echo
      - name: side
        spec:
          taskType: echo
"""


def test_failure_cascades_through_every_descendant() -> None:
    runtime, registry = build_runtime("cascade")
    _, ids = register(runtime, _CHAIN)

    impacted, _, _ = runtime.mark_failed(
        ids["a"], None, {"error": "boom"}, "2026-10-10T00:00:00Z"
    )

    assert [t for t, _ in impacted] == [ids["b"], ids["c"], ids["d"]]
    for name in ("b", "c", "d"):
        record = runtime.get_record(ids[name])
        assert record is not None
        assert record.status == TaskStatus.FAILED
        assert record.error == f"Dependency {ids['a']} failed"
    side = runtime.get_record(ids["side"])
    assert side is not None
    assert side.status == TaskStatus.PENDING
    assert registry.calls[-1]["failed"] == [ids[name] for name in ("a", "b", "c", "d")]


@pytest.mark.anyio
async def test_cascaded_failures_remain_failed_after_restart() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, results = await runtime.register("owner", "org", _CHAIN)
    ids = {result.graph_node_name: result.task_id for result in results}
    runtime.mark_failed(ids["a"], None, {"error": "boom"}, "2026-10-10T00:00:00Z")

    restored = _runtime(registry)
    assert await restored.rehydrate() == 1

    for current in (runtime, restored):
        for name in ("a", "b", "c", "d"):
            info = current.describe_task(ids[name])
            assert info is not None
            assert info.status == TaskStatus.FAILED
            assert info.failed
            assert not info.completed
            assert not info.pending_dependencies
            if name != "a":
                assert info.error == f"Dependency {ids['a']} failed"
        side = current.describe_task(ids["side"])
        assert side is not None
        assert side.status == TaskStatus.PENDING
        assert not side.failed
    assert restored.ready_queue_length() == 1
