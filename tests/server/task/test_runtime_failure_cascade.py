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


_DIAMOND = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: diamond
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
        dependsOn: [a]
        spec:
          taskType: echo
      - name: d
        dependsOn: [b, c]
        spec:
          taskType: echo
"""


def test_descendant_reached_through_two_paths_fails_once() -> None:
    runtime, _ = build_runtime("cascade-diamond")
    _, ids = register(runtime, _DIAMOND)

    impacted, _, _ = runtime.mark_failed(
        ids["a"], None, {"error": "boom"}, "2026-10-10T00:00:00Z"
    )

    reason = f"Dependency {ids['a']} failed"
    assert sorted(impacted) == sorted((ids[name], reason) for name in ("b", "c", "d"))
    assert impacted[-1][0] == ids["d"]


_EPOCH_CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: epoch-chain
  annotations:
    schedule_hint:
      node_schedule_in_epoch_order: true
      node_execution_order:
        - [a]
        - [b, side]
        - [c]
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
      - name: side
        spec:
          taskType: echo
      - name: c
        dependsOn: [b]
        spec:
          taskType: echo
"""


def test_epoch_order_names_dependency_for_descendants_only() -> None:
    runtime, _ = build_runtime("cascade-epoch")
    _, ids = register(runtime, _EPOCH_CHAIN)

    impacted, _, _ = runtime.mark_failed(
        ids["a"], None, {"error": "boom"}, "2026-10-10T00:00:00Z"
    )

    dependency = f"Dependency {ids['a']} failed"
    blocked = f"Blocked by failed task {ids['a']} in earlier epoch"
    assert impacted == [
        (ids["b"], dependency),
        (ids["c"], dependency),
        (ids["side"], blocked),
    ]
    for name, reason in (("b", dependency), ("c", dependency), ("side", blocked)):
        record = runtime.get_record(ids[name])
        assert record is not None
        assert record.status == TaskStatus.FAILED
        assert record.error == reason


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
