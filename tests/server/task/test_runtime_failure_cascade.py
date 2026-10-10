"""A failed task fails every PENDING task downstream of it, at any depth."""

import threading

from server.task.models import TaskStatus
from tests.server.task.merge_harness import build_runtime, register

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
    runtime, _ = build_runtime("cascade")
    _, ids = register(runtime, _CHAIN)
    stop = threading.Event()
    ready = {runtime.next_ready(stop, timeout=0.01) for _ in range(2)}
    assert ids["a"] in ready

    impacted, _, _ = runtime.mark_failed(
        ids["a"], None, {"error": "boom"}, "2026-10-10T00:00:00Z"
    )

    assert [t for t, _ in impacted] == [ids["b"], ids["c"], ids["d"]]
    for name, parent in (("b", "a"), ("c", "b"), ("d", "c")):
        record = runtime.get_record(ids[name])
        assert record is not None
        assert record.status == TaskStatus.FAILED
        assert record.error == f"Dependency {ids[parent]} failed"
    side = runtime.get_record(ids["side"])
    assert side is not None
    assert side.status != TaskStatus.FAILED
