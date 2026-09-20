"""Dispatching a batch with a redacted merged child fails only that child.

The parent still dispatches, carrying its remaining valid children; the redacted
child is failed individually and its terminal event is marked as a child failure.
"""

import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from server.task.models import TaskStatus
from shared.utils.redact import REDACTED
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.task.merge_harness import build_runtime, register

_PAYLOAD = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: merge-batch
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
      - name: b
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
      - name: c
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
"""


class _RecordingDispatcher(CapturingDispatcher):
    """CapturingDispatcher that also records emitted task events."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[dict[str, Any]] = []

    def _emit_task_event(
        self,
        event_type: str,
        task_id: str,
        *,
        worker_id: str | None = None,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
        is_child: bool = False,
    ) -> None:
        self.events.append(
            {"type": event_type, "task_id": task_id, "is_child": is_child}
        )


def _redact_inference(runtime: Any, task_id: str) -> None:
    record = runtime._tasks[task_id]
    redacted = record.task.spec.model_copy(update={"inference": {"api_key": REDACTED}})
    record.task = record.task.model_copy(update={"spec": redacted})


def test_dispatch_fails_redacted_merged_child_and_carries_survivors() -> None:
    runtime, _ = build_runtime("dispatch-merged-redaction")
    _, nodes = register(runtime, _PAYLOAD)
    parent, survivor, redacted = nodes["a"], nodes["b"], nodes["c"]
    _redact_inference(runtime, redacted)

    worker = SimpleNamespace(id="w-1", node_id="nde-1")
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1

    disp = _RecordingDispatcher(
        runtime=cast(Any, runtime),
        worker_registry=cast(Any, registry),
        results_dir=Path(tempfile.gettempdir()),
        logger=logging.getLogger("dispatch-merged-redaction"),
        worker_selection_strategy="first_fit",
        enable_context_reuse=False,
        enable_task_merge=True,
        task_merge_max_batch_size=4,
    )

    assert disp.dispatch_once(parent) is True

    # The redacted child failed and was unlinked from the parent.
    assert runtime._tasks[redacted].status == TaskStatus.FAILED
    assert runtime._tasks[redacted].merged_parent_id is None
    # The parent dispatched carrying only the surviving child.
    registry.publish_task.assert_called_once()
    message = registry.publish_task.call_args.args[1]
    assert [c.task_id for c in (message.merged_children or [])] == [survivor]
    # Exactly one TASK_FAILED for the redacted child, marked as a child failure.
    child_events = [
        e
        for e in disp.events
        if e["task_id"] == redacted and e["type"] == "TASK_FAILED"
    ]
    assert len(child_events) == 1
    assert child_events[0]["is_child"] is True
