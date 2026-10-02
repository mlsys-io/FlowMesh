"""Dispatching a python task whose inputs name a stage it does not depend on."""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from server.task.models import TaskStatus
from shared.schemas.result import BaseExecutorResult, ResultEnvelope, write_result
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.task.merge_harness import build_runtime, register

_PAYLOAD = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: python-inputs
spec:
  graph:
    nodes:
      - name: prep
        spec:
          taskType: echo
          data:
            type: list
            items: [x]
      - name: score
        dependsOn: [prep]
        spec:
          taskType: python
          inputs:
            - stage: nope
          code: |
            def main(inputs):
                return 1
"""


def test_unknown_input_stage_fails_the_task(tmp_path: Path) -> None:
    runtime, _ = build_runtime("dispatch-python-inputs")
    _, nodes = register(runtime, _PAYLOAD)
    runtime._tasks[nodes["prep"]].status = TaskStatus.DONE
    write_result(
        tmp_path, ResultEnvelope(task_id=nodes["prep"], result=BaseExecutorResult())
    )

    worker = SimpleNamespace(id="w-1", node_id="nde-1")
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1

    disp = CapturingDispatcher(
        runtime=cast(Any, runtime),
        worker_registry=cast(Any, registry),
        results_dir=tmp_path,
        logger=logging.getLogger("dispatch-python-inputs"),
        worker_selection_strategy="first_fit",
        enable_context_reuse=False,
    )

    assert disp.dispatch_once(nodes["score"]) is True
    registry.publish_task.assert_not_called()
    assert [(task_id, error) for task_id, error, _ in disp.failed] == [
        (nodes["score"], f"Unknown input stage 'nope' for task {nodes['score']}")
    ]
