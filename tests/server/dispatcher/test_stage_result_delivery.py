"""Getting an upstream stage's result to the server when a dependent reads it."""

import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest

from server.task.models import TaskStatus
from shared.schemas.result import BaseExecutorResult, ResultEnvelope, write_result
from shared.tasks.components.output import OutputDestinationHTTP
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.task.merge_harness import build_runtime

_TWO_STAGE = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: two-stage
spec:
  stages:
    - name: prep
      spec:
        taskType: echo
        {prep_output}
        data:
          type: list
          items: [x]
    - name: score
      dependsOn: [prep]
      spec:
        taskType: {score_type}
"""

_ECHO_BODY = """
        data:
          type: list
          items: [y]
"""

_PYTHON_BODY = """
        code: |
          def main(prep):
              return 1
"""


def _payload(score_type: str = "echo", prep_output: str = "") -> str:
    body = _PYTHON_BODY if score_type == "python" else _ECHO_BODY
    return (
        _TWO_STAGE.format(prep_output=prep_output, score_type=score_type).rstrip()
        + "\n"
        + body
    )


def register(runtime: Any, payload: str) -> tuple[str, dict[str, str]]:
    """Register a workflow; return its id and a {stage name: task_id} map."""
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", payload, format="native")
    )
    records = [runtime.get_record(r.task_id) for r in results]
    return workflow_id, {
        str(rec.local_name or rec.graph_node_name): rec.task_id for rec in records
    }


def _dispatcher(
    runtime: Any, results_dir: Path, grace_sec: int = 120
) -> tuple[CapturingDispatcher, mock.Mock]:
    worker = SimpleNamespace(id="w-1", node_id="nde-1")
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1
    disp = CapturingDispatcher(
        runtime=cast(Any, runtime),
        worker_registry=cast(Any, registry),
        results_dir=results_dir,
        logger=logging.getLogger("stage-result-delivery"),
        worker_selection_strategy="first_fit",
        enable_context_reuse=False,
        stage_result_grace_sec=grace_sec,
    )
    return disp, registry


def _published_output(registry: mock.Mock) -> Any:
    message = registry.publish_task.call_args.args[1]
    return message.task.spec.output


@pytest.mark.parametrize("score_type", ["echo", "python"])
def test_a_stage_with_a_reading_dependent_uploads_its_result(
    tmp_path: Path, score_type: str
) -> None:
    runtime, _ = build_runtime("stage-result-upload")
    _, nodes = register(runtime, _payload(score_type))
    disp, registry = _dispatcher(runtime, tmp_path)

    assert disp.dispatch_once(nodes["prep"]) is True

    output = _published_output(registry)
    assert isinstance(output.destination, OutputDestinationHTTP)
    assert output.destination.url is None


def test_a_declared_destination_is_kept(tmp_path: Path) -> None:
    runtime, _ = build_runtime("stage-result-declared")
    prep_output = "output:\n          destination:\n            type: local"
    _, nodes = register(runtime, _payload(prep_output=prep_output))
    disp, registry = _dispatcher(runtime, tmp_path)

    assert disp.dispatch_once(nodes["prep"]) is True

    assert _published_output(registry).destination.type == "local"


def test_a_stage_without_dependents_is_left_alone(tmp_path: Path) -> None:
    runtime, _ = build_runtime("stage-result-leaf")
    _, nodes = register(runtime, _payload())
    runtime._tasks[nodes["prep"]].status = TaskStatus.DONE
    write_result(
        tmp_path, ResultEnvelope(task_id=nodes["prep"], result=BaseExecutorResult())
    )
    disp, registry = _dispatcher(runtime, tmp_path)

    assert disp.dispatch_once(nodes["score"]) is True

    assert _published_output(registry) is None


@pytest.mark.parametrize("score_type", ["echo", "python"])
def test_a_missing_result_is_waited_for_within_the_grace(
    tmp_path: Path, score_type: str
) -> None:
    runtime, _ = build_runtime("stage-result-waiting")
    _, nodes = register(runtime, _payload(score_type))
    prep = runtime._tasks[nodes["prep"]]
    prep.status = TaskStatus.DONE
    prep.finished_ts = time.time()
    disp, registry = _dispatcher(runtime, tmp_path)

    assert disp.dispatch_once(nodes["score"]) is False

    registry.publish_task.assert_not_called()
    assert disp.failed == []
    assert [task_id for task_id, _ in disp.requeued] == [nodes["score"]]


@pytest.mark.parametrize("score_type", ["echo", "python"])
def test_a_result_that_never_arrives_fails_the_dependent(
    tmp_path: Path, score_type: str
) -> None:
    runtime, _ = build_runtime("stage-result-lost")
    _, nodes = register(runtime, _payload(score_type))
    prep = runtime._tasks[nodes["prep"]]
    prep.status = TaskStatus.DONE
    prep.finished_ts = time.time() - 300
    disp, registry = _dispatcher(runtime, tmp_path, grace_sec=120)

    assert disp.dispatch_once(nodes["score"]) is True

    registry.publish_task.assert_not_called()
    [(task_id, error, _)] = disp.failed
    assert task_id == nodes["score"]
    assert f"Result of task {nodes['prep']} has not reached the server" in error
    assert "WORKER_UPLOAD_RESULTS=1" in error
