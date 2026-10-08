"""Getting an upstream stage's result to the server when a dependent reads it."""

import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest

from server.dispatcher.base import StageReferenceNotReady, StageResultMissing
from server.task.models import TaskStatus
from shared.schemas.result import BaseExecutorResult, ResultEnvelope, write_result
from shared.utils.result_delivery import artifacts_ready
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.dispatcher.test_merged_child_redaction_dispatch import (
    _RecordingDispatcher,
)
from tests.server.task.merge_harness import build_runtime
from tests.shared.test_result_delivery import populate

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
    runtime: Any,
    results_dir: Path,
    grace_sec: int = 120,
    result_delivery_enabled: bool = True,
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
        result_delivery_enabled=result_delivery_enabled,
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

    message = registry.publish_task.call_args.args[1]
    assert message.task.spec.output is None
    request = message.result_delivery[nodes["prep"]]
    assert request.all_artifacts is (score_type == "python")
    assert request.artifact_fields == []
    assert message.result_delivery_enabled is True


def test_disabled_delivery_still_describes_what_dependents_need(
    tmp_path: Path,
) -> None:
    runtime, _ = build_runtime("stage-result-delivery-off")
    _, nodes = register(runtime, _payload("python"))
    disp, registry = _dispatcher(runtime, tmp_path, result_delivery_enabled=False)

    assert disp.dispatch_once(nodes["prep"]) is True

    message = registry.publish_task.call_args.args[1]
    assert message.result_delivery_enabled is False
    assert message.result_delivery[nodes["prep"]].all_artifacts is True


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
    assert "TASK_RESULT_DELIVERY" not in error


def test_a_missing_result_with_delivery_disabled_names_the_setting(
    tmp_path: Path,
) -> None:
    runtime, _ = build_runtime("stage-result-lost-delivery-off")
    _, nodes = register(runtime, _payload())
    prep = runtime._tasks[nodes["prep"]]
    prep.status = TaskStatus.DONE
    prep.finished_ts = time.time() - 300
    disp, registry = _dispatcher(
        runtime, tmp_path, grace_sec=120, result_delivery_enabled=False
    )

    assert disp.dispatch_once(nodes["score"]) is True

    registry.publish_task.assert_not_called()
    [(task_id, error, _)] = disp.failed
    assert task_id == nodes["score"]
    assert "TASK_RESULT_DELIVERY=false" in error


def test_named_reference_requests_only_selected_artifact_and_hydration_descriptor(
    tmp_path: Path,
) -> None:
    runtime, _ = build_runtime("stage-named-artifact")
    payload = (
        _payload("python")
        + "        inputs: []\n        env:\n          MODEL: '${prep.model}'\n"
    )
    _, nodes = register(runtime, payload)
    disp, registry = _dispatcher(runtime, tmp_path)
    assert disp.dispatch_once(nodes["prep"])
    request = registry.publish_task.call_args.args[1].result_delivery[nodes["prep"]]
    assert request.all_artifacts is False
    assert request.artifact_fields == ["model"]
    runtime._tasks[nodes["prep"]].status = TaskStatus.DONE
    base = populate(
        tmp_path,
        task_id=nodes["prep"],
        dispatch_id=runtime._tasks[nodes["prep"]].result_dispatch,
    )
    registry.reset_mock()
    assert disp.dispatch_once(nodes["score"])
    message = registry.publish_task.call_args.args[1]
    [descriptor] = message.artifact_inputs[nodes["score"]]
    assert descriptor.task_id == nodes["prep"]
    assert descriptor.path == "model"
    assert descriptor.source == (base / "artifacts" / "model").as_posix()
    assert descriptor.generation


def test_transitive_consumer_demand_and_python_code_exemption(tmp_path: Path) -> None:
    runtime, _ = build_runtime("stage-transitive-demand")
    payload = _payload() + """
    - name: final
      dependsOn: [score]
      spec:
        taskType: python
        inputs: []
        env:
          MODEL: '${prep.model}'
        code: |
          def main():
              return '${prep.unused}'
"""
    _, nodes = register(runtime, payload)
    disp, registry = _dispatcher(runtime, tmp_path)
    assert disp.dispatch_once(nodes["prep"])
    request = registry.publish_task.call_args.args[1].result_delivery[nodes["prep"]]
    assert request.all_artifacts is False
    assert request.artifact_fields == ["model"]


def test_missing_mounted_artifacts_uses_the_result_grace(tmp_path: Path) -> None:
    runtime, _ = build_runtime("stage-partial-artifact")
    _, nodes = register(runtime, _payload("python"))
    upstream = runtime._tasks[nodes["prep"]]
    upstream.status = TaskStatus.DONE
    upstream.finished_ts = time.time()
    write_result(
        tmp_path,
        ResultEnvelope(
            task_id=upstream.task_id,
            result=BaseExecutorResult(),
            metadata={"independent_results": True},
        ),
    )
    disp, registry = _dispatcher(runtime, tmp_path)
    assert disp.dispatch_once(nodes["score"]) is False
    assert not registry.publish_task.called
    upstream.finished_ts -= 300
    assert disp.dispatch_once(nodes["score"]) is True
    assert disp.failed[0][0] == nodes["score"]


@pytest.mark.parametrize("later_child_waiting", [False, True])
def test_expired_merged_child_does_not_fail_parent_or_other_workflow(
    tmp_path: Path, later_child_waiting: bool
) -> None:
    runtime, _ = build_runtime("merged-delivery-expiry")
    payload = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: batch
spec:
  graph:
    nodes:
      - name: task
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
"""
    _, first = register(runtime, payload)
    _, second = register(runtime, payload)
    _, third = register(runtime, payload)
    parent, expired, survivor = first["task"], second["task"], third["task"]
    _, registry = _dispatcher(runtime, tmp_path)
    disp = _RecordingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        results_dir=tmp_path,
        logger=logging.getLogger("merged-expiry"),
        worker_selection_strategy="first_fit",
        enable_context_reuse=False,
        enable_task_merge=True,
        task_merge_max_batch_size=3,
    )
    resolve = disp._resolve_stage_references

    def resolve_child(identifier: str, task: Any, record: Any) -> Any:
        if identifier == expired:
            raise StageResultMissing("upstream delivery expired")
        if identifier == survivor and later_child_waiting:
            raise StageReferenceNotReady("upstream delivery pending")
        return resolve(identifier, task, record)

    with mock.patch.object(
        disp, "_resolve_stage_references", side_effect=resolve_child
    ):
        assert disp.dispatch_once(parent) is (not later_child_waiting)
    expired_record = runtime.get_record(expired)
    parent_record = runtime.get_record(parent)
    survivor_record = runtime.get_record(survivor)
    assert (
        expired_record is not None
        and parent_record is not None
        and survivor_record is not None
    )
    assert expired_record.status == TaskStatus.FAILED
    assert expired_record.merged_parent_id is None
    assert parent_record.status != TaskStatus.FAILED
    assert survivor_record.status != TaskStatus.FAILED
    assert disp.failed == []
    assert any(
        event["task_id"] == expired and event["is_child"] for event in disp.events
    )
    if later_child_waiting:
        registry.publish_task.assert_not_called()
    else:
        assert [
            child.task_id
            for child in registry.publish_task.call_args.args[1].merged_children
        ] == [survivor]


@pytest.mark.parametrize("named", [True, False])
def test_retried_producer_cannot_reuse_previous_dispatch_result(
    tmp_path: Path, named: bool
) -> None:
    runtime, _ = build_runtime("retry-delivery")
    _, nodes = register(runtime, _payload("python"))
    upstream = nodes["prep"]
    disp, registry = _dispatcher(runtime, tmp_path)
    assert disp.dispatch_once(upstream)
    first_dispatch = runtime._tasks[upstream].result_dispatch
    populate(tmp_path, task_id=upstream, dispatch_id=first_dispatch)
    runtime.mark_pending(upstream, increment_retry=True)
    assert disp.dispatch_once(upstream)
    record = runtime._tasks[upstream]
    assert record.result_dispatch != first_dispatch
    record.status = TaskStatus.DONE
    record.finished_ts = time.time()
    if not named:
        record.local_name = None
        record.graph_node_name = None
    registry.reset_mock()
    assert disp.dispatch_once(nodes["score"]) is False
    assert not registry.publish_task.called
    record.finished_ts -= 300
    assert disp.dispatch_once(nodes["score"]) is True
    assert disp.failed[0][0] == nodes["score"]


def test_merged_child_dispatch_identity_changes_without_retry_counter(
    tmp_path: Path,
) -> None:
    runtime, _ = build_runtime("merged-retry-delivery")
    _, nodes = register(runtime, _payload())
    parent, child = nodes["prep"], nodes["score"]
    runtime.prepare_result_dispatch(parent, [child], "first-dispatch")
    attempts = runtime._tasks[child].attempts
    runtime.prepare_result_dispatch(parent, [child], "second-dispatch")
    assert runtime._tasks[child].attempts == attempts
    assert runtime._tasks[child].result_dispatch == "second-dispatch"


def test_dispatch_checks_each_artifact_reference_once_without_hashing(
    tmp_path: Path,
) -> None:
    runtime, _ = build_runtime("stage-artifact-check-once")
    payload = (
        _payload("python")
        + "        inputs: []\n        env:\n          MODEL: '${prep.model}'\n"
    )
    _, nodes = register(runtime, payload)
    disp, registry = _dispatcher(runtime, tmp_path)
    assert disp.dispatch_once(nodes["prep"])
    runtime._tasks[nodes["prep"]].status = TaskStatus.DONE
    populate(
        tmp_path,
        task_id=nodes["prep"],
        dispatch_id=runtime._tasks[nodes["prep"]].result_dispatch,
    )
    registry.reset_mock()

    with (
        mock.patch(
            "server.dispatcher.base.artifacts_ready", wraps=artifacts_ready
        ) as ready,
        mock.patch(
            "shared.utils.result_delivery.describe_file",
            side_effect=AssertionError("dispatch must not hash artifacts"),
        ),
    ):
        assert disp.dispatch_once(nodes["score"])

    assert registry.publish_task.called
    assert ready.call_count == 1
    assert ready.call_args.kwargs["verify_content"] is False


def test_merged_children_each_carry_their_own_delivery_request(
    tmp_path: Path,
) -> None:
    # A merged batch spans workflows: each child whose dependent reads it gets a
    # request of its own, and a child without such a dependent gets none.
    runtime, _ = build_runtime("merged-delivery-requests")
    batch = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: batch
spec:
  graph:
    nodes:
      - name: task
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
"""
    reader = batch + """
      - name: use
        dependsOn: [task]
        spec:
          taskType: python
          inputs: []
          env:
            ANSWER: '${task.answer}'
          code: |
            def main():
                return 1
"""
    _, first = register(runtime, reader)
    _, second = register(runtime, reader)
    _, third = register(runtime, batch)
    parent, reading_child, leaf_child = first["task"], second["task"], third["task"]
    _, registry = _dispatcher(runtime, tmp_path)
    disp = _RecordingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        results_dir=tmp_path,
        logger=logging.getLogger("merged-requests"),
        worker_selection_strategy="first_fit",
        enable_context_reuse=False,
        enable_task_merge=True,
        task_merge_max_batch_size=3,
    )

    assert disp.dispatch_once(parent) is True

    message = registry.publish_task.call_args.args[1]
    assert {child.task_id for child in message.merged_children} == {
        reading_child,
        leaf_child,
    }
    assert set(message.result_delivery) == {parent, reading_child}
    for identifier in (parent, reading_child):
        request = message.result_delivery[identifier]
        assert request.all_artifacts is False
        assert request.artifact_fields == ["answer"]
        record = runtime.get_record(identifier)
        assert record is not None
        assert record.result_dispatch == message.result_dispatch
    # The declared output is untouched; nothing turns on the HTTP destination.
    assert message.task.spec.output is None
    assert all(child.spec.output is None for child in message.merged_children)
