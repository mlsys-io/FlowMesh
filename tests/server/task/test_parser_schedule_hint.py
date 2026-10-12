"""Parser schedule-hint behavior and schema validation tests."""

import textwrap
from typing import Any

import pytest
import yaml

from server.task.parser import ParsedTask, ParsedWorkflow, parse_workflow
from shared.tasks.components.output import (
    OutputDestinationHTTPTemplate,
)
from shared.tasks.specs import InferenceSpecTemplate


def _task_map(parsed: ParsedWorkflow) -> dict[str, ParsedTask]:
    """Map graph node name to parsed task entry for concise assertions."""
    return {
        task.graph_node_name: task
        for task in parsed.tasks
        if task.graph_node_name is not None
    }


def test_parse_task_sets_schedule_in_epoch_order_true_for_nested_node_order() -> None:
    """
    Validate nested node_execution_order.
    - schedule_in_epoch_order should be True.
    - position_in_epoch should reflect index within each epoch.
    - nodes in later epochs reset position_in_epoch to 0.
    """
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: graph
          annotations:
            schedule_hint:
              node_schedule_in_epoch_order: true
              node_execution_order:
                - [a, b]
                - [c]
        spec:
          graph:
            nodes:
              - name: a
                spec:
                  taskType: echo
              - name: b
                spec:
                  taskType: echo
              - name: c
                dependsOn: [a]
                spec:
                  taskType: echo
        """)

    parsed = parse_workflow(payload, "native")
    by_node = _task_map(parsed)

    assert parsed.schedule_in_epoch_order is True
    assert parsed.epoch_groups == [["a", "b"], ["c"]]
    assert by_node["a"].position_in_epoch == 0
    assert by_node["b"].position_in_epoch == 1
    assert by_node["c"].position_in_epoch == 0


def test_parse_task_sets_schedule_in_epoch_order_true_for_flat_node_order() -> None:
    """
    Validate flat node_execution_order.
    - Treated as a single epoch.
    - schedule_in_epoch_order should be True.
    - position_in_epoch should follow the flat list order.
    """
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: graph
          annotations:
            schedule_hint:
              node_execution_order: [a, b, c]
        spec:
          graph:
            nodes:
              - name: a
                spec:
                  taskType: echo
              - name: b
                spec:
                  taskType: echo
              - name: c
                dependsOn: [a]
                spec:
                  taskType: echo
        """)

    parsed = parse_workflow(payload, "native")
    by_node = _task_map(parsed)

    assert parsed.schedule_in_epoch_order is True
    assert parsed.epoch_groups == [["a", "b", "c"]]
    assert by_node["a"].position_in_epoch == 0
    assert by_node["b"].position_in_epoch == 1
    assert by_node["c"].position_in_epoch == 2


def test_parse_task_sets_schedule_in_epoch_order_false_for_nested_node_order() -> None:
    """
    Validate unordered epochs with nested node_execution_order.
    - schedule_in_epoch_order should be False.
    - position_in_epoch should be omitted (None) for all nodes.
    """
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: graph
          annotations:
            schedule_hint:
              node_schedule_in_epoch_order: false
              node_execution_order:
                - [a, b]
                - [c]
        spec:
          graph:
            nodes:
              - name: a
                spec:
                  taskType: echo
              - name: b
                spec:
                  taskType: echo
              - name: c
                dependsOn: [a]
                spec:
                  taskType: echo
        """)

    parsed = parse_workflow(payload, "native")
    by_node = _task_map(parsed)

    assert parsed.schedule_in_epoch_order is False
    assert parsed.epoch_groups == [["a", "b"], ["c"]]
    assert by_node["a"].position_in_epoch is None
    assert by_node["b"].position_in_epoch is None
    assert by_node["c"].position_in_epoch is None


def test_parse_task_sets_schedule_in_epoch_order_none_when_no_node_order() -> None:
    """
    Validate missing node_execution_order.
    - schedule_in_epoch_order should be None.
    - position_in_epoch should be None for all nodes.
    """
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: graph
        spec:
          graph:
            nodes:
              - name: a
                spec:
                  taskType: echo
              - name: b
                spec:
                  taskType: echo
        """)

    parsed = parse_workflow(payload, "native")
    by_node = _task_map(parsed)

    assert parsed.schedule_in_epoch_order is None
    assert parsed.epoch_groups is None
    assert by_node["a"].position_in_epoch is None
    assert by_node["b"].position_in_epoch is None


def test_parse_task_applies_nested_selected_worker_schema() -> None:
    """
    Validate nested selected_worker schema.
    - global applies to all nodes.
    - selected overrides per-node values.
    """
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: graph
          annotations:
            schedule_hint:
              selected_worker:
                global: [w1, w2]
                selected:
                  b: [w3]
        spec:
          graph:
            nodes:
              - name: a
                spec:
                  taskType: echo
              - name: b
                spec:
                  taskType: echo
        """)

    parsed = parse_workflow(payload, "native")
    by_node = _task_map(parsed)

    assert by_node["a"].selected_worker == ["w1", "w2"]
    assert by_node["b"].selected_worker == ["w3"]


def test_parse_task_rejects_unknown_node_in_selected_worker_map() -> None:
    """
    Validate rejection of unknown nodes in selected_worker.selected.
    - unknown node name should raise ValueError.
    """
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: graph
          annotations:
            schedule_hint:
              selected_worker:
                selected:
                  unknown: w1
        spec:
          graph:
            nodes:
              - name: a
                spec:
                  taskType: echo
              - name: b
                spec:
                  taskType: echo
        """)

    with pytest.raises(ValueError, match="references unknown node 'unknown'"):
        parse_workflow(payload, "native")


def test_parse_task_allows_epoch_flag_without_node_order() -> None:
    """
    Validate node_schedule_in_epoch_order without node_execution_order.
    - schedule_in_epoch_order should remain None.
    - no error should be raised.
    """
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: graph
          annotations:
            schedule_hint:
              node_schedule_in_epoch_order: true
        spec:
          graph:
            nodes:
              - name: a
                spec:
                  taskType: echo
        """)

    parsed = parse_workflow(payload, "native")
    assert parsed.schedule_in_epoch_order is None
    assert parsed.epoch_groups is None


def test_parse_task_rejects_flat_order_with_false_epoch_flag() -> None:
    """
    Validate flat node_execution_order with node_schedule_in_epoch_order=false.
    - flat order requires in-epoch ordering.
    - should raise ValueError.
    """
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: graph
          annotations:
            schedule_hint:
              node_schedule_in_epoch_order: false
              node_execution_order: [a, b]
        spec:
          graph:
            nodes:
              - name: a
                spec:
                  taskType: echo
              - name: b
                spec:
                  taskType: echo
        """)

    with pytest.raises(ValueError, match="requires.*node_schedule_in_epoch_order"):
        parse_workflow(payload, "native")


def test_parse_task_allows_placeholders_in_typed_template_fields() -> None:
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: placeholder-task
        spec:
          taskType: inference
          sloSeconds: ${inputs.slo}
          enforce_cpu: ${inputs.enforce_cpu}
          output:
            destination:
              type: http
              timeoutSec: ${inputs.timeout}
          model:
            source:
              trust_remote_code: ${inputs.trust}
          parallel:
            enabled: ${inputs.parallel_enabled}
            max_shards: ${inputs.max_shards}
        """)

    parsed = parse_workflow(payload, "native")
    spec = parsed.tasks[0].task.spec

    assert isinstance(spec, InferenceSpecTemplate)
    assert spec.sloSeconds == "${inputs.slo}"
    assert spec.enforce_cpu == "${inputs.enforce_cpu}"
    assert spec.output is not None
    assert spec.output.destination is not None
    assert isinstance(spec.output.destination, OutputDestinationHTTPTemplate)
    assert spec.output.destination.timeoutSec == "${inputs.timeout}"
    assert spec.model is not None
    assert spec.model.source is not None
    assert spec.model.source.trust_remote_code == "${inputs.trust}"
    assert spec.parallel is not None
    assert spec.parallel.enabled == "${inputs.parallel_enabled}"
    assert spec.parallel.max_shards == "${inputs.max_shards}"


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        (" wkr-1 ", ["wkr-1"]),
        (["wkr-1", "wkr-1", "wkr-2"], ["wkr-1", "wkr-2"]),
        ({"global": ["wkr-1"]}, ["wkr-1"]),
        ({"global": ["wkr-1"], "selected": {"single": "wkr-2"}}, ["wkr-2"]),
    ],
)
def test_single_task_preserves_worker_pins(hint: Any, expected: list[str]) -> None:
    payload = yaml.safe_dump(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {
                "name": "single",
                "annotations": {"schedule_hint": {"selected_worker": hint}},
            },
            "spec": {"taskType": "echo"},
        }
    )
    parsed = parse_workflow(payload, "native")
    assert parsed.tasks[0].selected_worker == expected


def test_single_task_rejects_unknown_worker_pin_map_key() -> None:
    payload = yaml.safe_dump(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {
                "name": "single",
                "annotations": {
                    "schedule_hint": {
                        "selected_worker": {"selected": {"other": "wkr-1"}}
                    }
                },
            },
            "spec": {"taskType": "echo"},
        }
    )
    with pytest.raises(ValueError, match="unknown node 'other'"):
        parse_workflow(payload, "native")


@pytest.mark.parametrize("shape", ["single", "stages", "graph"])
def test_worker_fallback_policy_survives_expansion(shape: str) -> None:
    spec: dict[str, Any] = {"taskType": "echo"}
    if shape == "stages":
        spec = {"stages": [{"name": "a", "spec": {"taskType": "echo"}}]}
    elif shape == "graph":
        spec = {"graph": {"nodes": [{"name": "a", "spec": {"taskType": "echo"}}]}}
    payload = yaml.safe_dump(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Workflow",
            "metadata": {
                "annotations": {
                    "schedule_hint": {
                        "selected_worker": "wkr-old",
                        "allow_worker_fallback": True,
                    }
                }
            },
            "spec": spec,
        }
    )
    parsed = parse_workflow(payload, "native")
    task = parsed.tasks[0]
    assert task.selected_worker == ["wkr-old"]
    assert task.task.metadata is not None and task.task.metadata.annotations is not None
    hint = task.task.metadata.annotations.schedule_hint
    assert hint is not None and hint.allow_worker_fallback


@pytest.mark.parametrize("value", ["true", 1, None])
def test_worker_fallback_requires_a_boolean(value: Any) -> None:
    payload = yaml.safe_dump(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {
                "annotations": {"schedule_hint": {"allow_worker_fallback": value}}
            },
            "spec": {"taskType": "echo"},
        }
    )
    with pytest.raises(ValueError, match="allow_worker_fallback"):
        parse_workflow(payload, "native")
