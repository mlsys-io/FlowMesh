"""Tests for conditional task execution in FlowMesh.

When a task spec has a ``condition`` field, the server should evaluate it
against the upstream node's result and skip the task (mark completed without
dispatch) if the condition is not met.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from server.dispatcher.base import Dispatcher
from server.task.parser import parse_workflow
from shared.tasks.specs import (
    DataRetrievalSpecStrict,
    InferenceSpecStrict,
)
from shared.tasks.specs.common import ConditionSpec
from shared.tasks.task_type import TaskType


class TestConditionSpecValidation:
    """ConditionSpec Pydantic model validation."""

    def test_valid_condition(self) -> None:
        spec = ConditionSpec(node="judge_0", field="verdict", equals="insufficient")
        assert spec.node == "judge_0"
        assert spec.field == "verdict"
        assert spec.equals == "insufficient"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConditionSpec(node="judge_0", field="verdict")  # type: ignore[call-arg]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ConditionSpec(
                node="judge_0",
                field="verdict",
                equals="insufficient",
                extra_field="bad",  # type: ignore[call-arg]
            )

    def test_serialization_round_trip(self) -> None:
        spec = ConditionSpec(node="a", field="x", equals="yes")
        data = spec.model_dump()
        assert data == {"node": "a", "field": "x", "equals": "yes"}
        restored = ConditionSpec.model_validate(data)
        assert restored == spec


class TestConditionEvaluation:
    """Test the condition evaluation logic extracted from the dispatcher."""

    @staticmethod
    def _evaluate(
        upstream_result: dict[str, Any],
        condition: ConditionSpec,
    ) -> bool:
        """Simplified condition evaluator matching dispatcher logic.

        Returns True if the condition is met (task should proceed).
        """
        dispatcher_instance = object.__new__(Dispatcher)
        actual = dispatcher_instance._dig_path(
            upstream_result, condition.field.split(".")
        )
        return str(actual) == condition.equals

    def test_condition_met(self) -> None:
        result = {"verdict": "insufficient"}
        cond = ConditionSpec(node="j", field="verdict", equals="insufficient")
        assert self._evaluate(result, cond) is True

    def test_condition_not_met(self) -> None:
        result = {"verdict": "sufficient"}
        cond = ConditionSpec(node="j", field="verdict", equals="insufficient")
        assert self._evaluate(result, cond) is False

    def test_missing_field_returns_none(self) -> None:
        result: dict[str, Any] = {}
        cond = ConditionSpec(node="j", field="verdict", equals="insufficient")
        assert self._evaluate(result, cond) is False

    def test_nested_field_path(self) -> None:
        result = {"deep": {"nested": "value"}}
        cond = ConditionSpec(node="j", field="deep.nested", equals="value")
        assert self._evaluate(result, cond) is True

    def test_numeric_value_as_string(self) -> None:
        result = {"count": 42}
        cond = ConditionSpec(node="j", field="count", equals="42")
        assert self._evaluate(result, cond) is True


class TestConditionInTaskSpec:
    """Verify condition field works in concrete task spec types."""

    def test_inference_spec_with_condition(self) -> None:
        spec = InferenceSpecStrict(
            taskType=TaskType.INFERENCE,
            dependsOn=["upstream_task"],
            condition=ConditionSpec(
                node="upstream_task", field="verdict", equals="insufficient"
            ),
        )
        assert spec.condition is not None
        assert spec.condition.node == "upstream_task"

    def test_inference_spec_without_condition(self) -> None:
        spec = InferenceSpecStrict(
            taskType=TaskType.INFERENCE, dependsOn=["upstream_task"]
        )
        assert spec.condition is None

    def test_data_retrieval_spec_with_condition(self) -> None:
        spec = DataRetrievalSpecStrict(
            taskType=TaskType.DATA_RETRIEVAL,
            dependsOn=["judge"],
            condition=ConditionSpec(
                node="judge", field="verdict", equals="insufficient"
            ),
        )
        assert spec.condition is not None

    def test_condition_serializes_in_spec(self) -> None:
        spec = InferenceSpecStrict(
            taskType=TaskType.INFERENCE,
            dependsOn=["j"],
            condition=ConditionSpec(node="j", field="v", equals="no"),
        )
        data = spec.model_dump(exclude_none=True)
        assert "condition" in data
        assert data["condition"]["node"] == "j"

    def test_model_rejects_spec_level_depends_on_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="must appear in dependsOn"):
            InferenceSpecStrict(
                taskType=TaskType.INFERENCE,
                dependsOn=["other_task"],
                condition=ConditionSpec(node="missing", field="v", equals="yes"),
            )

    def test_model_skips_check_when_no_spec_level_depends_on(self) -> None:
        spec = InferenceSpecStrict(
            taskType=TaskType.INFERENCE,
            condition=ConditionSpec(node="upstream", field="v", equals="yes"),
        )
        assert spec.condition is not None

    def test_parser_rejects_condition_without_matching_depends_on(self) -> None:
        yaml_payload = """
apiVersion: flowmesh/v1
kind: WorkflowTemplate
metadata:
  name: bad-condition
spec:
  taskType: echo
  graph:
    nodes:
      - name: upstream
        spec:
          taskType: echo
          data:
            type: list
            items: ["a"]
      - name: conditional
        dependsOn: ["upstream"]
        spec:
          taskType: echo
          condition:
            node: missing_node
            field: items.0.output
            equals: "yes"
          data:
            type: list
            items: ["b"]
"""
        with pytest.raises(ValueError, match="must appear in dependsOn"):
            parse_workflow(yaml_payload, "native")
