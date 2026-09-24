"""Echo executor tests: the literal "list" path and the list-Lambda "function" path."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from shared.schemas.result import EchoItem, EchoResult
from shared.tasks import TaskType
from worker.executors.base_executor import ExecutionError
from worker.executors.echo_executor import EchoExecutor

from .factories import make_worker_config, make_worker_task_message


def _spec(data: dict, upstream: dict | None = None) -> dict:
    spec: dict = {"taskType": "echo", "data": data}
    if upstream:
        spec["_upstreamResults"] = upstream
    return spec


def _run(
    data: dict, upstream: dict | None = None, tmp_path: Path | None = None
) -> EchoResult:
    executor = EchoExecutor(make_worker_config())
    task = make_worker_task_message(
        _spec(data, upstream), task_type=TaskType.ECHO, task_id="tsk-echo"
    )
    return executor.run(task, tmp_path or Path("/tmp/echo-out"))


def _echo_result(*outputs: JsonValue) -> EchoResult:
    return EchoResult(items=[EchoItem(output=o) for o in outputs], count=len(outputs))


class TestListPath:
    def test_literal_items_are_echoed(self) -> None:
        result = _run({"type": "list", "items": ["a", "b", "c"]})
        assert [i.output for i in result.items] == ["a", "b", "c"]


class TestFunctionPath:
    def test_explode_one_input_list_into_rows(self) -> None:
        result = _run(
            {
                "type": "function",
                "function": "lambda args: [x * 2 for x in args[0]]",
                "arguments": [{"items": [1, 2, 3]}],
            }
        )
        assert [i.output for i in result.items] == [2, 4, 6]

    def test_filter_rows(self) -> None:
        result = _run(
            {
                "type": "function",
                "function": "lambda args: [x for x in args[0] if x % 2 == 0]",
                "arguments": [{"items": [1, 2, 3, 4]}],
            }
        )
        assert [i.output for i in result.items] == [2, 4]

    def test_split_one_input_into_two_outputs(self) -> None:
        result = _run(
            {
                "type": "function",
                "function": "lambda args: [args[0][:2], args[0][2:]]",
                "arguments": [{"items": [1, 2, 3, 4]}],
            }
        )
        assert [i.output for i in result.items] == [[1, 2], [3, 4]]

    def test_cross_product_into_groups_of_different_sizes(self) -> None:
        result = _run(
            {
                "type": "function",
                "function": (
                    "lambda args: [[[a, b] for b in args[1]] for a in args[0]]"
                ),
                "arguments": [{"items": [1, 2]}, {"items": ["x", "y", "z"]}],
            }
        )
        assert [i.output for i in result.items] == [
            [[1, "x"], [1, "y"], [1, "z"]],
            [[2, "x"], [2, "y"], [2, "z"]],
        ]

    def test_collapse_groups_back_to_one_row_per_group(self) -> None:
        result = _run(
            {
                "type": "function",
                "function": "lambda args: [sum(g) for g in args[0]]",
                "arguments": [{"items": [[1, 2], [3, 4, 5]]}],
            }
        )
        assert [i.output for i in result.items] == [3, 12]

    def test_node_path_argument_reads_upstream_echo_result(self) -> None:
        upstream = {"echo-a": _echo_result("p", "q")}
        result = _run(
            {
                "type": "function",
                "function": "lambda args: [args[0].upper()]",
                "arguments": [{"node": "echo-a", "path": "items[0].output"}],
            },
            upstream=upstream,
        )
        assert [i.output for i in result.items] == ["P"]

    def test_literal_items_argument(self) -> None:
        result = _run(
            {
                "type": "function",
                "function": "lambda args: [args[0]]",
                "arguments": [{"items": ["a", "b"]}],
            }
        )
        assert [i.output for i in result.items] == [["a", "b"]]

    def test_non_list_return_raises(self) -> None:
        with pytest.raises(ExecutionError, match="must return a list"):
            _run(
                {
                    "type": "function",
                    "function": "lambda args: 'not a list'",
                    "arguments": [{"items": [1]}],
                }
            )

    def test_non_json_element_raises(self) -> None:
        with pytest.raises(ExecutionError, match="function failed"):
            _run(
                {
                    "type": "function",
                    "function": "lambda args: [object()]",
                    "arguments": [{"items": [1]}],
                }
            )

    def test_nested_non_json_element_raises(self) -> None:
        with pytest.raises(ExecutionError, match="function failed"):
            _run(
                {
                    "type": "function",
                    "function": "lambda args: [{'k': set([1])}]",
                    "arguments": [{"items": [1]}],
                }
            )
