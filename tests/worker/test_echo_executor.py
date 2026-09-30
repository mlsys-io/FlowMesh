"""Echo executor tests: the literal "list" path and rejection of the removed
"function" mode."""

from pathlib import Path

import pytest

from shared.schemas.result import EchoResult
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


class TestListPath:
    def test_literal_items_are_echoed(self) -> None:
        result = _run({"type": "list", "items": ["a", "b", "c"]})
        assert [i.output for i in result.items] == ["a", "b", "c"]


class TestFunctionModeRejected:
    def test_type_function_is_rejected(self) -> None:
        """A legacy ``type: function`` payload is rejected with a pointer to the
        python task, even when it also carries top-level items."""
        with pytest.raises(ExecutionError, match="use a python task instead"):
            _run(
                {
                    "type": "function",
                    "function": "lambda args: args[0]",
                    "arguments": [{"items": [1, 2, 3]}],
                    "items": ["ignored"],
                }
            )

    def test_type_function_with_items_is_not_run_as_list(self) -> None:
        """A ``type: function`` payload that also has top-level items must not
        silently fall through to list mode."""
        with pytest.raises(ExecutionError, match="use a python task instead"):
            _run({"type": "function", "items": ["a", "b"]})
