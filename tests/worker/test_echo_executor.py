"""Echo executor tests: the literal "list" path."""

from pathlib import Path

from shared.schemas.result import EchoResult
from shared.tasks import TaskType
from worker.executors.base_executor import RunControl
from worker.executors.echo_executor import EchoExecutor

from .factories import make_worker_config, make_worker_task_message


def _spec(data: dict, upstream: dict | None = None) -> dict:
    spec: dict = {"taskType": "echo", "data": data}
    if upstream:
        spec["_upstreamResults"] = upstream
    return spec


def _run(data: dict, out_dir: Path, upstream: dict | None = None) -> EchoResult:
    executor = EchoExecutor(make_worker_config())
    task = make_worker_task_message(
        _spec(data, upstream), task_type=TaskType.ECHO, task_id="tsk-echo"
    )
    return executor.run(task, out_dir, RunControl(task.task_id))


def test_literal_items_are_echoed(tmp_path: Path) -> None:
    result = _run({"type": "list", "items": ["a", "b", "c"]}, tmp_path)
    assert [i.output for i in result.items] == ["a", "b", "c"]
