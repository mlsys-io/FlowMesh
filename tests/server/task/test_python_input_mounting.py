"""Dispatcher resolution of a python task's input stages."""

import logging
from pathlib import Path
from typing import cast

import pytest

from server.dispatcher.base import Dispatcher, StageReferenceNotReady
from server.registries.worker import WorkerRegistry
from server.task.models import TaskRecord, TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.result import BaseExecutorResult, ResultEnvelope, write_result
from shared.tasks import TaskType
from shared.tasks.specs import PythonSpecStrict
from tests.server.task.test_ssh_result_mounting import _DummyRuntime, _task_template

CODE = "def main(inputs):\n    return len(inputs)\n"


def _record(task_id: str, name: str, status: str, **spec: object) -> TaskRecord:
    task_type = TaskType.PYTHON if spec else TaskType.ECHO
    payload = {"code": CODE, **spec} if spec else {}
    return TaskRecord(
        task_id=task_id,
        workflow_id="wf-1",
        owner_id="owner",
        source="raw",
        task=_task_template(task_type, **payload),
        status=status,
        task_type=task_type.value,
        local_name=name,
    )


def _dispatcher(
    records: list[TaskRecord], deps: list[str], results_dir: Path = Path("/tmp")
) -> Dispatcher:
    current = records[-1]
    return Dispatcher(
        runtime=cast(
            TaskRuntime,
            _DummyRuntime(
                {r.task_id: r for r in records}, depends_on={current.task_id: deps}
            ),
        ),
        worker_registry=cast(WorkerRegistry, object()),
        results_dir=results_dir,
        logger=logging.getLogger("test-python-inputs"),
    )


def _spec(record: TaskRecord) -> PythonSpecStrict:
    return PythonSpecStrict.model_validate(record.task.spec.model_dump())


def test_no_inputs_mounts_every_direct_dependency(tmp_path: Path) -> None:
    a = _record("t-a", "prep", TaskStatus.DONE)
    b = _record("t-b", "raw", TaskStatus.DONE)
    py = _record("t-py", "score", TaskStatus.PENDING, entrypoint="main")
    for identifier in ("t-a", "t-b"):
        write_result(
            tmp_path, ResultEnvelope(task_id=identifier, result=BaseExecutorResult())
        )
    dispatcher = _dispatcher([a, b, py], ["t-a", "t-b"], tmp_path)
    assert dispatcher._resolve_upstream_task_ids(py, _spec(py)) == {
        "prep": "t-a",
        "raw": "t-b",
    }


def test_explicit_inputs_are_resolved_like_ssh(tmp_path: Path) -> None:
    a = _record("t-a", "prep", TaskStatus.DONE)
    b = _record("t-b", "raw", TaskStatus.DONE)
    py = _record("t-py", "score", TaskStatus.PENDING, inputs=[{"stage": "raw"}])
    for identifier in ("t-a", "t-b"):
        write_result(
            tmp_path, ResultEnvelope(task_id=identifier, result=BaseExecutorResult())
        )
    dispatcher = _dispatcher([a, b, py], ["t-a", "t-b"], tmp_path)
    assert dispatcher._resolve_upstream_task_ids(py, _spec(py)) == {"raw": "t-b"}


def test_no_dependencies_means_no_inputs() -> None:
    py = _record("t-py", "score", TaskStatus.PENDING, entrypoint="main")
    assert _dispatcher([py], [])._resolve_upstream_task_ids(py, _spec(py)) is None


def test_unfinished_dependency_requeues() -> None:
    a = _record("t-a", "prep", TaskStatus.DISPATCHED)
    py = _record("t-py", "score", TaskStatus.PENDING, entrypoint="main")
    with pytest.raises(StageReferenceNotReady):
        _dispatcher([a, py], ["t-a"])._resolve_upstream_task_ids(py, _spec(py))


PRICE_CODE = 'def main(inputs):\n    total = 3.5\n    return f"${total:.2f}"\n'


def test_code_is_not_a_placeholder_template(tmp_path: Path) -> None:
    a = _record("t-a", "prep", TaskStatus.DONE)
    write_result(tmp_path, ResultEnvelope(task_id="t-a", result=BaseExecutorResult()))
    py = _record("t-py", "score", TaskStatus.PENDING, code=PRICE_CODE)
    dispatcher = _dispatcher([a, py], ["t-a"], tmp_path)

    assert not py.task.has_placeholder()
    rendered = dispatcher._resolve_stage_references("t-py", py.task, py)
    assert isinstance(rendered.spec, PythonSpecStrict)
    assert rendered.spec.code == PRICE_CODE


def test_placeholders_outside_code_still_resolve(tmp_path: Path) -> None:
    a = _record("t-a", "prep", TaskStatus.DONE)
    write_result(tmp_path, ResultEnvelope(task_id="t-a", result=BaseExecutorResult()))
    py = _record(
        "t-py",
        "score",
        TaskStatus.PENDING,
        code=PRICE_CODE,
        env={"UPSTREAM": "${prep.task_id}"},
    )
    dispatcher = _dispatcher([a, py], ["t-a"], tmp_path)

    assert py.task.has_placeholder()
    rendered = dispatcher._resolve_stage_references("t-py", py.task, py)
    assert isinstance(rendered.spec, PythonSpecStrict)
    assert rendered.spec.env == {"UPSTREAM": "t-a"}
    assert rendered.spec.code == PRICE_CODE
