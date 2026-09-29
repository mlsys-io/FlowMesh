"""Tests for `TaskRuntime.list_tasks` workflow-scoped filtering."""

import asyncio
import logging
from typing import Any, cast

from server.task.runtime import TaskRuntime

from .merge_harness import WorkerRegistryStub, build_runtime

_PAYLOAD = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: list-tasks
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
      - name: b
        spec:
          taskType: echo
"""


class _BuildSpyRuntime(TaskRuntime):
    """Counts TaskInfo builds to prove non-matching tasks are never materialised."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.build_count = 0

    def _build_task_info_locked(self, task_id: str, record: Any) -> Any:
        self.build_count += 1
        return super()._build_task_info_locked(task_id, record)


def _spy_runtime() -> _BuildSpyRuntime:
    async def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    registry = type(
        "Registry",
        (),
        {
            "register_workflow_async": _noop,
            "commit_transition": lambda *a, **k: None,
            "save_task_states_async": _noop,
            "save_workflow_sched_async": _noop,
        },
    )()
    return _BuildSpyRuntime(
        cast(Any, registry), cast(Any, WorkerRegistryStub()), logging.getLogger("t")
    )


async def _register(runtime: TaskRuntime, payload: str) -> str:
    workflow_id, _ = await runtime.register("owner", "org", payload, format="native")
    return workflow_id


def test_list_tasks_filters_by_workflow_id() -> None:
    runtime, _ = build_runtime()
    wf_a = asyncio.run(_register(runtime, _PAYLOAD))
    wf_b = asyncio.run(_register(runtime, _PAYLOAD))

    tasks_a = runtime.list_tasks(workflow_id=wf_a)
    assert {t.task_id for t in tasks_a} == set(
        runtime._tasks[task_id].task_id
        for task_id in runtime._tasks
        if runtime._tasks[task_id].workflow_id == wf_a
    )
    assert all(t.workflow_id == wf_a for t in tasks_a)
    assert len(tasks_a) == 2
    assert wf_b not in {t.workflow_id for t in tasks_a}


def test_list_tasks_without_filter_returns_everything() -> None:
    runtime, _ = build_runtime()
    wf_a = asyncio.run(_register(runtime, _PAYLOAD))
    wf_b = asyncio.run(_register(runtime, _PAYLOAD))

    tasks = runtime.list_tasks()
    assert {t.workflow_id for t in tasks} == {wf_a, wf_b}
    assert len(tasks) == 4


def test_workflow_filter_does_not_build_non_matching_tasks() -> None:
    runtime = _spy_runtime()
    wf_a = asyncio.run(_register(runtime, _PAYLOAD))
    asyncio.run(_register(runtime, _PAYLOAD))  # second workflow, 2 more tasks

    runtime.build_count = 0
    tasks = runtime.list_tasks(workflow_id=wf_a)

    assert len(tasks) == 2
    assert runtime.build_count == 2
