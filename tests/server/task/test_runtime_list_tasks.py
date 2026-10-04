"""Tests for `TaskRuntime.list_tasks` workflow-scoped filtering."""

import asyncio
import logging
import time
from typing import Any, cast
from unittest import mock

from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from shared.utils.redact import REDACTED, redact_raw_yaml

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

_CREDENTIAL_PAYLOAD = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: list-tasks
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: api
          api:
            headers:
              Authorization: Bearer SECRET
      - name: b
        spec:
          taskType: api
          api:
            headers:
              Authorization: Bearer SECRET
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

    tasks_a = runtime.list_tasks(workflow_ids=[wf_a])
    assert {t.task_id for t in tasks_a} == set(
        runtime._tasks[task_id].task_id
        for task_id in runtime._tasks
        if runtime._tasks[task_id].workflow_id == wf_a
    )
    assert all(t.workflow_id == wf_a for t in tasks_a)
    assert len(tasks_a) == 2
    assert wf_b not in {t.workflow_id for t in tasks_a}


def test_list_tasks_filters_by_multiple_workflow_ids() -> None:
    runtime, _ = build_runtime()
    wf_a = asyncio.run(_register(runtime, _PAYLOAD))
    wf_b = asyncio.run(_register(runtime, _PAYLOAD))

    tasks = runtime.list_tasks(workflow_ids=[wf_a, wf_b])
    assert {t.workflow_id for t in tasks} == {wf_a, wf_b}
    assert len(tasks) == 4


def test_list_tasks_filters_by_status() -> None:
    runtime, _ = build_runtime()
    asyncio.run(_register(runtime, _PAYLOAD))

    tasks = runtime.list_tasks(statuses=["PENDING"])
    assert {t.task_id for t in tasks} == set(runtime._tasks)
    assert all(t.status == "PENDING" for t in tasks)

    tasks = runtime.list_tasks(statuses=["DONE"])
    assert tasks == []


def test_list_tasks_without_filter_returns_everything() -> None:
    runtime, _ = build_runtime()
    wf_a = asyncio.run(_register(runtime, _PAYLOAD))
    wf_b = asyncio.run(_register(runtime, _PAYLOAD))

    tasks = runtime.list_tasks()
    assert {t.workflow_id for t in tasks} == {wf_a, wf_b}
    assert len(tasks) == 4


def test_task_statuses_returns_statuses_without_building_task_infos() -> None:
    runtime = _spy_runtime()
    asyncio.run(_register(runtime, _PAYLOAD))

    runtime.build_count = 0
    statuses = runtime.task_statuses()

    assert statuses == {
        task_id: record.status for task_id, record in runtime._tasks.items()
    }
    assert runtime.build_count == 0


def test_workflow_filter_does_not_build_non_matching_tasks() -> None:
    runtime = _spy_runtime()
    wf_a = asyncio.run(_register(runtime, _PAYLOAD))
    asyncio.run(_register(runtime, _PAYLOAD))  # second workflow, 2 more tasks

    runtime.build_count = 0
    tasks = runtime.list_tasks(workflow_ids=[wf_a])

    assert len(tasks) == 2
    assert runtime.build_count == 2


def test_source_redacted_once_per_workflow_at_registration() -> None:
    runtime, _ = build_runtime()
    expected = redact_raw_yaml(_CREDENTIAL_PAYLOAD)

    with mock.patch(
        "server.task.runtime.redact_raw_yaml", wraps=redact_raw_yaml
    ) as spy:
        wf_a = asyncio.run(_register(runtime, _CREDENTIAL_PAYLOAD))
        assert spy.call_count == 1
        asyncio.run(_register(runtime, _CREDENTIAL_PAYLOAD))
        assert spy.call_count == 2

    # Every record in the workflow shares the same redacted source.
    sources = {
        runtime._tasks[task_id].source
        for task_id in runtime._tasks
        if runtime._tasks[task_id].workflow_id == wf_a
    }
    assert sources == {expected}
    assert "SECRET" not in expected

    # Listing does not re-redact: no credential in the built TaskInfos.
    with mock.patch(
        "server.task.runtime.redact_raw_yaml", wraps=redact_raw_yaml
    ) as spy:
        infos = runtime.list_tasks(workflow_ids=[wf_a])
        assert spy.call_count == 0
    for info in infos:
        assert info.source == expected
        assert REDACTED in info.source
        assert "SECRET" not in info.source


def test_register_parses_off_the_event_loop() -> None:
    runtime, _ = build_runtime()

    def _slow_parse(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.5)
        return parse_workflow(*args, **kwargs)

    async def _run() -> tuple[str, list[Any]]:
        t0 = asyncio.get_running_loop().time()
        register_task = asyncio.create_task(
            runtime.register("owner", "org", _PAYLOAD, format="native")
        )
        await asyncio.sleep(0.05)  # let register start and block on the parse
        # A concurrent coroutine completes while register is awaiting the parse.
        await asyncio.sleep(0)
        elapsed = asyncio.get_running_loop().time() - t0
        workflow_id, results = await register_task
        assert elapsed < 0.3
        return workflow_id, results

    with mock.patch("server.task.runtime.parse_workflow", side_effect=_slow_parse):
        workflow_id, results = asyncio.run(_run())

    assert workflow_id
    assert results


def test_persisted_and_rehydrated_dump_hold_no_raw_credential() -> None:
    persisted: list[Any] = []

    async def _save(self: Any, items: Any) -> None:
        persisted.extend(items)

    async def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    registry = type(
        "Registry",
        (),
        {
            "register_workflow_async": _noop,
            "commit_transition": lambda *a, **k: None,
            "save_task_states_async": _save,
            "save_workflow_sched_async": _noop,
        },
    )()
    runtime = TaskRuntime(
        cast(Any, registry), cast(Any, WorkerRegistryStub()), logging.getLogger("t")
    )
    wf_a = asyncio.run(_register(runtime, _CREDENTIAL_PAYLOAD))

    # The persisted JSON carries only the redacted source.
    for item in persisted:
        blob = item.model_dump_json()
        assert "SECRET" not in blob
        assert REDACTED in blob

    # A rehydrated record (built from the persisted JSON) dumps clean too.
    from server.registries.workflow import PersistedTask

    rehydrated = [
        PersistedTask.model_validate_json(item.model_dump_json()) for item in persisted
    ]
    for item in rehydrated:
        assert item.record.workflow_id == wf_a
        dumped = item.record.model_dump()
        assert "SECRET" not in dumped["source"]
        assert REDACTED in dumped["source"]
