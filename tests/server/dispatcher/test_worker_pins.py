"""Worker-pin policy across registration, dispatch, merging, and rehydration."""

import asyncio
import logging
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
import yaml

from server.dispatcher import Dispatcher
from server.registries.worker import Worker, WorkerRegistry
from server.services.worker_deaths import WorkerDeathMarks
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from tests.server.task.test_runtime_rehydrate import FakeWorkflowRegistry

type PinSetup = tuple[TaskRuntime, Dispatcher, mock.Mock]


def _register(
    runtime: TaskRuntime,
    pins: list[str] | None = None,
    fallback: bool | None = False,
    task_type: str = "inference",
) -> str:
    spec: dict[str, Any] = {"taskType": task_type}
    if task_type == "inference":
        spec["model"] = {"source": {"identifier": "llama"}}
    if task_type == "ssh":
        spec.update(interactive=False, command=["true"])
    hint: dict[str, Any] = {"selected_worker": pins}
    if fallback is not None:
        hint["allow_worker_fallback"] = fallback
    payload = yaml.safe_dump(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "worker-pin", "annotations": {"schedule_hint": hint}},
            "spec": spec,
        }
    )
    _, tasks = asyncio.run(runtime.register("owner", "org", payload))
    return tasks[0].task_id


@pytest.fixture
def setup(tmp_path: Path) -> PinSetup:
    registry = mock.Mock(spec=WorkerRegistry)
    worker = Worker(
        id="wkr-new",
        namespace="test",
        cluster="test",
        node_id="nde-1",
        node_alias="test-node",
    )
    registry.worker_exists.side_effect = lambda wid: wid in {"wkr-new", "wkr-live"}
    registry.is_worker_stale.return_value = False
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1
    runtime = TaskRuntime(
        cast(Any, FakeWorkflowRegistry()),
        registry,
        logging.getLogger("worker-pins"),
        worker_pin_fallback=False,
        worker_deaths=WorkerDeathMarks(),
    )
    dispatcher = Dispatcher(
        runtime,
        registry,
        tmp_path,
        logging.getLogger("worker-pins"),
        enable_context_reuse=False,
        worker_selection_strategy="first_fit",
        no_worker_grace_sec=0,
    )
    return runtime, dispatcher, registry


@pytest.mark.parametrize("task_type", ["inference", "ssh", "echo"])
def test_strict_absent_pin_never_publishes(setup: PinSetup, task_type: str) -> None:
    runtime, dispatcher, registry = setup
    task_id = _register(runtime, ["wkr-old"], task_type=task_type)
    dispatcher.dispatch_once(task_id)
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.status == TaskStatus.FAILED
    assert record.assigned_worker is None
    assert record.attempts == 0
    registry.publish_task.assert_not_called()


def test_opted_in_singleton_inference_dispatches_with_default_merging(
    setup: PinSetup,
) -> None:
    runtime, dispatcher, registry = setup
    task_id = _register(runtime, ["wkr-old"], fallback=True)
    assert dispatcher.dispatch_once(task_id)
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.status == TaskStatus.DISPATCHED
    assert record.assigned_worker == "wkr-new"
    assert record.selected_worker == ["wkr-old"]
    assert record.attempts == 0
    registry.publish_task.assert_called_once()


@pytest.mark.parametrize(
    ("pins", "fallback", "merges"),
    [
        (["wkr-old"], True, True),
        (["wkr-old"], False, False),
        (["wkr-old", "wkr-other-old"], False, False),
        (["wkr-live"], True, False),
        (["wkr-live", "wkr-old"], True, False),
        (["wkr-new"], False, True),
        (None, False, True),
    ],
)
def test_merge_siblings_obey_their_own_pins(
    setup: PinSetup, pins: list[str] | None, fallback: bool, merges: bool
) -> None:
    runtime, dispatcher, registry = setup
    parent = _register(runtime, ["wkr-old"], fallback=True)
    child = _register(runtime, pins, fallback=fallback)
    assert dispatcher.dispatch_once(parent)
    message = registry.publish_task.call_args.args[1]
    assert [item.task_id for item in message.merged_children or []] == (
        [child] if merges else []
    )
    record = runtime.get_record(child)
    assert record is not None
    assert record.status == (TaskStatus.DISPATCHED if merges else TaskStatus.PENDING)
    assert record.selected_worker == pins


def test_strict_parent_can_merge_an_opted_in_stale_sibling(setup: PinSetup) -> None:
    runtime, dispatcher, registry = setup
    parent = _register(runtime, ["wkr-new"])
    child = _register(runtime, ["wkr-old"], fallback=True)
    assert dispatcher.dispatch_once(parent)
    message = registry.publish_task.call_args.args[1]
    assert [item.task_id for item in message.merged_children or []] == [child]


def test_merge_excludes_a_sibling_that_failed_on_the_selected_worker(
    setup: PinSetup,
) -> None:
    runtime, dispatcher, registry = setup
    parent = _register(runtime, ["wkr-old"], fallback=True)
    child = _register(runtime, ["wkr-other-old"], fallback=True)
    record = runtime.get_record(child)
    assert record is not None
    record.failed_workers.append("wkr-new")
    assert dispatcher.dispatch_once(parent)
    message = registry.publish_task.call_args.args[1]
    assert not message.merged_children
    assert record.status == TaskStatus.PENDING
    assert record.assigned_worker is None
    assert record.attempts == 0


def test_opted_in_registered_pin_waits_without_spending_attempts(
    setup: PinSetup,
) -> None:
    runtime, dispatcher, registry = setup
    task_id = _register(runtime, ["wkr-live", "wkr-old"], fallback=True)
    registry.satisfying_workers.return_value.append(
        registry.idle_satisfying_pool.return_value[0].model_copy(
            update={"id": "wkr-live"}
        )
    )
    assert dispatcher.dispatch_once(task_id) is False
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    registry.publish_task.assert_not_called()


@pytest.mark.parametrize(
    ("merge", "exists"), [(False, [False, True]), (True, [False, False, True])]
)
def test_pin_registration_race_requeues_without_spending_attempts(
    setup: PinSetup, merge: bool, exists: list[bool]
) -> None:
    runtime, dispatcher, registry = setup
    dispatcher._task_merge_enabled = merge
    task_id = _register(runtime, ["wkr-old"], fallback=True)
    registry.worker_exists.side_effect = exists
    assert dispatcher.dispatch_once(task_id) is False
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert record.merged_children is None
    registry.publish_task.assert_not_called()


def test_opted_in_stale_pin_preserves_failed_worker_exclusion(setup: PinSetup) -> None:
    runtime, dispatcher, registry = setup
    task_id = _register(runtime, ["wkr-old"], fallback=True)
    record = runtime.get_record(task_id)
    assert record is not None
    record.failed_workers.append("wkr-new")
    dispatcher.dispatch_once(task_id)
    assert record.status == TaskStatus.FAILED
    registry.publish_task.assert_not_called()


def test_fallback_policy_and_merge_index_survive_rehydration(setup: PinSetup) -> None:
    runtime, _, registry = setup
    parent = _register(runtime, ["wkr-old"], fallback=True)
    child = _register(runtime, ["wkr-other-old"], fallback=True)
    restored = TaskRuntime(
        runtime._workflow_registry, registry, logging.getLogger("restored-pins")
    )
    assert asyncio.run(restored.rehydrate()) == 2
    assert restored.plan_merge(parent, 4, "wkr-new") == [child]
    record = restored.get_record(parent)
    assert record is not None
    assert record.selected_worker == ["wkr-old"]
    assert (
        record.task.metadata is not None
        and record.task.metadata.annotations is not None
    )
    hint = record.task.metadata.annotations.schedule_hint
    assert hint is not None and hint.allow_worker_fallback
    assert record.allow_worker_fallback


@pytest.mark.parametrize(
    ("server_default", "hint", "expected"),
    [
        (True, None, True),
        (False, None, False),
        (True, False, False),
        (False, True, True),
    ],
)
def test_explicit_hint_overrides_the_server_default(
    setup: PinSetup, server_default: bool, hint: bool | None, expected: bool
) -> None:
    runtime, _, _ = setup
    runtime._worker_pin_fallback = server_default
    record = runtime.get_record(_register(runtime, ["wkr-old"], fallback=hint))
    assert record is not None
    assert record.allow_worker_fallback is expected


def test_server_default_is_fixed_at_submission(setup: PinSetup) -> None:
    runtime, _, registry = setup
    runtime._worker_pin_fallback = True
    task_id = _register(runtime, ["wkr-old"], fallback=None)
    restored = TaskRuntime(
        runtime._workflow_registry,
        registry,
        logging.getLogger("restored-default"),
        worker_pin_fallback=False,
    )
    asyncio.run(restored.rehydrate())
    record = restored.get_record(task_id)
    assert record is not None
    assert record.allow_worker_fallback


def test_server_default_relaxes_an_unhinted_absent_pin(setup: PinSetup) -> None:
    runtime, dispatcher, registry = setup
    runtime._worker_pin_fallback = True
    task_id = _register(runtime, ["wkr-old"], fallback=None, task_type="echo")
    assert dispatcher.dispatch_once(task_id)
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.assigned_worker == "wkr-new"
    registry.publish_task.assert_called_once()


@pytest.mark.parametrize("rehydrate", [False, True])
def test_transformers_peers_dispatch_independently(
    setup: PinSetup, rehydrate: bool
) -> None:
    runtime, dispatcher, registry = setup
    spec = {
        "taskType": "inference",
        "model": {
            "source": {"identifier": "tiny"},
            "transformers": {"device_map": "cpu"},
        },
    }
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
            "spec": {
                "graph": {
                    "nodes": [{"name": name, "spec": spec} for name in ["a", "b"]]
                }
            },
        }
    )
    _, tasks = asyncio.run(runtime.register("owner", "org", payload))
    if rehydrate:
        for task in tasks:
            record = runtime.get_record(task.task_id)
            assert record is not None
            record.merge_key = "legacy-transformers-merge-key"
            runtime._persist_locked(task.task_id)
        restored = TaskRuntime(
            runtime._workflow_registry,
            registry,
            logging.getLogger("restored-transformers"),
        )
        assert asyncio.run(restored.rehydrate()) == 1
        dispatcher._runtime = restored
    for task in tasks:
        assert dispatcher.dispatch_once(task.task_id)
    assert registry.publish_task.call_count == 2
    for call in registry.publish_task.call_args_list:
        assert not call.args[1].merged_children


def _deaths(runtime: TaskRuntime) -> WorkerDeathMarks:
    deaths = runtime._worker_deaths
    assert deaths is not None
    return deaths


@pytest.mark.parametrize(("fallback", "falls_back"), [(True, True), (False, False)])
def test_dead_pinned_worker_relaxes_only_an_opted_in_pin(
    setup: PinSetup, fallback: bool, falls_back: bool
) -> None:
    runtime, dispatcher, registry = setup
    task_id = _register(runtime, ["wkr-live"], fallback=fallback, task_type="echo")
    _deaths(runtime).mark("wkr-live")
    assert dispatcher.dispatch_once(task_id) is falls_back
    record = runtime.get_record(task_id)
    assert record is not None
    if falls_back:
        assert record.assigned_worker == "wkr-new"
    else:
        assert record.status == TaskStatus.FAILED
        registry.publish_task.assert_not_called()


def test_stale_pinned_worker_waits_for_the_watchdog_verdict(setup: PinSetup) -> None:
    runtime, dispatcher, registry = setup
    registry.is_worker_stale.side_effect = lambda wid: wid == "wkr-live"
    task_id = _register(runtime, ["wkr-live"], fallback=True, task_type="echo")

    # An expired heartbeat alone (e.g. right after a root restart) keeps the pin
    # and does not burn the no-worker grace.
    assert dispatcher.dispatch_once(task_id) is False
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.status == TaskStatus.PENDING
    assert record.no_eligible_since is None
    assert record.attempts == 0
    registry.publish_task.assert_not_called()

    _deaths(runtime).mark("wkr-live")
    assert dispatcher.dispatch_once(task_id)
    assert record.assigned_worker == "wkr-new"


def test_stale_pinned_worker_without_watchdog_uses_the_grace(setup: PinSetup) -> None:
    runtime, dispatcher, registry = setup
    runtime._worker_deaths = WorkerDeathMarks(enabled=False)
    registry.is_worker_stale.side_effect = lambda wid: wid == "wkr-live"
    task_id = _register(runtime, ["wkr-live"], fallback=True, task_type="echo")
    dispatcher.dispatch_once(task_id)
    record = runtime.get_record(task_id)
    assert record is not None
    assert record.status == TaskStatus.FAILED


def test_plan_merge_checks_each_pinned_worker_once(setup: PinSetup) -> None:
    runtime, _, registry = setup
    parent = _register(runtime, ["wkr-old"], fallback=True)
    children = [_register(runtime, ["wkr-old"], fallback=True) for _ in range(3)]
    registry.worker_exists.reset_mock()
    assert runtime.plan_merge(parent, 4, "wkr-new") == children
    assert registry.worker_exists.call_count == 1
