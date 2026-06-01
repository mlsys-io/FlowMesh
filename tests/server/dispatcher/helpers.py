"""Shared helpers for dispatcher tests."""

import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from server.dispatcher import Dispatcher


class CapturingDispatcher(Dispatcher):
    """Dispatcher that records _fail_task / _requeue_task calls instead of acting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failed: list[tuple[str, str, dict[str, Any]]] = []
        self.requeued: list[tuple[str, dict[str, Any]]] = []

    def _fail_task(self, task_id: str, error_message: str, **kwargs: Any) -> None:
        self.failed.append((task_id, error_message, kwargs))

    def _requeue_task(self, task_id: str, **kwargs: Any) -> None:
        self.requeued.append((task_id, kwargs))


class WorkflowRegistryStub:
    """Minimal workflow registry for driving TaskRuntime.register in tests."""

    async def register_workflow_async(self, workflow_id: str, tasks: list[Any]) -> None:
        return None

    def mark_task_dispatched(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_done(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_failed(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_pending(self, workflow_id: str, *task_ids: str) -> None: ...
    def mark_task_cancelled(self, workflow_id: str, *task_ids: str) -> None: ...
    def save_task_states(self, items: Any) -> None: ...
    async def save_task_states_async(self, items: Any) -> None: ...
    def save_workflow_sched(
        self, wid: str, in_epoch_order: bool, frontier: int
    ) -> None: ...  # noqa: E501
    async def save_workflow_sched_async(
        self, wid: str, in_epoch_order: bool, frontier: int
    ) -> None: ...  # noqa: E501


def make_capturing_dispatcher(
    runtime: Any = None,
    idle_ids: list[str] | None = None,
    satisfying_ids: list[str] | None = None,
    grace_sec: int = 60,
) -> CapturingDispatcher:
    """Build a CapturingDispatcher whose registry returns the given worker ids."""
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [
        SimpleNamespace(id=wid) for wid in (idle_ids or [])
    ]
    registry.satisfying_workers.return_value = [
        SimpleNamespace(id=wid) for wid in (satisfying_ids or [])
    ]
    return CapturingDispatcher(
        runtime=runtime if runtime is not None else mock.Mock(),
        worker_registry=registry,
        results_dir=Path(tempfile.gettempdir()),
        logger=logging.getLogger("dispatcher-test"),
        no_worker_grace_sec=grace_sec,
    )
