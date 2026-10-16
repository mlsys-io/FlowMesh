"""A restarted server re-registers running forward-mode sessions on their own ports."""

import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.services.port_forward import PortForwardService
from server.task.models import TaskStatus
from tests.server.test_ssh_forward_ports import (
    _free_port_range,
    _make_service,
    _ssh_info,
)


def _record(
    port: int | None, status: str = TaskStatus.DISPATCHED, mode: str = "forward"
) -> Any:
    ssh: dict[str, Any] = {
        "session_id": "ssn-1",
        "username": "flowmesh",
        "mode": mode,
        "host": "lum.id",
    }
    if port is not None:
        ssh["port"] = port
    return SimpleNamespace(
        status=status,
        assigned_worker="wkr-1",
        workflow_id="wfl-1",
        latest_update={"ssh": ssh},
    )


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.anyio
async def test_restored_session_keeps_its_port(persistent: bool) -> None:
    start, end = _free_port_range(3)
    svc = _make_service(start, end, persistent_listeners=persistent)
    await svc.start()
    try:
        update = MagicMock()
        await svc.restore_forwards(cast(Any, {"tsk-old": _record(start + 2)}), update)
        assert svc._port_to_task[start + 2] == "tsk-old"
        update.assert_not_called()

        # The next new session must not be handed the restored session's port.
        new = await svc._register_task_async(
            "tsk-new", "wfl-2", "wkr-1", _ssh_info("ssn-2")
        )
        assert new["port"] != start + 2
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_taken_port_is_reassigned_and_the_endpoint_rewritten() -> None:
    start, end = _free_port_range(2)
    svc = _make_service(start, end)
    await svc.start()
    try:
        await svc._register_task_async("tsk-live", "wfl-2", "wkr-1", _ssh_info("ssn-2"))
        taken = next(p for p, t in svc._port_to_task.items() if t == "tsk-live")
        update = MagicMock()
        await svc.restore_forwards(cast(Any, {"tsk-old": _record(taken)}), update)
        moved = next(p for p, t in svc._port_to_task.items() if t == "tsk-old")
        assert moved != taken
        task_id, payload = update.call_args.args
        assert task_id == "tsk-old" and payload["ssh"]["port"] == moved
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_only_active_forward_sessions_are_restored() -> None:
    start, end = _free_port_range(3)
    svc = _make_service(start, end)
    await svc.start()
    try:
        records = {
            "tsk-done": _record(start, status=TaskStatus.DONE),
            "tsk-proxy": _record(start + 1, mode="proxy"),
        }
        await svc.restore_forwards(cast(Any, records), MagicMock())
        assert svc._port_to_task == {}
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_worker_not_yet_registered_is_retried() -> None:
    start, end = _free_port_range(2)
    worker = MagicMock()
    worker.node_id = "nde-test"
    registry = MagicMock()
    registry.get_worker_async = AsyncMock(side_effect=[None, None, worker])
    svc = PortForwardService(
        redis_client=MagicMock(),
        node_registry=MagicMock(),
        worker_registry=registry,
        ssh_connections=None,
        bind_host="127.0.0.1",
        public_host="lum.id",
        port_start=start,
        port_end=end,
        persistent_listeners=True,
        logger=logging.getLogger("test.forward_restore"),
    )
    await svc.start()
    try:
        await svc.restore_forwards(
            cast(Any, {"tsk-old": _record(start + 1)}),
            MagicMock(),
            attempts=5,
            retry_delay_sec=0,
        )
        assert svc._port_to_task.get(start + 1) == "tsk-old"
    finally:
        await svc.stop()
