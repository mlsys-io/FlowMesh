"""A restarted root serves each running forward session again on its published port,
and relays its connections to the worker's current node."""

import asyncio
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.services.port_forward import PortForwardService
from server.task.models import TaskRecord, TaskStatus
from shared.tasks import TaskType
from tests.server.task.test_ssh_result_mounting import _task_template
from tests.server.test_ssh_forward_ports import (
    _free_port_range,
    _make_service,
    _ssh_info,
)


def _session(
    task_id: str,
    port: int | None,
    status: str = TaskStatus.DISPATCHED,
    mode: str = "forward",
    key: str = "ssh",
    assigned_worker: str | None = "wkr-1",
) -> TaskRecord:
    endpoint: dict[str, Any] = {"mode": mode, "host": "lum.id"}
    if key == "ssh":
        endpoint.update(session_id=f"ssn-{task_id}", username="flowmesh")
    if port is not None:
        endpoint["port"] = port
    task_type = TaskType.SSH if key == "ssh" else TaskType.SERVE
    return TaskRecord(
        task_id=task_id,
        workflow_id="wfl-1",
        owner_id="owner",
        source="raw",
        task=_task_template(task_type),
        status=status,
        task_type=task_type.value,
        assigned_worker=assigned_worker,
        latest_update={key: endpoint},
    )


def _port_of(svc: PortForwardService, task_id: str) -> int | None:
    return next((p for p, t in svc._port_to_task.items() if t == task_id), None)


@contextmanager
def _listening(port: int) -> Iterator[None]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen()
        yield
    finally:
        sock.close()


@pytest.mark.anyio
@pytest.mark.parametrize("persistent", [True, False])
async def test_a_restored_session_keeps_its_published_port(persistent: bool) -> None:
    start, end = _free_port_range(3)
    svc = _make_service(start, end, persistent_listeners=persistent)
    await svc.start()
    try:
        await svc.restore_sessions([_session("tsk-1", end)])
        assert _port_of(svc, "tsk-1") == end

        new = await svc._register_task_async(
            "tsk-new", "wfl-2", "wkr-1", _ssh_info("ssn-new")
        )
        assert new["port"] != end
    finally:
        await svc.stop()


@pytest.mark.anyio
@pytest.mark.parametrize("persistent", [True, False])
async def test_a_session_whose_port_another_process_holds_stays_unreachable(
    persistent: bool,
) -> None:
    start, end = _free_port_range(2)
    svc = _make_service(start, end, persistent_listeners=persistent)
    with _listening(end):
        await svc.start()
        try:
            await svc.restore_sessions([_session("tsk-1", end)])

            assert _port_of(svc, "tsk-1") is None
            assert "tsk-1" not in svc._sessions
            assert "tsk-1" not in svc._registrations
        finally:
            await svc.stop()


@pytest.mark.anyio
async def test_a_restored_session_does_not_take_a_live_session_s_port() -> None:
    start, end = _free_port_range(2)
    svc = _make_service(start, end)
    await svc.start()
    try:
        live = await svc._register_task_async(
            "tsk-live", "wfl-2", "wkr-1", _ssh_info("ssn-live")
        )
        await svc.restore_sessions([_session("tsk-1", live["port"])])

        assert svc._port_to_task == {live["port"]: "tsk-live"}
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_a_port_outside_the_range_is_not_reclaimed() -> None:
    start, end = _free_port_range(2)
    svc = _make_service(start, end, persistent_listeners=False)
    await svc.start()
    try:
        await svc.restore_sessions([_session("tsk-1", end + 1)])

        assert svc._port_to_task == {}
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_only_a_running_task_s_forward_session_is_restored() -> None:
    start, end = _free_port_range(6)
    svc = _make_service(start, end, persistent_listeners=False)
    await svc.start()
    try:
        await svc.restore_sessions(
            [
                _session("tsk-forward", start),
                _session("tsk-proxy", start + 1, mode="proxy"),
                _session("tsk-done", start + 2, status=TaskStatus.DONE),
                _session("tsk-pending", start + 3, status=TaskStatus.PENDING),
                _session("tsk-unassigned", start + 4, assigned_worker=None),
                _session("tsk-no-port", None),
            ]
        )

        assert svc._port_to_task == {start: "tsk-forward"}
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_a_serve_session_is_restored_under_its_task_id() -> None:
    start, end = _free_port_range(2)
    svc = _make_service(start, end)
    await svc.start()
    try:
        await svc.restore_sessions([_session("tsk-serve", end, key="serve")])

        assert _port_of(svc, "tsk-serve") == end
        assert svc._sessions["tsk-serve"].session_id == "tsk-serve"
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_a_session_of_an_unregistered_worker_does_not_stop_the_others() -> None:
    start, end = _free_port_range(2)
    svc = _make_service(start, end)
    registry = cast(Any, svc._worker_registry)
    registry.get_worker_async = AsyncMock(
        side_effect=lambda worker_id: (
            None if worker_id == "wkr-gone" else SimpleNamespace(node_id="nde-1")
        )
    )
    await svc.start()
    try:
        await svc.restore_sessions(
            [
                _session("tsk-gone", start, assigned_worker="wkr-gone"),
                _session("tsk-1", end),
            ]
        )

        assert svc._port_to_task == {end: "tsk-1"}
    finally:
        await svc.stop()


@pytest.mark.anyio
async def test_a_connection_is_relayed_to_the_worker_s_current_node() -> None:
    start, end = _free_port_range(1)
    svc = _make_service(start, end)
    await svc.start()
    try:
        await svc.restore_sessions([_session("tsk-1", start)])
        cast(Any, svc._worker_registry).get_worker_async = AsyncMock(
            return_value=SimpleNamespace(node_id="nde-rebound")
        )
        exec_node_cmd = AsyncMock(return_value=MagicMock(success=False))
        cast(Any, svc._node_registry).exec_node_cmd = exec_node_cmd

        reader, writer = await asyncio.open_connection("127.0.0.1", start)
        assert await asyncio.wait_for(reader.read(1), timeout=2.0) == b""
        writer.close()

        node_id, command = exec_node_cmd.call_args.args
        assert node_id == "nde-rebound"
        assert command.payload["endpoint_id"] == "ssn-tsk-1"
    finally:
        await svc.stop()
