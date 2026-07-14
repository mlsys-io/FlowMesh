import asyncio
import logging
import socket
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.services.port_forward import PortForwardService


def _free_port_range(count: int) -> tuple[int, int]:
    """Find a contiguous run of `count` free localhost TCP ports.

    Probes an anchor port from the ephemeral range, then verifies the next
    `count-1` ports above it are also bindable, retrying on collision.
    """
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            start = s.getsockname()[1]
        end = start + count - 1
        held: list[socket.socket] = []
        try:
            for port in range(start, end + 1):
                sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sk.bind(("127.0.0.1", port))
                held.append(sk)
            return start, end
        except OSError:
            continue
        finally:
            for sk in held:
                sk.close()
    raise RuntimeError("Could not find a free contiguous port range")


def _make_service(port_start: int, port_end: int) -> PortForwardService:
    worker = MagicMock()
    worker.node_id = "nde-test"
    worker_registry = MagicMock()
    worker_registry.get_worker_async = AsyncMock(return_value=worker)
    return PortForwardService(
        redis_client=MagicMock(),
        node_registry=MagicMock(),
        worker_registry=worker_registry,
        ssh_audit=None,
        bind_host="127.0.0.1",
        public_host="lum.id",
        port_start=port_start,
        port_end=port_end,
        logger=logging.getLogger("test.ssh_forward"),
    )


async def _tcp_accepts(port: int) -> bool:
    """True if a TCP connection to 127.0.0.1:port is accepted (health-check probe)."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), timeout=2.0
        )
    except OSError:
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    return True


def _ssh_info(session_id: str, target_port: int) -> dict:
    return {
        "session_id": session_id,
        "username": "flowmesh",
        "_relay_target": {"host": "127.0.0.1", "port": target_port},
    }


class TestSshForwardPersistentPorts:
    @pytest.mark.anyio
    async def test_all_ports_listen_before_any_session(self) -> None:
        start, end = _free_port_range(3)
        svc = _make_service(start, end)
        await svc.start()
        try:
            # Every port in the range accepts TCP even though no session exists,
            # so a load-balancer health check always passes.
            for port in range(start, end + 1):
                assert await _tcp_accepts(port), f"port {port} not listening"
        finally:
            await svc.stop()

    @pytest.mark.anyio
    async def test_unassigned_port_connection_is_closed(self) -> None:
        start, end = _free_port_range(2)
        svc = _make_service(start, end)
        await svc.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", start)
            # No session on this port: the handler closes without sending data.
            data = await asyncio.wait_for(reader.read(1), timeout=2.0)
            assert data == b""
            writer.close()
        finally:
            await svc.stop()

    @pytest.mark.anyio
    async def test_register_assigns_port_and_reuses_on_reregister(self) -> None:
        start, end = _free_port_range(3)
        svc = _make_service(start, end)
        await svc.start()
        try:
            payload = await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-a", 2201)
            )
            assert payload["host"] == "lum.id"
            assert payload["mode"] == "forward"
            assert start <= payload["port"] <= end
            first_port = payload["port"]

            # Re-registering the same task keeps its assigned port.
            again = await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-a", 2201)
            )
            assert again["port"] == first_port

            # A different task gets a different port.
            other = await svc._register_task_async(
                "tsk-b", "wfl-b", "wkr-b", _ssh_info("ssn-b", 2202)
            )
            assert other["port"] != first_port
        finally:
            await svc.stop()

    @pytest.mark.anyio
    async def test_unregister_frees_port_but_keeps_listener(self) -> None:
        start, end = _free_port_range(2)
        svc = _make_service(start, end)
        await svc.start()
        try:
            payload = await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-a", 2201)
            )
            port = payload["port"]
            await svc._unregister_task_async("tsk-a")
            # Listener stays bound (health check still passes) ...
            assert await _tcp_accepts(port)
            # ... and the freed port is handed back out to the next task.
            reused = await svc._register_task_async(
                "tsk-c", "wfl-c", "wkr-c", _ssh_info("ssn-c", 2203)
            )
            assert reused["port"] == port
        finally:
            await svc.stop()

    @pytest.mark.anyio
    async def test_pool_exhaustion_raises(self) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end)
        await svc.start()
        try:
            await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-a", 2201)
            )
            with pytest.raises(RuntimeError, match="No available forward ports"):
                await svc._register_task_async(
                    "tsk-b", "wfl-b", "wkr-b", _ssh_info("ssn-b", 2202)
                )
        finally:
            await svc.stop()
