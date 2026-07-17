import asyncio
import logging
import socket
from typing import Any, cast
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


def _make_service(
    port_start: int, port_end: int, persistent_listeners: bool = True
) -> PortForwardService:
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
        persistent_listeners=persistent_listeners,
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
    async def test_stop_closes_listener_created_during_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end)
        bound = asyncio.Event()
        continue_start = asyncio.Event()
        original_start_server = asyncio.start_server

        async def delayed_start_server(*args: Any, **kwargs: Any) -> asyncio.Server:
            server = await original_start_server(*args, **kwargs)
            bound.set()
            await continue_start.wait()
            return server

        monkeypatch.setattr(asyncio, "start_server", delayed_start_server)
        starting = asyncio.create_task(svc.start())
        await bound.wait()
        stopping = asyncio.create_task(svc.stop())

        await asyncio.sleep(0)
        assert not stopping.done()

        continue_start.set()
        await starting
        await stopping

        assert not await _tcp_accepts(start)

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
    async def test_connection_dispatches_endpoint_active_when_handler_starts(
        self,
    ) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end)
        old_worker = MagicMock(node_id="nde-old")
        new_worker = MagicMock(node_id="nde-new")
        cast(Any, svc._worker_registry).get_worker_async = AsyncMock(
            return_value=old_worker
        )
        dispatch_started = asyncio.Event()
        continue_uplink = asyncio.Event()
        dispatched_commands = []

        async def delayed_exec_node_cmd(
            node_id: str, command: Any, timeout: float
        ) -> MagicMock:
            assert timeout == 5.0
            dispatched_commands.append((node_id, command))
            dispatch_started.set()
            await continue_uplink.wait()
            return MagicMock(success=False, message="test uplink failure")

        cast(Any, svc._node_registry).exec_node_cmd = AsyncMock(
            side_effect=delayed_exec_node_cmd
        )
        await svc.start()
        try:
            payload = await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-old", 2201)
            )
            reader, writer = await asyncio.open_connection("127.0.0.1", payload["port"])
            await dispatch_started.wait()

            cast(Any, svc._worker_registry).get_worker_async = AsyncMock(
                return_value=new_worker
            )
            await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-new", 2202)
            )
            continue_uplink.set()

            assert await asyncio.wait_for(reader.read(1), timeout=2.0) == b""
            node_id, command = dispatched_commands[0]
            assert node_id == "nde-old"
            assert command.payload["session_id"] == "ssn-old"
            assert command.payload["target_port"] == 2201
            writer.close()
            await writer.wait_closed()
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


class TestSshForwardSessionListeners:
    @pytest.mark.anyio
    async def test_stop_waits_for_dynamic_listener_binding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end, persistent_listeners=False)
        bound = asyncio.Event()
        continue_bind = asyncio.Event()
        original_start_server = asyncio.start_server

        async def delayed_start_server(*args: Any, **kwargs: Any) -> asyncio.Server:
            server = await original_start_server(*args, **kwargs)
            bound.set()
            await continue_bind.wait()
            return server

        monkeypatch.setattr(asyncio, "start_server", delayed_start_server)
        await svc.start()
        registration = asyncio.create_task(
            svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-a", 2201)
            )
        )
        await bound.wait()
        stopping = asyncio.create_task(svc.stop())

        await asyncio.sleep(0)
        assert not stopping.done()

        continue_bind.set()
        with pytest.raises(RuntimeError, match="registration was cancelled"):
            await registration
        await stopping

        assert not await _tcp_accepts(start)

    @pytest.mark.anyio
    async def test_new_registration_waits_for_superseded_listener_binding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end, persistent_listeners=False)
        first_bind_started = asyncio.Event()
        continue_first_bind = asyncio.Event()
        original_start_server = asyncio.start_server
        calls = 0

        async def delayed_first_start_server(
            *args: Any, **kwargs: Any
        ) -> asyncio.Server:
            nonlocal calls
            calls += 1
            server = await original_start_server(*args, **kwargs)
            if calls == 1:
                first_bind_started.set()
                await continue_first_bind.wait()
            return server

        monkeypatch.setattr(asyncio, "start_server", delayed_first_start_server)
        await svc.start()
        try:
            older = asyncio.create_task(
                svc._register_task_async(
                    "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-old", 2201)
                )
            )
            await first_bind_started.wait()

            newer = asyncio.create_task(
                svc._register_task_async(
                    "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-new", 2202)
                )
            )
            await asyncio.sleep(0)
            assert not newer.done()

            continue_first_bind.set()
            with pytest.raises(RuntimeError, match="registration was cancelled"):
                await older
            payload = await newer

            assert payload["port"] == start
            assert svc._sessions["tsk-a"].session_id == "ssn-new"
        finally:
            await svc.stop()

    @pytest.mark.anyio
    async def test_listener_exists_only_while_session_is_registered(self) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end, persistent_listeners=False)
        await svc.start()
        try:
            assert not await _tcp_accepts(start)

            payload = await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-a", 2201)
            )
            assert payload["port"] == start
            assert await _tcp_accepts(start)

            await svc._unregister_task_async("tsk-a")

            assert not await _tcp_accepts(start)
        finally:
            await svc.stop()

    @pytest.mark.anyio
    async def test_registration_skips_an_occupied_port(self) -> None:
        start, end = _free_port_range(2)
        occupied = await asyncio.start_server(
            lambda _reader, _writer: None,
            "127.0.0.1",
            start,
        )
        svc = _make_service(start, end, persistent_listeners=False)
        await svc.start()
        try:
            payload = await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-a", 2201)
            )

            assert payload["port"] == end
        finally:
            await svc.stop()
            occupied.close()
            await occupied.wait_closed()

    @pytest.mark.anyio
    async def test_late_registration_does_not_replace_newer_session(self) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end, persistent_listeners=False)
        lookup_started = asyncio.Event()
        unblock_lookup = asyncio.Event()
        worker = MagicMock()
        worker.node_id = "nde-test"
        calls = 0

        async def get_worker(_: str) -> MagicMock:
            nonlocal calls
            calls += 1
            if calls == 1:
                lookup_started.set()
                await unblock_lookup.wait()
            return worker

        cast(Any, svc._worker_registry).get_worker_async = AsyncMock(
            side_effect=get_worker
        )
        await svc.start()
        try:
            older = asyncio.create_task(
                asyncio.to_thread(
                    svc.register_port_forward,
                    "tsk-a",
                    "wfl-a",
                    "wkr-a",
                    _ssh_info("ssn-old", 2201),
                )
            )
            await lookup_started.wait()

            newer = await svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-new", 2202)
            )
            unblock_lookup.set()

            with pytest.raises(RuntimeError, match="registration was cancelled"):
                await older
            assert svc._sessions["tsk-a"].session_id == "ssn-new"
            assert svc._sessions["tsk-a"].port == newer["port"]
        finally:
            await svc.stop()

    @pytest.mark.anyio
    async def test_timed_out_registration_cannot_publish_late_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end, persistent_listeners=False)
        lookup_started = asyncio.Event()
        unblock_lookup = asyncio.Event()
        worker = MagicMock()
        worker.node_id = "nde-test"

        async def get_worker(_: str) -> MagicMock:
            lookup_started.set()
            await unblock_lookup.wait()
            return worker

        monkeypatch.setattr("server.services.port_forward._DEFAULT_TIMEOUT_SEC", 0.01)
        cast(Any, svc._worker_registry).get_worker_async = AsyncMock(
            side_effect=get_worker
        )
        await svc.start()
        try:
            timed_out = asyncio.create_task(
                asyncio.to_thread(
                    svc.register_port_forward,
                    "tsk-a",
                    "wfl-a",
                    "wkr-a",
                    _ssh_info("ssn-a", 2201),
                )
            )
            await lookup_started.wait()

            with pytest.raises(TimeoutError):
                await timed_out
            unblock_lookup.set()
            await asyncio.sleep(0)

            assert "tsk-a" not in svc._sessions
            assert not await _tcp_accepts(start)
        finally:
            await svc.stop()

    @pytest.mark.anyio
    async def test_stop_invalidates_registration_waiting_for_worker(self) -> None:
        start, end = _free_port_range(1)
        svc = _make_service(start, end, persistent_listeners=False)
        lookup_started = asyncio.Event()
        unblock_lookup = asyncio.Event()
        worker = MagicMock()
        worker.node_id = "nde-test"

        async def get_worker(_: str) -> MagicMock:
            lookup_started.set()
            await unblock_lookup.wait()
            return worker

        cast(Any, svc._worker_registry).get_worker_async = AsyncMock(
            side_effect=get_worker
        )
        await svc.start()
        registration = asyncio.create_task(
            svc._register_task_async(
                "tsk-a", "wfl-a", "wkr-a", _ssh_info("ssn-a", 2201)
            )
        )
        await lookup_started.wait()
        await svc.stop()
        unblock_lookup.set()

        with pytest.raises(RuntimeError, match="registration was cancelled"):
            await registration
        assert not await _tcp_accepts(start)
