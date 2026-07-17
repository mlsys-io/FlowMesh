import asyncio
import logging
import secrets
from collections.abc import Callable, Coroutine
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any

from shared.schemas.command import CommandMessage, CommandType
from shared.utils import new_ssh_connection_id, now_iso
from shared.utils.encoding import (
    decode_base64_text_to_bytes,
    encode_bytes_to_base64_text,
)

from ..clients.redis import RedisClient, relay_down_key, relay_up_key
from ..registries.node import NodeRegistry
from ..registries.worker import WorkerRegistry
from ..schemas.ssh import SSHConnectionInfo
from .ssh_audit import SshAuditService

_STREAM_MAXLEN = 1000
_READ_CHUNK = 16384
_DEFAULT_TIMEOUT_SEC = 5.0


@dataclass(slots=True)
class _AuditContext:
    """Connection-audit metadata for a forwarded session.

    Fed to the audit service when a client connects; not used by the forwarding path.
    """

    workflow_id: str | None
    worker_id: str
    username: str | None


@dataclass(slots=True, eq=False)
class _Registration:
    _cancelled: Event = field(default_factory=Event, repr=False, compare=False)

    def cancel(self) -> None:
        """Mark the registration as cancelled."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Return True if the registration has been cancelled."""
        return self._cancelled.is_set()


@dataclass(slots=True)
class PortForwardSession:
    task_id: str
    node_id: str
    session_id: str
    target_host: str
    """The worker-internal host to which the connection is forwarded."""
    target_port: int
    """The worker-internal port to which the connection is forwarded."""
    port: int
    """The local port on which the port-forward service listens for this session."""
    server: asyncio.AbstractServer | None
    audit: _AuditContext
    registration: _Registration


class PortForwardService:
    def __init__(
        self,
        redis_client: RedisClient,
        node_registry: NodeRegistry,
        worker_registry: WorkerRegistry,
        ssh_audit: SshAuditService | None,
        bind_host: str,
        public_host: str,
        port_start: int,
        port_end: int,
        persistent_listeners: bool,
        logger: logging.Logger,
    ) -> None:
        self._redis = redis_client
        self._node_registry = node_registry
        self._worker_registry = worker_registry
        self._ssh_audit = ssh_audit
        self._logger = logger
        self._bind_host = bind_host
        self._public_host = public_host
        self._port_start = port_start
        self._port_end = port_end
        self._persistent_listeners = persistent_listeners
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_lock = Lock()
        self._running = False
        self._sessions: dict[str, PortForwardSession] = {}
        self._servers: dict[int, asyncio.AbstractServer] = {}
        self._port_to_task: dict[int, str] = {}
        self._used_ports: set[int] = set()
        self._registrations: dict[str, _Registration] = {}
        self._pending_dynamic: set[_Registration] = set()
        self._no_pending_dynamic = asyncio.Event()
        self._no_pending_dynamic.set()
        self._lifecycle_lock = asyncio.Lock()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the service and optionally bind its listener pool."""
        async with self._lifecycle_lock:
            async with self._lock:
                if self._running:
                    return

            servers: dict[int, asyncio.AbstractServer] = {}
            if self._persistent_listeners:
                try:
                    for port in range(self._port_start, self._port_end + 1):
                        try:
                            server = await asyncio.start_server(
                                self._make_handler(port),
                                host=self._bind_host,
                                port=port,
                            )
                        except OSError as exc:
                            self._logger.warning(
                                "Port forward: cannot bind port %s: %s", port, exc
                            )
                            continue
                        servers[port] = server
                except BaseException:
                    await self._close_servers(list(servers.values()))
                    raise

            async with self._lock:
                self._servers = servers
                self._running = True
                with self._loop_lock:
                    self._loop = asyncio.get_running_loop()

            if self._persistent_listeners:
                self._logger.info(
                    "Port forward listening on %s port(s) %s-%s",
                    len(servers),
                    self._port_start,
                    self._port_end,
                )

    async def stop(self) -> None:
        """Stop listeners and prevent queued registrations from publishing state."""
        async with self._lifecycle_lock:
            async with self._lock:
                self._running = False
                for registration in self._registrations.values():
                    registration.cancel()
                servers = list(self._servers.values())
                servers.extend(
                    session.server
                    for session in self._sessions.values()
                    if session.server is not None
                )
                self._servers.clear()
                self._sessions.clear()
                self._port_to_task.clear()
                self._used_ports.clear()
                self._registrations.clear()
                with self._loop_lock:
                    self._loop = None

            await self._close_servers(servers)
            await self._no_pending_dynamic.wait()

    def _make_handler(
        self, port: int
    ) -> Callable[
        [asyncio.StreamReader, asyncio.StreamWriter], Coroutine[Any, Any, None]
    ]:
        async def _handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await self._handle_client(port, reader, writer)

        return _handler

    def register_port_forward(
        self,
        task_id: str,
        workflow_id: str | None,
        assigned_worker: str,
        endpoint: dict[str, Any],
    ) -> dict[str, Any]:
        loop = self._require_loop()
        registration = _Registration()
        fut = asyncio.run_coroutine_threadsafe(
            self._register_task_async(
                task_id, workflow_id, assigned_worker, endpoint, registration
            ),
            loop,
        )
        try:
            return fut.result(timeout=_DEFAULT_TIMEOUT_SEC)
        except FutureTimeoutError:
            registration.cancel()
            fut.cancel()
            self._schedule_registration_invalidation(task_id, registration, loop)
            raise

    def unregister_task(self, task_id: str) -> None:
        loop = self._get_loop()
        if loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(
            self._unregister_task_async(task_id), loop
        )
        fut.result(timeout=_DEFAULT_TIMEOUT_SEC)

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._get_loop()
        if loop is None:
            raise RuntimeError("Forward service not started")
        return loop

    def _get_loop(self) -> asyncio.AbstractEventLoop | None:
        with self._loop_lock:
            return self._loop

    async def _register_task_async(
        self,
        task_id: str,
        workflow_id: str | None,
        assigned_worker: str,
        endpoint: dict[str, Any],
        registration: _Registration | None = None,
    ) -> dict[str, Any]:
        if registration is None:
            registration = _Registration()
        relay_target = endpoint.get("_relay_target")
        if not isinstance(relay_target, dict):
            raise RuntimeError("Missing relay target for forward-mode task")
        session_id = endpoint.get("session_id")
        if not session_id:
            raise RuntimeError("Missing session_id for forward-mode task")
        target_host = relay_target.get("host")
        target_port = relay_target.get("port")
        if not (target_host and target_port):
            raise RuntimeError("Incomplete relay target for forward-mode task")
        username = (
            str(raw_username)
            if (raw_username := endpoint.get("username")) is not None
            else None
        )

        async with self._lock:
            self._start_registration_locked(task_id, registration)

        try:
            worker = await self._worker_registry.get_worker_async(assigned_worker)
        except BaseException:
            await self._invalidate_registration(task_id, registration)
            raise
        if worker is None:
            await self._invalidate_registration(task_id, registration)
            raise RuntimeError(f"Assigned worker not found: {assigned_worker}")

        audit = _AuditContext(
            workflow_id=workflow_id,
            worker_id=assigned_worker,
            username=username,
        )
        created = False
        async with self._lock:
            self._require_registration_locked(task_id, registration)
            session = self._sessions.get(task_id)
            if session is not None:
                # Update existing session info
                session.node_id = worker.node_id
                session.session_id = str(session_id)
                session.target_host = str(target_host)
                session.target_port = int(target_port)
                session.audit.workflow_id = workflow_id
                session.audit.worker_id = assigned_worker
                session.audit.username = username
                session.registration = registration
            elif self._persistent_listeners:
                session = self._create_persistent_session_locked(
                    task_id,
                    worker.node_id,
                    str(session_id),
                    str(target_host),
                    int(target_port),
                    audit,
                    registration,
                )
                self._sessions[task_id] = session
                self._port_to_task[session.port] = task_id
                created = True

        if session is None:
            session = await self._create_dynamic_session(
                task_id,
                worker.node_id,
                str(session_id),
                str(target_host),
                int(target_port),
                audit,
                registration,
            )
            created = True
        if created:
            self._logger.info(
                "Assigned forward port %s to task %s", session.port, task_id
            )

        updated = dict(endpoint)
        updated["host"] = self._public_host
        updated["port"] = session.port
        updated["mode"] = "forward"
        return updated

    def _start_registration_locked(
        self, task_id: str, registration: _Registration
    ) -> None:
        if not self._running:
            raise RuntimeError("Forward service not started")
        if previous := self._registrations.get(task_id):
            previous.cancel()
        self._registrations[task_id] = registration

    def _require_registration_locked(
        self, task_id: str, registration: _Registration
    ) -> None:
        if (
            not self._running
            or registration.is_cancelled()
            or self._registrations.get(task_id) is not registration
        ):
            raise RuntimeError("Forward registration was cancelled")

    def _create_persistent_session_locked(
        self,
        task_id: str,
        node_id: str,
        session_id: str,
        target_host: str,
        target_port: int,
        audit: _AuditContext,
        registration: _Registration,
    ) -> PortForwardSession:
        for port in range(self._port_start, self._port_end + 1):
            if port in self._servers and port not in self._port_to_task:
                return PortForwardSession(
                    task_id=task_id,
                    node_id=node_id,
                    session_id=session_id,
                    target_host=target_host,
                    target_port=target_port,
                    port=port,
                    audit=audit,
                    registration=registration,
                    server=None,
                )
        raise RuntimeError("No available forward ports")

    async def _create_dynamic_session(
        self,
        task_id: str,
        node_id: str,
        session_id: str,
        target_host: str,
        target_port: int,
        audit: _AuditContext,
        registration: _Registration,
    ) -> PortForwardSession:
        unavailable_ports: set[int] = set()
        while True:
            pending_drained: asyncio.Event | None = None
            async with self._lock:
                self._require_registration_locked(task_id, registration)
                port = next(
                    (
                        candidate
                        for candidate in range(self._port_start, self._port_end + 1)
                        if candidate not in self._used_ports
                        and candidate not in unavailable_ports
                    ),
                    None,
                )
                if port is None:
                    if not self._pending_dynamic:
                        raise RuntimeError("No available forward ports")
                    pending_drained = self._no_pending_dynamic
                else:
                    self._used_ports.add(port)
                    self._pending_dynamic.add(registration)
                    self._no_pending_dynamic.clear()
            if port is None:
                assert pending_drained is not None
                await pending_drained.wait()
                continue
            try:
                server = await asyncio.start_server(
                    self._make_handler(port), host=self._bind_host, port=port
                )
            except OSError:
                unavailable_ports.add(port)
                await self._release_dynamic_port(port, registration)
                continue
            except BaseException:
                await self._release_dynamic_port(port, registration)
                raise

            try:
                async with self._lock:
                    self._require_registration_locked(task_id, registration)
                    session = PortForwardSession(
                        task_id=task_id,
                        node_id=node_id,
                        session_id=session_id,
                        target_host=target_host,
                        target_port=target_port,
                        port=port,
                        audit=audit,
                        registration=registration,
                        server=server,
                    )
                    self._sessions[task_id] = session
                    self._port_to_task[port] = task_id
                    self._complete_dynamic_reservation_locked(registration)
                    return session
            except BaseException:
                await self._close_dynamic_server(server, port, registration)
                raise

    async def _unregister_task_async(self, task_id: str) -> None:
        async with self._lock:
            if registration := self._registrations.pop(task_id, None):
                registration.cancel()
            session = self._remove_session_locked(task_id)
        if session is not None and session.server is not None:
            await self._close_dynamic_server(session.server, session.port)

    async def _invalidate_registration(
        self, task_id: str, registration: _Registration
    ) -> None:
        async with self._lock:
            if self._registrations.get(task_id) is not registration:
                return
            registration.cancel()
            self._registrations.pop(task_id, None)
            session = self._sessions.get(task_id)
            if session is not None and session.registration is registration:
                session = self._remove_session_locked(task_id)
            else:
                session = None
        if session is not None and session.server is not None:
            await self._close_dynamic_server(session.server, session.port)

    def _schedule_registration_invalidation(
        self, task_id: str, registration: _Registration, loop: asyncio.AbstractEventLoop
    ) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                self._invalidate_registration(task_id, registration), loop
            )
        except RuntimeError:
            pass

    def _remove_session_locked(self, task_id: str) -> PortForwardSession | None:
        session = self._sessions.pop(task_id, None)
        if session is None:
            return None
        self._port_to_task.pop(session.port, None)
        return session

    async def _close_dynamic_server(
        self,
        server: asyncio.AbstractServer,
        port: int,
        registration: _Registration | None = None,
    ) -> None:
        try:
            await self._close_servers([server])
        finally:
            await self._release_dynamic_port(port, registration)

    async def _release_dynamic_port(
        self, port: int, registration: _Registration | None
    ) -> None:
        async with self._lock:
            self._used_ports.discard(port)
            if registration is not None:
                self._complete_dynamic_reservation_locked(registration)

    def _complete_dynamic_reservation_locked(self, registration: _Registration) -> None:
        self._pending_dynamic.discard(registration)
        if not self._pending_dynamic:
            self._no_pending_dynamic.set()

    async def _close_servers(self, servers: list[asyncio.AbstractServer]) -> None:
        for server in servers:
            server.close()
        for server in servers:
            try:
                await server.wait_closed()
            except Exception:
                pass

    async def _handle_client(
        self,
        port: int,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        async with self._lock:
            task_id = self._port_to_task.get(port)
            session = self._sessions.get(task_id) if task_id is not None else None
        if session is None or task_id is None:
            # No session occupies this port: an external health-check probe or a
            # connection that arrived before the port was assigned / after it was
            # released. Close cleanly.
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return

        relay_token = secrets.token_hex(32)
        connection_id = new_ssh_connection_id()
        peer = writer.get_extra_info("peername")
        source_ip: str | None = None
        source_port: int | None = None
        if isinstance(peer, tuple):
            raw_ip, *raw_port = peer
            source_ip = str(raw_ip)
            if raw_port:
                try:
                    source_port = int(raw_port[0])
                except Exception:
                    source_port = None
        try:
            await self._start_server_uplink(session, relay_token)
        except Exception as exc:
            self._logger.warning(
                "Failed to start relay uplink for task %s: %s", task_id, exc
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return

        if self._ssh_audit is not None:
            try:
                await self._ssh_audit.register_connection(
                    SSHConnectionInfo(
                        connection_id=connection_id,
                        access_mode="forward",
                        task_id=task_id,
                        workflow_id=session.audit.workflow_id,
                        worker_id=session.audit.worker_id,
                        node_id=session.node_id,
                        session_id=session.session_id,
                        username=(
                            username
                            if (username := session.audit.username) is not None
                            else "flowmesh"
                        ),
                        source_ip=source_ip,
                        source_port=source_port,
                        connected_at=now_iso(),
                    )
                )
            except Exception:
                self._logger.debug(
                    "Failed to register SSH audit connection %s",
                    connection_id,
                    exc_info=True,
                )

        up = relay_up_key(relay_token)
        down = relay_down_key(relay_token)

        async def redis_to_client() -> None:
            last_id = "0"
            while True:
                rows = await self._redis.asyncio.xread_telemetry(
                    {up: last_id}, count=10, block_ms=5000
                )
                if not rows:
                    continue
                for _, entries in rows:
                    for entry_id, fields in entries:
                        last_id = entry_id
                        if "eof" in fields:
                            return
                        raw = fields.get("d")
                        if raw:
                            writer.write(decode_base64_text_to_bytes(raw))
                            await writer.drain()

        async def client_to_redis() -> None:
            try:
                while True:
                    data = await reader.read(_READ_CHUNK)
                    if not data:
                        break
                    await self._redis.asyncio.xadd_telemetry(
                        down,
                        {"d": encode_bytes_to_base64_text(data)},
                        maxlen=_STREAM_MAXLEN,
                        approximate=True,
                    )
            finally:
                try:
                    await self._redis.asyncio.xadd_telemetry(
                        down,
                        {"eof": "1"},
                        maxlen=_STREAM_MAXLEN,
                        approximate=True,
                    )
                except Exception:
                    pass

        t1 = asyncio.create_task(redis_to_client())
        t2 = asyncio.create_task(client_to_redis())
        try:
            _, pending = await asyncio.wait(
                [t1, t2], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            if self._ssh_audit is not None:
                try:
                    await self._ssh_audit.unregister_connection(connection_id)
                except Exception:
                    self._logger.debug(
                        "Failed to unregister SSH audit connection %s",
                        connection_id,
                        exc_info=True,
                    )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _start_server_uplink(
        self, session: PortForwardSession, relay_token: str
    ) -> None:
        """Ask the assigned node to start a relay uplink for this session."""
        cmd = CommandMessage(
            command=CommandType.START_RELAY,
            payload={
                "relay_token": relay_token,
                "target_host": session.target_host,
                "target_port": session.target_port,
                "session_id": session.session_id,
            },
        )
        resp = await self._node_registry.exec_node_cmd(
            session.node_id, cmd, timeout=_DEFAULT_TIMEOUT_SEC
        )
        if not resp.success:
            raise RuntimeError(resp.message or "Server refused START_RELAY")
