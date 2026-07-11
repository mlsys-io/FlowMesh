import asyncio
import logging
import secrets
from dataclasses import dataclass
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


@dataclass(slots=True)
class _AuditContext:
    """Connection-audit metadata for a forwarded session.

    Fed to the audit service when a client connects; not used by the forwarding path.
    """

    workflow_id: str | None
    worker_id: str
    username: str | None


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
    server: asyncio.AbstractServer
    audit: _AuditContext


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
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sessions: dict[str, PortForwardSession] = {}
        self._used_ports: set[int] = set()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._used_ports.clear()
            self._loop = None
        for session in sessions:
            await self._close_session(session, release_port=False)

    def register_port_forward(
        self,
        task_id: str,
        workflow_id: str | None,
        assigned_worker: str,
        endpoint: dict[str, Any],
    ) -> dict[str, Any]:
        loop = self._require_loop()
        fut = asyncio.run_coroutine_threadsafe(
            self._register_task_async(task_id, workflow_id, assigned_worker, endpoint),
            loop,
        )
        return fut.result(timeout=5.0)

    def unregister_task(self, task_id: str) -> None:
        loop = self._loop
        if loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(
            self._unregister_task_async(task_id), loop
        )
        fut.result(timeout=5.0)

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is None:
            raise RuntimeError("Forward service not started")
        return loop

    async def _register_task_async(
        self,
        task_id: str,
        workflow_id: str | None,
        assigned_worker: str,
        endpoint: dict[str, Any],
    ) -> dict[str, Any]:
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
        worker = await self._worker_registry.get_worker_async(assigned_worker)
        if worker is None:
            raise RuntimeError(f"Assigned worker not found: {assigned_worker}")
        username = (
            str(raw_username)
            if (raw_username := endpoint.get("username")) is not None
            else None
        )

        async with self._lock:
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
                updated = dict(endpoint)
                updated["host"] = self._public_host
                updated["port"] = session.port
                updated["mode"] = "forward"
                return updated

        # Create new session for this task
        session = await self._create_session(
            task_id=task_id,
            node_id=worker.node_id,
            session_id=str(session_id),
            target_host=str(target_host),
            target_port=int(target_port),
            audit=_AuditContext(
                workflow_id=workflow_id,
                worker_id=assigned_worker,
                username=username,
            ),
        )

        stale_session: PortForwardSession | None = None
        async with self._lock:
            existing = self._sessions.get(task_id)
            if existing is None:
                self._sessions[task_id] = session
            else:
                # Another session was created; update it with the new info.
                existing.node_id = worker.node_id
                existing.session_id = str(session_id)
                existing.target_host = str(target_host)
                existing.target_port = int(target_port)
                existing.audit.workflow_id = workflow_id
                existing.audit.worker_id = assigned_worker
                existing.audit.username = username
                stale_session = session
                session = existing
        if stale_session is not None:
            await self._close_session(stale_session)

        updated = dict(endpoint)
        updated["host"] = self._public_host
        updated["port"] = session.port
        updated["mode"] = "forward"
        return updated

    async def _create_session(
        self,
        task_id: str,
        node_id: str,
        session_id: str,
        target_host: str,
        target_port: int,
        audit: _AuditContext,
    ) -> PortForwardSession:
        async def _client_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await self._handle_client(task_id, reader, writer)

        bind_host = self._bind_host
        for port in range(self._port_start, self._port_end + 1):
            if not await self._reserve_port(port):
                continue
            try:
                server = await asyncio.start_server(
                    _client_handler, host=bind_host, port=port
                )
            except OSError:
                await self._release_port(port)
                continue
            self._logger.info("Allocated forward port %s for task %s", port, task_id)
            return PortForwardSession(
                task_id=task_id,
                node_id=node_id,
                session_id=session_id,
                target_host=target_host,
                target_port=target_port,
                port=port,
                server=server,
                audit=audit,
            )
        raise RuntimeError("No available forward ports")

    async def _unregister_task_async(self, task_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(task_id, None)
            if session is None:
                return
        await self._close_session(session)

    async def _reserve_port(self, port: int) -> bool:
        async with self._lock:
            if port in self._used_ports:
                return False
            self._used_ports.add(port)
            return True

    async def _release_port(self, port: int) -> None:
        async with self._lock:
            self._used_ports.discard(port)

    async def _close_session(
        self, session: PortForwardSession, release_port: bool = True
    ) -> None:
        if release_port:
            await self._release_port(session.port)
        session.server.close()
        try:
            await session.server.wait_closed()
        except Exception:
            pass

    async def _handle_client(
        self,
        task_id: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        async with self._lock:
            session = self._sessions.get(task_id)
        if session is None:
            # Task unregistered while client was connecting
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
            session.node_id, cmd, timeout=5.0
        )
        if not resp.success:
            raise RuntimeError(resp.message or "Server refused START_RELAY")
