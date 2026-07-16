import asyncio
import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends
from fastapi import Path as ApiPath
from fastapi import Request, WebSocket, WebSocketDisconnect, status

from shared.schemas.command import CommandMessage, CommandType
from shared.utils import new_ssh_connection_id, now_iso
from shared.utils.encoding import (
    decode_base64_text_to_bytes,
    encode_bytes_to_base64_text,
)
from shared.utils.json import safe_get

from ...app_state import (
    get_logger,
    get_node_registry,
    get_redis_client,
    get_runtime,
    get_ssh_audit,
    get_ssh_proxy_enabled,
    get_worker_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    authenticate_websocket,
    require_permission,
)
from ...clients.redis import RedisClient, relay_down_key, relay_up_key
from ...hooks import ResourceAction, ResourceKind
from ...registries.node import NodeRegistry
from ...registries.worker import Worker, WorkerRegistry
from ...schemas.ssh import SSHConnectionInfo
from ...services.ssh_audit import SshAuditService
from ...task.models import TaskRecord
from ...task.runtime import TaskRuntime
from ...utils.misc import filter_models_by_queries

router = APIRouter(prefix="/ssh", tags=["SSH"])

_STREAM_MAXLEN = 1000


async def _start_server_uplink(
    record: TaskRecord,
    node_registry: NodeRegistry,
    worker_registry: WorkerRegistry,
    relay_token: str,
) -> Worker:
    worker_id = record.assigned_worker
    if not worker_id:
        raise RuntimeError("SSH relay task has no assigned worker")
    worker = await worker_registry.get_worker_async(worker_id)
    if worker is None:
        raise RuntimeError(f"Assigned worker not found: {worker_id}")
    ssh_info = safe_get(record.latest_update, "ssh")
    if not isinstance(ssh_info, dict):
        raise RuntimeError("Missing SSH info in latest_update")
    relay_target = ssh_info.get("_relay_target")
    if not isinstance(relay_target, dict):
        raise RuntimeError("Missing relay target in latest_update")
    session_id = ssh_info.get("session_id")
    if not session_id:
        raise RuntimeError("Missing SSH session_id in latest_update")

    cmd = CommandMessage(
        command=CommandType.START_RELAY,
        payload={
            "relay_token": relay_token,
            "target_host": relay_target.get("host"),
            "target_port": relay_target.get("port"),
            "session_id": session_id,
        },
    )
    resp = await node_registry.exec_node_cmd(worker.node_id, cmd, timeout=5.0)
    if not resp.success:
        raise RuntimeError(resp.message or "Server refused START_RELAY")
    return worker


@router.websocket("/tasks/{task_id}/proxy")
async def ssh_proxy(
    websocket: WebSocket,
    task_id: str = ApiPath(..., min_length=1),
    principal: PrincipalContext = Depends(authenticate_websocket),
    runtime: TaskRuntime = Depends(get_runtime),
    redis_client: RedisClient = Depends(get_redis_client),
    logger: logging.Logger = Depends(get_logger),
    proxy_enabled: bool = Depends(get_ssh_proxy_enabled),
    node_registry: NodeRegistry = Depends(get_node_registry),
    worker_registry: WorkerRegistry = Depends(get_worker_registry),
    ssh_audit: SshAuditService | None = Depends(get_ssh_audit),
) -> None:
    """Proxy an SSH session over WebSocket.

    Connects a client to an active SSH task published in ``proxy`` mode.
    The server asks the assigned node to start an uplink for this
    connection, then relays SSH bytes between the client WebSocket and Redis
    Streams.

    Authentication: bearer token from the ``Authorization`` header, or
    ``?token=...`` query param for browser clients that can't set headers.

    Close behavior:
    - ``4401``: missing or invalid bearer token
    - ``4403``: proxy access disabled, or principal not authorized for the task
    - ``4404``: task not found
    - ``1011``: relay/uplink unavailable
    """
    try:
        await require_permission(
            principal, ResourceKind.TASK, task_id, ResourceAction.READ, logger
        )
    except Exception:
        await websocket.close(code=4403, reason="forbidden")
        return

    if not proxy_enabled:
        await websocket.close(code=4403, reason="proxy disabled")
        return

    record = runtime.get_record(task_id)
    if record is None:
        await websocket.close(code=4404, reason="task not found")
        return

    relay_token = secrets.token_hex(32)
    connection_id = new_ssh_connection_id()
    try:
        worker = await _start_server_uplink(
            record, node_registry, worker_registry, relay_token
        )
    except Exception as exc:
        logger.warning(
            "Failed to ensure SSH relay uplink for task %s: %s", task_id, exc
        )
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR, reason="relay unavailable"
        )
        return

    up = relay_up_key(relay_token)
    down = relay_down_key(relay_token)

    await websocket.accept()
    if ssh_audit is not None:
        ssh_info = safe_get(record.latest_update, "ssh")
        session_id = safe_get(ssh_info, "session_id")
        username = safe_get(ssh_info, "username")
        client = websocket.client
        if client is None:
            source_ip = source_port = None
        else:
            source_ip, source_port = client.host, client.port
        try:
            await ssh_audit.register_connection(
                SSHConnectionInfo(
                    connection_id=connection_id,
                    access_mode="proxy",
                    task_id=task_id,
                    workflow_id=record.workflow_id,
                    worker_id=worker.id,
                    node_id=worker.node_id,
                    session_id=str(session_id) if session_id else None,
                    username=str(username) if username else None,
                    source_ip=source_ip,
                    source_port=source_port,
                    connected_at=now_iso(),
                )
            )
        except Exception:
            logger.debug(
                "Failed to register SSH audit connection %s",
                connection_id,
                exc_info=True,
            )
    logger.info("SSH relay started: task=%s", task_id)

    async def redis_to_client() -> None:
        last_id = "0"
        while True:
            rows: Any = await redis_client.asyncio.xread_telemetry(
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
                        await websocket.send_bytes(decode_base64_text_to_bytes(raw))

    async def client_to_redis() -> None:
        try:
            while True:
                data = await websocket.receive_bytes()
                await redis_client.asyncio.xadd_telemetry(
                    down,
                    {"d": encode_bytes_to_base64_text(data)},
                    maxlen=_STREAM_MAXLEN,
                    approximate=True,
                )
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await redis_client.asyncio.xadd_telemetry(
                    down, {"eof": "1"}, maxlen=_STREAM_MAXLEN, approximate=True
                )
            except Exception:
                pass

    t1 = asyncio.create_task(redis_to_client())
    t2 = asyncio.create_task(client_to_redis())
    try:
        _, pending = await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        for name, key in (("up", up), ("down", down)):
            try:
                await redis_client.asyncio.delete_telemetry(key)
            except Exception:
                logger.debug(
                    "Failed to delete SSH %s stream for task %s",
                    name,
                    task_id,
                    exc_info=True,
                )
        if ssh_audit is not None:
            try:
                await ssh_audit.unregister_connection(connection_id)
            except Exception:
                logger.debug(
                    "Failed to unregister SSH audit connection %s",
                    connection_id,
                    exc_info=True,
                )
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("SSH relay ended: task=%s", task_id)


@router.get(
    "/connections",
    summary="List SSH connections",
    description="List active SSH connections.",
    response_description="List of active SSH connection records.",
)
async def list_ssh_connections(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    ssh_audit: SshAuditService | None = Depends(get_ssh_audit),
    logger: logging.Logger = Depends(get_logger),
) -> list[SSHConnectionInfo]:
    await require_permission(
        principal, ResourceKind.SYSTEM, None, ResourceAction.ADMIN, logger
    )
    if ssh_audit is None:
        return []
    connections = await ssh_audit.list_connections()
    return filter_models_by_queries(connections, request.query_params)
