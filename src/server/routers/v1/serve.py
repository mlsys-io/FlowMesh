import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as ApiPath
from fastapi import Request, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from shared.schemas.command import CommandMessage, CommandType
from shared.tasks import TaskType
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
    get_ssh_proxy_enabled,
    get_worker_registry,
)
from ...clients.redis import RedisClient, relay_down_key, relay_up_key
from ...registries.node import NodeRegistry
from ...registries.worker import Worker, WorkerRegistry
from ...task.models import TaskRecord, TaskStatus
from ...task.runtime import TaskRuntime

router = APIRouter(prefix="/serve", tags=["Serve"])

_STREAM_MAXLEN = 1000
_READ_CHUNK = 16384
_MAX_PROXY_BODY_BYTES = 100 * 1024 * 1024

# Hop-by-hop headers: managed by the ASGI server, not forwarded.
_HOP_BY_HOP_REQUEST_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",  # codespell:ignore te
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
_HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",  # codespell:ignore te
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _resolve_serve_relay_target(
    runtime: TaskRuntime, task_id: str
) -> tuple[TaskRecord, str, int]:
    """Resolve the serve task's relay endpoint.

    The target comes only from task state, so caller input cannot choose an arbitrary
    upstream host or port.
    """
    record = runtime.get_record(task_id)
    if record is None or record.task_type != TaskType.SERVE:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "serve task not found")
    if record.status != TaskStatus.DISPATCHED:
        raise HTTPException(status.HTTP_409_CONFLICT, "serve task is not running")
    serve_info = safe_get(record.latest_update, "serve")
    if not isinstance(serve_info, dict):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "serve task has no endpoint info yet"
        )
    if serve_info.get("mode") != "proxy":
        raise HTTPException(
            status.HTTP_409_CONFLICT, "serve task is not in proxy access mode"
        )
    relay_target = serve_info.get("_relay_target")
    if not isinstance(relay_target, dict):
        raise HTTPException(status.HTTP_409_CONFLICT, "serve task has no relay target")
    target_host = relay_target.get("host")
    target_port = relay_target.get("port")
    if not target_host or not target_port:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "serve task relay target is incomplete"
        )
    return record, str(target_host), int(target_port)


async def _start_serve_uplink(
    record: TaskRecord,
    node_registry: NodeRegistry,
    worker_registry: WorkerRegistry,
    relay_token: str,
    target_host: str,
    target_port: int,
) -> Worker:
    worker_id = record.assigned_worker
    if not worker_id:
        raise RuntimeError("Serve relay task has no assigned worker")
    worker = await worker_registry.get_worker_async(worker_id)
    if worker is None:
        raise RuntimeError(f"Assigned worker not found: {worker_id}")
    cmd = CommandMessage(
        command=CommandType.START_RELAY,
        payload={
            "relay_token": relay_token,
            "target_host": target_host,
            "target_port": target_port,
            "session_id": record.task_id,
        },
    )
    resp = await node_registry.exec_node_cmd(worker.node_id, cmd, timeout=5.0)
    if not resp.success:
        raise RuntimeError(resp.message or "Server refused START_RELAY")
    return worker


async def _pump_from_relay(
    redis_client: RedisClient, up_key: str, reader: asyncio.StreamReader
) -> None:
    """Copy relay bytes into ``reader``.

    A missing ``up`` stream is eof only after the stream has existed once; before that
    the worker may simply not have produced data yet, assuming that the key cannot be
    deleted before the first read.
    """
    last_id = "0"
    seen = False
    try:
        while True:
            rows: Any = await redis_client.asyncio.xread_telemetry(
                {up_key: last_id}, count=10, block_ms=5000
            )
            if not rows:
                if await redis_client.asyncio.exists_telemetry(up_key):
                    seen = True
                    continue
                if seen:
                    reader.feed_eof()
                    return
                continue
            seen = True
            for _, entries in rows:
                for entry_id, fields in entries:
                    last_id = entry_id
                    if "eof" in fields:
                        reader.feed_eof()
                        return
                    raw = fields.get("d")
                    if raw:
                        reader.feed_data(decode_base64_text_to_bytes(raw))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        reader.set_exception(exc)


async def _send_to_relay(redis_client: RedisClient, down_key: str, data: bytes) -> None:
    await redis_client.asyncio.xadd_telemetry(
        down_key,
        {"d": encode_bytes_to_base64_text(data)},
        maxlen=_STREAM_MAXLEN,
        approximate=True,
    )


async def _signal_relay_eof(redis_client: RedisClient, down_key: str) -> None:
    """Signal relay eof so the worker stops forwarding."""
    try:
        await redis_client.asyncio.xadd_telemetry(
            down_key, {"eof": "1"}, maxlen=_STREAM_MAXLEN, approximate=True
        )
    except Exception:
        pass


async def _delete_relay_streams(
    redis_client: RedisClient, up_key: str, down_key: str
) -> None:
    """Remove relay streams after server-side teardown."""
    try:
        await redis_client.asyncio.delete_telemetry(up_key)
    except Exception:
        pass
    try:
        await redis_client.asyncio.delete_telemetry(down_key)
    except Exception:
        pass


class _UplinkCleanup:
    """Run per-request relay cleanup exactly once.

    Cleanup is shielded so cancellation by one caller cannot interrupt eof signaling
    for the relay.
    """

    def __init__(
        self,
        pump_task: asyncio.Task[None],
        redis_client: RedisClient,
        up_key: str,
        down_key: str,
    ) -> None:
        self._pump_task = pump_task
        self._redis_client = redis_client
        self._up_key = up_key
        self._down_key = down_key
        self._task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._cleanup_once())
        await asyncio.shield(self._task)

    async def _cleanup_once(self) -> None:
        self._pump_task.cancel()
        await _signal_relay_eof(self._redis_client, self._down_key)
        try:
            await self._pump_task
        except (asyncio.CancelledError, Exception):
            pass
        await _delete_relay_streams(self._redis_client, self._up_key, self._down_key)


def _connection_nominated_headers(raw_connection: str) -> set[str]:
    """RFC 7230 lets ``Connection`` name additional per-hop headers to drop, e.g.
    ``Connection: X-Foo`` makes ``X-Foo`` hop-by-hop for this hop only."""
    return {
        token.strip().lower() for token in raw_connection.split(",") if token.strip()
    }


def _header_values(headers: list[tuple[str, str]], name: str) -> list[str]:
    """Return all values for a repeated HTTP header name."""
    lower_name = name.lower()
    return [
        value for header_name, value in headers if header_name.lower() == lower_name
    ]


async def _read_capped_body(request: Request) -> bytes:
    """Read a bounded request body.

    This PAT-exempt route must reject oversized unauthenticated bodies before buffering
    them for the upstream vLLM server.
    """
    declared_length = request.headers.get("content-length")
    if declared_length is not None:
        if not declared_length.isdigit():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid content-length")
        declared_length_bytes = int(declared_length)
        if declared_length_bytes > _MAX_PROXY_BODY_BYTES:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE, "request body too large"
            )

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_PROXY_BODY_BYTES:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE, "request body too large"
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _serialize_request(
    request: Request, upstream_path: str, target_host: str, target_port: int
) -> bytes:
    """Serialize the request for the relay target.

    Force ``Connection: close`` so relay eof cleanly marks the end of the upstream
    response.
    """
    body = await _read_capped_body(request)
    path = "/" + upstream_path
    if request.url.query:
        path += "?" + request.url.query
    dynamic_hop_by_hop = _connection_nominated_headers(
        ",".join(request.headers.getlist("connection"))
    )
    headers = [
        (name, value)
        for name, value in request.headers.items()
        if name.lower() not in _HOP_BY_HOP_REQUEST_HEADERS
        and name.lower() not in dynamic_hop_by_hop
    ]
    headers.append(("Host", f"{target_host}:{target_port}"))
    headers.append(("Connection", "close"))
    headers.append(("Content-Length", str(len(body))))
    header_text = "".join(f"{name}: {value}\r\n" for name, value in headers)
    head = f"{request.method} {path} HTTP/1.1\r\n{header_text}\r\n"
    return head.encode("latin-1") + body


async def _read_response_head(
    reader: asyncio.StreamReader,
) -> tuple[int, list[tuple[str, str]]]:
    status_line = await reader.readline()
    if not status_line:
        raise RuntimeError("Empty response from upstream vLLM server")
    parts = status_line.decode("latin-1").strip().split(" ", 2)
    if len(parts) < 2:
        raise RuntimeError(f"Malformed status line: {status_line!r}")
    status_code = int(parts[1])
    headers: list[tuple[str, str]] = []
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        name, _, value = line.decode("latin-1").partition(":")
        headers.append((name.strip(), value.strip()))
    return status_code, headers


async def _iter_fixed_length_body(
    reader: asyncio.StreamReader, length: int
) -> AsyncIterator[bytes]:
    remaining = length
    while remaining > 0:
        chunk = await reader.read(min(_READ_CHUNK, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        yield chunk


async def _iter_until_eof_body(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while True:
        chunk = await reader.read(_READ_CHUNK)
        if not chunk:
            break
        yield chunk


async def _iter_chunked_body(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Yield decoded bytes from a chunked HTTP/1.1 body.

    The outbound ``StreamingResponse`` should choose client framing instead of
    forwarding upstream chunk markers as payload.
    """
    while True:
        size_line = await reader.readline()
        size_str = size_line.split(b";", 1)[0].strip()
        size = int(size_str, 16)
        if size == 0:
            while True:
                trailer = await reader.readline()
                if trailer in (b"\r\n", b"\n", b""):
                    break
            return
        data = await reader.readexactly(size)
        yield data
        await reader.readexactly(2)  # trailing CRLF after each chunk's data


@router.api_route(
    "/tasks/{task_id}/{upstream_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    summary="Serve task HTTP proxy",
    description=(
        "Reverse-proxy an HTTP request to a serve task's vLLM server over the "
        "relay uplink. PAT-exempt: authenticated solely by the vLLM api-key "
        "the client sends as `Authorization`, forwarded to vLLM untouched."
    ),
)
async def serve_proxy(
    request: Request,
    task_id: str = ApiPath(..., min_length=1),
    upstream_path: str = ApiPath(...),
    runtime: TaskRuntime = Depends(get_runtime),
    redis_client: RedisClient = Depends(get_redis_client),
    logger: logging.Logger = Depends(get_logger),
    proxy_enabled: bool = Depends(get_ssh_proxy_enabled),
    node_registry: NodeRegistry = Depends(get_node_registry),
    worker_registry: WorkerRegistry = Depends(get_worker_registry),
) -> StreamingResponse:
    if not proxy_enabled:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "proxy disabled")

    record, target_host, target_port = _resolve_serve_relay_target(runtime, task_id)

    relay_token = secrets.token_hex(32)
    try:
        await _start_serve_uplink(
            record,
            node_registry,
            worker_registry,
            relay_token,
            target_host,
            target_port,
        )
    except Exception as exc:
        logger.warning(
            "Failed to start serve relay uplink for task %s: %s", task_id, exc
        )
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "relay unavailable") from exc

    up = relay_up_key(relay_token)
    down = relay_down_key(relay_token)

    reader = asyncio.StreamReader()
    pump_task = asyncio.create_task(_pump_from_relay(redis_client, up, reader))
    cleanup = _UplinkCleanup(pump_task, redis_client, up, down)
    response_ready = False

    try:
        request_bytes = await _serialize_request(
            request, upstream_path, target_host, target_port
        )
        await _send_to_relay(redis_client, down, request_bytes)

        # No timeout: a slow non-streaming generation is bounded only by relay eof.
        status_code, headers = await _read_response_head(reader)

        lower_headers = {name.lower(): value for name, value in headers}
        raw_content_length = lower_headers.get("content-length")
        content_length: int | None = None
        if raw_content_length is not None:
            if raw_content_length.isdigit():
                content_length = int(raw_content_length)
            else:
                logger.warning(
                    "Serve proxy upstream returned malformed Content-Length "
                    "%r for task %s; reading until EOF instead",
                    raw_content_length,
                    task_id,
                )
        is_chunked = "chunked" in lower_headers.get("transfer-encoding", "").lower()
        dynamic_response_hop_by_hop = _connection_nominated_headers(
            ",".join(_header_values(headers, "connection"))
        )
        response_headers = {
            name: value
            for name, value in headers
            if name.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS
            and name.lower() not in dynamic_response_hop_by_hop
        }

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                if is_chunked:
                    async for chunk in _iter_chunked_body(reader):
                        yield chunk
                elif content_length is not None:
                    async for chunk in _iter_fixed_length_body(reader, content_length):
                        yield chunk
                else:
                    async for chunk in _iter_until_eof_body(reader):
                        yield chunk
            finally:
                await cleanup.run()

        # Background cleanup covers responses abandoned before body iteration starts.
        response = StreamingResponse(
            body_iter(),
            status_code=status_code,
            headers=response_headers,
            background=BackgroundTask(cleanup.run),
        )
        response_ready = True
        return response
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Serve proxy upstream error for task %s: %s", task_id, exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "upstream error") from exc
    finally:
        if not response_ready:
            await cleanup.run()
