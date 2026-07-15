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
_RESPONSE_HEAD_TIMEOUT_SEC = 30.0
_MAX_PROXY_BODY_BYTES = 100 * 1024 * 1024

# Headers that describe the hop to the upstream vLLM server, not the resource
# itself; forwarding them verbatim would either duplicate framing the ASGI
# server manages itself or leak a stale connection-management directive.
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
    """Resolve the vLLM relay target for a live serve task.

    The target host/port always comes from the resolved task's own
    ``latest_update`` (never from caller input), so a client can only ever
    reach the vLLM server belonging to the task it names.
    """
    record = runtime.get_record(task_id)
    if record is None or record.task_type != TaskType.SERVE:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "serve task not found")
    if record.status != TaskStatus.DISPATCHED:
        raise HTTPException(status.HTTP_409_CONFLICT, "serve task is not running")
    serve_info = safe_get(record.latest_update, "serve")
    if not isinstance(serve_info, dict) or serve_info.get("mode") != "proxy":
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
    """Feed bytes read from the relay's ``up`` stream into ``reader``."""
    last_id = "0"
    try:
        while True:
            rows: Any = await redis_client.asyncio.xread_telemetry(
                {up_key: last_id}, count=10, block_ms=5000
            )
            if not rows:
                continue
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
    """Tell the worker-side relay to stop forwarding and close its TCP socket.

    Without this, a client disconnect or a proxy-side error leaves the
    worker's `redis_to_tcp` loop blocked on `xread` indefinitely and vLLM's
    generation for that request running unaborted.
    """
    try:
        await redis_client.asyncio.xadd_telemetry(
            down_key, {"eof": "1"}, maxlen=_STREAM_MAXLEN, approximate=True
        )
    except Exception:
        pass


class _UplinkCleanup:
    """Idempotent teardown for a per-request relay uplink.

    Both `body_iter`'s own `finally` and the response's `BackgroundTask` may
    reach this. Teardown runs as a single shared task, and every caller
    awaits it under `asyncio.shield` — so a caller that gets cancelled
    partway through can't abort the in-flight cleanup for whoever else is
    (or later starts) awaiting it. Idempotency keys off the shared task's
    completion, not merely having been entered once, so the relay `eof` is
    guaranteed to be sent exactly once even if the first caller is cancelled
    mid-cleanup.
    """

    def __init__(
        self, pump_task: asyncio.Task[None], redis_client: RedisClient, down_key: str
    ) -> None:
        self._pump_task = pump_task
        self._redis_client = redis_client
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


def _connection_nominated_headers(raw_connection: str) -> set[str]:
    """RFC 7230 lets ``Connection`` name additional per-hop headers to drop,
    e.g. ``Connection: X-Foo`` makes ``X-Foo`` hop-by-hop for this hop only."""
    return {
        token.strip().lower() for token in raw_connection.split(",") if token.strip()
    }


def _header_values(headers: list[tuple[str, str]], name: str) -> list[str]:
    """Collect every value for a (possibly repeated) header name.

    HTTP allows a header field to appear multiple times, equivalent to a
    single comma-joined value; a plain ``dict`` lookup would only see the
    last one.
    """
    lower_name = name.lower()
    return [
        value for header_name, value in headers if header_name.lower() == lower_name
    ]


async def _read_capped_body(request: Request) -> bytes:
    """Read the client's request body, rejecting it before the vLLM api-key
    ever authenticates it if it exceeds ``_MAX_PROXY_BODY_BYTES``.

    The route is intentionally PAT-exempt, so this cap is the only guard
    against an unauthenticated caller forcing an unbounded in-memory buffer.
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
    """Render the client's request as a raw HTTP/1.1 request to the vLLM target.

    ``Connection: close`` is forced so the upstream server ends the TCP
    connection once its response finishes, letting the response reader treat
    the relay's ``eof`` marker as an unambiguous end-of-body signal.
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
    """De-chunk an HTTP/1.1 ``Transfer-Encoding: chunked`` body.

    De-chunking here (rather than passing the wire framing through) lets the
    outbound `StreamingResponse` pick its own framing for the client, instead
    of double-encoding vLLM's chunk markers inside another transfer encoding.
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
    cleanup = _UplinkCleanup(pump_task, redis_client, down)
    response_ready = False

    try:
        request_bytes = await _serialize_request(
            request, upstream_path, target_host, target_port
        )
        await _send_to_relay(redis_client, down, request_bytes)
        status_code, headers = await asyncio.wait_for(
            _read_response_head(reader), timeout=_RESPONSE_HEAD_TIMEOUT_SEC
        )

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

        # `background` covers the response-abandoned-before-first-iteration
        # window (e.g. client disconnect while headers are still being
        # sent), where `body_iter` never starts and its own `finally` never
        # runs. If even that doesn't fire, `_pump_from_relay` still
        # self-terminates once vLLM (given `Connection: close`) finishes and
        # closes its socket, since the worker relay then writes `up`-eof —
        # a bounded, self-healing backstop, not an unbounded leak.
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
