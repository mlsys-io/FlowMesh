"""Tests for the PAT-exempt serve-task HTTP proxy router."""

import asyncio
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, status
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from server.clients.redis import RedisClient
from server.routers.v1 import serve as serve_router
from shared.schemas.command import CommandResponse
from shared.utils.encoding import (
    decode_base64_text_to_bytes,
    encode_bytes_to_base64_text,
)

PREFIX = "/api/v1"


class _FakeAsyncRedis:
    """Stands in for `AsyncRedisClient`, splitting the canned response across
    several stream entries to exercise cross-boundary reads through the
    `asyncio.StreamReader` adapter."""

    def __init__(self, response_bytes: bytes, chunk_size: int = 7) -> None:
        self._response_bytes = response_bytes
        self._chunk_size = chunk_size
        self.sent_to_down: list[bytes] = []
        self.eof_signaled = False
        self.eof_signal_count = 0
        self.deleted_keys: list[str] = []
        self._served = False

    async def xadd_telemetry(
        self,
        key: str,
        fields: dict,
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        if "eof" in fields:
            self.eof_signaled = True
            self.eof_signal_count += 1
            return "0-1"
        self.sent_to_down.append(decode_base64_text_to_bytes(fields["d"]))
        return "0-1"

    async def xread_telemetry(
        self, streams: dict, count: int | None = None, block_ms: int | None = None
    ) -> list:
        if self._served:
            return []
        self._served = True
        entries = []
        data = self._response_bytes
        idx = 0
        seq = 0
        while idx < len(data):
            seq += 1
            piece = data[idx : idx + self._chunk_size]
            idx += self._chunk_size
            entries.append((f"1-{seq}", {"d": encode_bytes_to_base64_text(piece)}))
        seq += 1
        entries.append((f"1-{seq}", {"eof": "1"}))
        up_key = next(iter(streams))
        return [(up_key, entries)]

    async def exists_telemetry(self, key: str) -> bool:
        """The stream is assumed to exist once the uplink is running; fakes
        that need to simulate an uncreated stream override this."""
        return True

    async def delete_telemetry(self, key: str) -> None:
        self.deleted_keys.append(key)


class _HangingAsyncRedis(_FakeAsyncRedis):
    """Never delivers a response, so callers waiting on the response head
    time out instead of completing (simulates a stalled/dead uplink whose
    stream exists but never receives data)."""

    async def xread_telemetry(
        self, streams: dict, count: int | None = None, block_ms: int | None = None
    ) -> list:
        await asyncio.sleep(1)
        return []


class _NeverCreatedAsyncRedis(_FakeAsyncRedis):
    """Simulates an up stream that has never appeared at all: every read
    times out empty and the key never exists. This must NOT be bailed on —
    a merely-delayed uplink, or a slow non-streaming vLLM generation that
    hasn't emitted anything yet, looks identical from the pump's side."""

    async def xread_telemetry(
        self, streams: dict, count: int | None = None, block_ms: int | None = None
    ) -> list:
        # A real block_ms=5000 read always suspends; yield here too so a
        # bounding `asyncio.wait_for` in tests can actually deliver its
        # cancellation instead of racing a tight, never-suspending loop.
        await asyncio.sleep(0)
        return []

    async def exists_telemetry(self, key: str) -> bool:
        return False


class _SeenThenGoneAsyncRedis(_FakeAsyncRedis):
    """The stream exists for the first few checks (seen), then disappears
    without ever delivering an eof entry — simulating the worker's
    post-completion cleanup running before this reader's next read."""

    def __init__(self, exists_for_checks: int = 1) -> None:
        super().__init__(b"")
        self._exists_for_checks = exists_for_checks
        self._checks = 0

    async def xread_telemetry(
        self, streams: dict, count: int | None = None, block_ms: int | None = None
    ) -> list:
        return []

    async def exists_telemetry(self, key: str) -> bool:
        self._checks += 1
        return self._checks <= self._exists_for_checks


class _DelayedStreamCreationAsyncRedis(_FakeAsyncRedis):
    """The stream doesn't exist for the first few reads — simulating a slow
    non-streaming vLLM generation with nothing emitted yet — then appears
    and delivers the response. The pump must wait through that window
    rather than bailing on "doesn't exist yet"."""

    def __init__(
        self, response_bytes: bytes, empty_reads_before_ready: int = 3
    ) -> None:
        super().__init__(response_bytes)
        self._empty_reads_before_ready = empty_reads_before_ready
        self._empty_reads = 0

    async def xread_telemetry(
        self, streams: dict, count: int | None = None, block_ms: int | None = None
    ) -> list:
        if self._empty_reads < self._empty_reads_before_ready:
            self._empty_reads += 1
            return []
        return await super().xread_telemetry(streams, count=count, block_ms=block_ms)

    async def exists_telemetry(self, key: str) -> bool:
        return self._empty_reads >= self._empty_reads_before_ready


class _SlowThenReadyAsyncRedis(_FakeAsyncRedis):
    """Delivers the response only after a short delay, simulating a
    legitimately slow (but healthy) non-streaming vLLM generation."""

    def __init__(self, response_bytes: bytes, delay: float = 0.2) -> None:
        super().__init__(response_bytes)
        self._delay = delay
        self._slept = False

    async def xread_telemetry(
        self, streams: dict, count: int | None = None, block_ms: int | None = None
    ) -> list:
        if not self._slept:
            self._slept = True
            await asyncio.sleep(self._delay)
        return await super().xread_telemetry(streams, count=count, block_ms=block_ms)


class _DelayedEofAsyncRedis(_FakeAsyncRedis):
    """Delays the `eof` xadd, opening a window in which a caller awaiting
    cleanup can be cancelled while that xadd is still in flight."""

    def __init__(self, response_bytes: bytes, delay: float = 0.05) -> None:
        super().__init__(response_bytes)
        self._delay = delay

    async def xadd_telemetry(
        self,
        key: str,
        fields: dict,
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        if "eof" in fields:
            await asyncio.sleep(self._delay)
        return await super().xadd_telemetry(
            key, fields, maxlen=maxlen, approximate=approximate
        )


class _FakeRedisClient:
    def __init__(self, response_bytes: bytes, chunk_size: int = 7) -> None:
        self.asyncio = _FakeAsyncRedis(response_bytes, chunk_size)


def _fixed_length_http_response(
    body: bytes,
    status_line: str = "HTTP/1.1 200 OK",
    content_type: str = "application/json",
) -> bytes:
    head = (
        f"{status_line}\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode("latin-1")
    return head + body


def _chunked_http_response(
    body_parts: list[bytes], status_line: str = "HTTP/1.1 200 OK"
) -> bytes:
    head = (
        f"{status_line}\r\nContent-Type: text/event-stream\r\n"
        "Transfer-Encoding: chunked\r\n\r\n"
    ).encode("latin-1")
    body = b""
    for part in body_parts:
        body += f"{len(part):x}\r\n".encode("ascii") + part + b"\r\n"
    body += b"0\r\n\r\n"
    return head + body


def _make_record(
    task_id: str = "tsk-abc",
    status: str = "DISPATCHED",
    task_type: str | None = "serve",
    assigned_worker: str | None = "wkr-1",
    latest_update: dict | None = None,
) -> MagicMock:
    record = MagicMock()
    record.task_id = task_id
    record.status = status
    record.task_type = task_type
    record.assigned_worker = assigned_worker
    record.latest_update = (
        latest_update
        if latest_update is not None
        else {
            "serve": {
                "mode": "proxy",
                "_relay_target": {"host": "127.0.0.1", "port": 9001},
                "api_key": "vllm-secret-key",
                "model": "Qwen/Qwen3-7B",
            }
        }
    )
    return record


def _make_app(
    record: MagicMock | None,
    response_bytes: bytes,
    proxy_enabled: bool = True,
    redis_client: _FakeRedisClient | None = None,
) -> tuple[FastAPI, _FakeRedisClient]:
    runtime = MagicMock()
    runtime.get_record.return_value = record

    if redis_client is None:
        redis_client = _FakeRedisClient(response_bytes)

    worker_registry = MagicMock()
    worker_registry.get_worker_async = AsyncMock(
        return_value=SimpleNamespace(id="wkr-1", node_id="nod-1")
    )

    node_registry = MagicMock()
    node_registry.exec_node_cmd = AsyncMock(
        return_value=CommandResponse(command_id="cmd-1", success=True)
    )

    app = FastAPI()
    app.state.runtime = runtime
    app.state.redis_client = redis_client
    app.state.logger = logging.getLogger("test.serve_proxy_router")
    app.state.serve_proxy_enabled = proxy_enabled
    app.state.node_registry = node_registry
    app.state.worker_registry = worker_registry
    app.include_router(serve_router.router, prefix=PREFIX)
    return app, redis_client


@pytest.mark.anyio
async def test_pat_exempt_no_auth_required() -> None:
    """The route has no `authenticate_connection`/`require_permission` gate:
    a request carrying no Lumid PAT at all still succeeds."""
    body = b'{"choices": []}'
    app, _ = _make_app(_make_record(), _fixed_length_http_response(body))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 200
    assert resp.content == body


@pytest.mark.anyio
async def test_authorization_header_forwarded_untouched() -> None:
    """The client's vLLM api-key `Authorization` header reaches the upstream
    request byte-for-byte."""
    app, redis_client = _make_app(_make_record(), _fixed_length_http_response(b"{}"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(
            f"{PREFIX}/serve/tasks/tsk-abc/v1/models",
            headers={"Authorization": "Bearer vllm-secret-key"},
        )

    assert resp.status_code == 200
    sent = b"".join(redis_client.asyncio.sent_to_down)
    assert b"Bearer vllm-secret-key\r\n" in sent
    assert b"authorization" in sent.lower()


@pytest.mark.anyio
async def test_streaming_response_passthrough() -> None:
    """Chunked (SSE-style) upstream bodies are de-chunked and streamed through
    without buffering the whole response."""
    parts = [
        b'data: {"delta": "wor"}\n\n',
        b'data: {"delta": "ld"}\n\n',
        b"data: [DONE]\n\n",
    ]
    app, _ = _make_app(_make_record(), _chunked_http_response(parts))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            f"{PREFIX}/serve/tasks/tsk-abc/v1/chat/completions",
            json={"stream": True},
        )

    assert resp.status_code == 200
    assert resp.content == b"".join(parts)


@pytest.mark.anyio
async def test_unknown_task_rejected() -> None:
    app, _ = _make_app(None, b"")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-missing/v1/models")

    assert resp.status_code == 404


@pytest.mark.anyio
async def test_non_serve_task_rejected() -> None:
    """SSRF guard: a task_id that resolves to a non-serve task must not be
    proxyable, regardless of its own relay target."""
    record = _make_record(task_type="inference")
    app, _ = _make_app(record, b"")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 404


@pytest.mark.anyio
async def test_not_running_task_rejected() -> None:
    record = _make_record(status="DONE")
    app, _ = _make_app(record, b"")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 409


@pytest.mark.anyio
async def test_non_proxy_mode_rejected() -> None:
    """A serve task running in forward/direct mode has no relay target to
    proxy through; the route must reject it rather than guess."""
    record = _make_record(
        latest_update={
            "serve": {
                "mode": "forward",
                "host": "server.example.com",
                "port": 32001,
            }
        }
    )
    app, _ = _make_app(record, b"")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 409


@pytest.mark.anyio
async def test_missing_relay_target_rejected() -> None:
    record = _make_record(latest_update={"serve": {"mode": "proxy"}})
    app, _ = _make_app(record, b"")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 409


@pytest.mark.anyio
async def test_no_endpoint_info_yet_rejected_with_clear_message() -> None:
    """Before the executor's first TASK_UPDATE, `latest_update` has no
    `serve` key at all; that must produce a clear, distinct 409 rather than
    being conflated with "wrong access mode"."""
    record = _make_record(latest_update={})
    app, _ = _make_app(record, b"")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 409
    assert resp.json()["detail"] == "serve task has no endpoint info yet"


@pytest.mark.anyio
async def test_proxy_disabled_returns_403() -> None:
    app, _ = _make_app(_make_record(), b"", proxy_enabled=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 403


@pytest.mark.anyio
async def test_oversized_body_returns_413(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared Content-Length over the cap is rejected before the
    unauthenticated body would otherwise be buffered in full."""
    monkeypatch.setattr(serve_router, "_MAX_PROXY_BODY_BYTES", 10)
    app, _ = _make_app(_make_record(), b"")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            f"{PREFIX}/serve/tasks/tsk-abc/v1/chat/completions",
            content=b"x" * 100,
        )

    assert resp.status_code == status.HTTP_413_CONTENT_TOO_LARGE


@pytest.mark.anyio
async def test_declared_content_length_checked_before_reading_body() -> None:
    """The Content-Length cap check must run before any body bytes are read
    off the wire, since the route authenticates nothing before that point."""

    async def _boom() -> dict:
        raise AssertionError(
            "body must not be read once Content-Length exceeds the cap"
        )

    scope = {
        "type": "http",
        "method": "POST",
        "headers": [
            (
                b"content-length",
                str(serve_router._MAX_PROXY_BODY_BYTES + 1).encode(),
            )
        ],
    }
    request = Request(scope, _boom)

    with pytest.raises(HTTPException) as exc_info:
        await serve_router._read_capped_body(request)

    assert exc_info.value.status_code == status.HTTP_413_CONTENT_TOO_LARGE


@pytest.mark.anyio
async def test_streamed_body_without_content_length_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body streamed without a Content-Length header (e.g. chunked request)
    is still capped incrementally, not just via the declared-length check."""
    monkeypatch.setattr(serve_router, "_MAX_PROXY_BODY_BYTES", 8)
    chunks = [b"12345", b"67890"]

    async def _receive() -> dict:
        if chunks:
            chunk = chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(chunks)}
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {"type": "http", "method": "POST", "headers": []}
    request = Request(scope, _receive)

    with pytest.raises(HTTPException) as exc_info:
        await serve_router._read_capped_body(request)

    assert exc_info.value.status_code == status.HTTP_413_CONTENT_TOO_LARGE


@pytest.mark.anyio
async def test_eof_signaled_to_relay_after_response_completes() -> None:
    """Once the proxied response finishes, the router must eof the `down`
    stream so the worker-side relay's `redis_to_tcp` loop stops looping and
    the upstream TCP connection is closed instead of leaking."""
    app, redis_client = _make_app(_make_record(), _fixed_length_http_response(b"{}"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 200
    assert redis_client.asyncio.eof_signaled is True


def test_no_response_head_timeout_constant() -> None:
    """A non-streaming vLLM generation can legitimately take longer than any
    fixed ceiling; there must be no wall-clock timeout wrapping the
    response-head read (removed in favor of uplink eof-on-failure + client
    disconnect handling)."""
    assert not hasattr(serve_router, "_RESPONSE_HEAD_TIMEOUT_SEC")


@pytest.mark.anyio
async def test_slow_non_streaming_generation_is_not_cut_off() -> None:
    """A response slower than the old fixed head-timeout would have allowed
    must still succeed now that the timeout has been removed."""
    body = b'{"choices": []}'
    redis_client = _FakeRedisClient(b"")
    redis_client.asyncio = _SlowThenReadyAsyncRedis(
        _fixed_length_http_response(body), delay=0.2
    )

    app, _ = _make_app(_make_record(), b"", redis_client=redis_client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 200
    assert resp.content == body


@pytest.mark.anyio
async def test_slow_generation_with_delayed_stream_creation_succeeds() -> None:
    """End-to-end: a non-streaming generation whose up stream doesn't exist
    yet (nothing emitted so far, mirroring real worker behavior since it
    only creates the stream on its first write) must complete successfully
    once the stream appears, not be killed early by the exists-check."""
    body = b'{"choices": []}'
    response = _fixed_length_http_response(body)
    redis_client = _FakeRedisClient(response)
    redis_client.asyncio = _DelayedStreamCreationAsyncRedis(
        response, empty_reads_before_ready=2
    )

    app, _ = _make_app(_make_record(), b"", redis_client=redis_client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 200
    assert resp.content == body


@pytest.mark.anyio
async def test_dead_relay_still_fails_fast_without_head_timeout() -> None:
    """Even with no wall-clock head timeout, a relay that immediately eofs
    with no data (e.g. the worker's connect-failure eof) still produces a
    fast failure rather than hanging."""
    app, redis_client = _make_app(_make_record(), b"")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 502
    assert redis_client.asyncio.eof_signaled is True


@pytest.mark.anyio
async def test_pump_keeps_waiting_when_stream_never_yet_seen() -> None:
    """A stream that has never appeared at all must NOT be bailed on — that
    would reintroduce a first-byte timeout and kill healthy slow requests
    (a delayed uplink and a slow non-streaming generation both look like
    this from the pump's side). The fake never creates the stream, so the
    pump must still be running when this gives up waiting on it."""
    redis_client = _FakeRedisClient(b"")
    redis_client.asyncio = _NeverCreatedAsyncRedis(b"")
    reader = asyncio.StreamReader()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            serve_router._pump_from_relay(
                cast(RedisClient, redis_client), "up-key", reader
            ),
            timeout=0.1,
        )

    assert not reader.at_eof()


@pytest.mark.anyio
async def test_pump_terminates_when_stream_seen_then_gone() -> None:
    """Once the stream has been observed to exist, its later disappearance
    (without ever delivering an eof entry) means the worker already
    finished and its own cleanup ran before this reader's next read —
    that transition, unlike "never seen", is treated as eof."""
    redis_client = _FakeRedisClient(b"")
    redis_client.asyncio = _SeenThenGoneAsyncRedis(exists_for_checks=1)
    reader = asyncio.StreamReader()

    await asyncio.wait_for(
        serve_router._pump_from_relay(
            cast(RedisClient, redis_client), "up-key", reader
        ),
        timeout=1.0,
    )

    assert reader.at_eof()


@pytest.mark.anyio
async def test_pump_waits_through_not_yet_seen_window_then_delivers_data() -> None:
    """A stream that doesn't exist yet (still-forming, e.g. no bytes emitted
    by a slow non-streaming generation) must not be bailed on; the pump
    keeps waiting until it actually appears and then delivers normally."""
    response = _fixed_length_http_response(b"{}")
    redis_client = _FakeRedisClient(response)
    redis_client.asyncio = _DelayedStreamCreationAsyncRedis(
        response, empty_reads_before_ready=3
    )
    reader = asyncio.StreamReader()

    await asyncio.wait_for(
        serve_router._pump_from_relay(
            cast(RedisClient, redis_client), "up-key", reader
        ),
        timeout=1.0,
    )

    buffered = await reader.read(-1)
    assert buffered == response
    assert reader.at_eof()


def _bare_request(
    method: str = "GET", headers: list[tuple[bytes, bytes]] | None = None
) -> Request:
    scope = {
        "type": "http",
        "method": method,
        "headers": headers or [],
        "path": "/api/v1/serve/tasks/tsk-abc/v1/models",
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "root_path": "",
    }

    async def _receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, _receive)


@pytest.mark.anyio
async def test_client_disconnect_before_response_head_still_signals_eof() -> None:
    """`asyncio.CancelledError` (a `BaseException`, not `Exception`) raised
    while awaiting the response head — e.g. a client disconnect mid-request —
    must still tear down the worker-side relay uplink instead of leaking it."""
    record = _make_record()
    runtime = MagicMock()
    runtime.get_record.return_value = record

    redis_client = _FakeRedisClient(b"")
    redis_client.asyncio = _HangingAsyncRedis(b"")

    worker_registry = MagicMock()
    worker_registry.get_worker_async = AsyncMock(
        return_value=SimpleNamespace(id="wkr-1", node_id="nod-1")
    )
    node_registry = MagicMock()
    node_registry.exec_node_cmd = AsyncMock(
        return_value=CommandResponse(command_id="cmd-1", success=True)
    )

    task = asyncio.ensure_future(
        serve_router.serve_proxy(
            _bare_request(),
            task_id="tsk-abc",
            upstream_path="v1/models",
            runtime=runtime,
            redis_client=cast(RedisClient, redis_client),
            logger=logging.getLogger("test.serve_proxy_router.cancel"),
            proxy_enabled=True,
            node_registry=node_registry,
            worker_registry=worker_registry,
        )
    )
    await asyncio.sleep(0)  # let the handler reach the hanging response-head wait
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert redis_client.asyncio.eof_signaled is True


@pytest.mark.anyio
async def test_malformed_upstream_content_length_falls_back_and_does_not_leak() -> None:
    """A malformed upstream Content-Length must not crash the handler (which
    would skip cleanup); it degrades to reading until EOF instead."""
    body = b'{"choices": []}'
    malformed = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b"Content-Length: not-a-number\r\n\r\n" + body
    )
    app, redis_client = _make_app(_make_record(), malformed)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 200
    assert resp.content == body
    assert redis_client.asyncio.eof_signaled is True


@pytest.mark.parametrize("bad_value", ["-1", "1_0", "+1", "abc", ""])
@pytest.mark.anyio
async def test_declared_content_length_strictly_validated(bad_value: str) -> None:
    """`int()` alone would accept `-1`, `+1`, and `1_0`; only a plain
    non-negative digit string is a valid declared Content-Length."""
    request = _bare_request(
        method="POST", headers=[(b"content-length", bad_value.encode())]
    )

    with pytest.raises(HTTPException) as exc_info:
        await serve_router._read_capped_body(request)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.anyio
async def test_uplink_cleanup_idempotent_when_invoked_twice() -> None:
    """Calling `_UplinkCleanup.run()` more than once must not double-cancel
    the pump task or send a second `eof` to the relay."""
    redis_client = _FakeRedisClient(b"")

    async def _never_returns() -> None:
        await asyncio.sleep(3600)

    pump_task = asyncio.ensure_future(_never_returns())
    cleanup = serve_router._UplinkCleanup(
        pump_task, cast(RedisClient, redis_client), "up-key", "down-key"
    )

    await cleanup.run()
    await cleanup.run()

    assert redis_client.asyncio.eof_signal_count == 1
    assert pump_task.cancelled()


@pytest.mark.anyio
async def test_cleanup_deletes_both_relay_streams() -> None:
    """After cleanup, both the up and down relay keys must be removed so a
    torn-down request doesn't leave its Redis keys behind."""
    redis_client = _FakeRedisClient(b"")

    async def _never_returns() -> None:
        await asyncio.sleep(3600)

    pump_task = asyncio.ensure_future(_never_returns())
    cleanup = serve_router._UplinkCleanup(
        pump_task, cast(RedisClient, redis_client), "up-key", "down-key"
    )

    await cleanup.run()

    assert redis_client.asyncio.deleted_keys == ["up-key", "down-key"]


@pytest.mark.anyio
async def test_cancelled_first_caller_does_not_block_eof_for_second_caller() -> None:
    """A caller cancelled partway through `run()` (e.g. `body_iter`'s
    `finally` unwinding under its own cancellation) must not prevent a later
    caller (the response's `BackgroundTask`) from still completing the relay
    `eof`. Idempotency must key off the shared cleanup task's completion,
    not merely having been entered once."""
    redis_client = _FakeRedisClient(b"")
    redis_client.asyncio = _DelayedEofAsyncRedis(b"", delay=0.05)

    async def _never_returns() -> None:
        await asyncio.sleep(3600)

    pump_task = asyncio.ensure_future(_never_returns())
    cleanup = serve_router._UplinkCleanup(
        pump_task, cast(RedisClient, redis_client), "up-key", "down-key"
    )

    first_caller = asyncio.ensure_future(cleanup.run())
    await asyncio.sleep(0.01)  # let it start the shared task and reach the delayed xadd
    first_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_caller

    # A second caller (e.g. the BackgroundTask) must still observe the
    # shared cleanup task complete successfully, eof included.
    await cleanup.run()

    assert redis_client.asyncio.eof_signal_count == 1
    assert pump_task.cancelled()


@pytest.mark.anyio
async def test_full_response_cleans_up_uplink_exactly_once() -> None:
    """Both `body_iter`'s `finally` and the response's `BackgroundTask` are
    reachable on a normal full response; idempotency must collapse that to a
    single cancel + eof, not two."""
    app, redis_client = _make_app(_make_record(), _fixed_length_http_response(b"{}"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.get(f"{PREFIX}/serve/tasks/tsk-abc/v1/models")

    assert resp.status_code == 200
    assert redis_client.asyncio.eof_signal_count == 1


@pytest.mark.anyio
async def test_background_task_alone_cleans_up_abandoned_response() -> None:
    """Simulates a response abandoned before `body_iter` is ever advanced
    (e.g. a client disconnect while Starlette is still sending headers): the
    response's `background` task, run on its own with the body iterator
    never touched, must still tear down the uplink exactly once."""
    record = _make_record()
    runtime = MagicMock()
    runtime.get_record.return_value = record

    redis_client = _FakeRedisClient(_fixed_length_http_response(b"{}"))

    worker_registry = MagicMock()
    worker_registry.get_worker_async = AsyncMock(
        return_value=SimpleNamespace(id="wkr-1", node_id="nod-1")
    )
    node_registry = MagicMock()
    node_registry.exec_node_cmd = AsyncMock(
        return_value=CommandResponse(command_id="cmd-1", success=True)
    )

    response = await serve_router.serve_proxy(
        _bare_request(),
        task_id="tsk-abc",
        upstream_path="v1/models",
        runtime=runtime,
        redis_client=cast(RedisClient, redis_client),
        logger=logging.getLogger("test.serve_proxy_router.abandoned"),
        proxy_enabled=True,
        node_registry=node_registry,
        worker_registry=worker_registry,
    )

    assert response.background is not None
    # `body_iterator` is intentionally never iterated here, matching a
    # response abandoned before its first `body_iter` advance.
    await response.background()

    assert redis_client.asyncio.eof_signal_count == 1
