"""Tests for the SSH proxy WebSocket route's relay-stream cleanup."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.clients.redis import relay_down_key, relay_up_key
from server.routers.v1 import ssh as ssh_router
from shared.schemas.command import CommandResponse

PREFIX = "/api/v1"


class _FakeAsyncRedis:
    """Never delivers anything on the `up` stream, so `redis_to_client`
    keeps looping (mirroring a live, otherwise-idle SSH session) until the
    client disconnect ends `client_to_redis` first and the route tears the
    connection down."""

    def __init__(self) -> None:
        self.down_writes: list[tuple[str, dict]] = []
        self.deleted_keys: list[str] = []

    async def xread_telemetry(
        self, streams: dict, count: int | None = None, block_ms: int | None = None
    ) -> list:
        await asyncio.sleep(0)
        return []

    async def xadd_telemetry(
        self,
        key: str,
        fields: dict,
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        self.down_writes.append((key, fields))
        return "0-1"

    async def delete_telemetry(self, key: str) -> None:
        self.deleted_keys.append(key)


def _make_app() -> tuple[FastAPI, _FakeAsyncRedis, AsyncMock]:
    record = MagicMock()
    record.assigned_worker = "wkr-1"
    record.workflow_id = "wfl-1"
    record.latest_update = {
        "ssh": {
            "_relay_target": {"host": "127.0.0.1", "port": 22},
            "session_id": "sess-1",
        }
    }

    runtime = MagicMock()
    runtime.get_record.return_value = record

    fake_redis = _FakeAsyncRedis()
    redis_client = MagicMock()
    redis_client.asyncio = fake_redis

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
    app.state.logger = logging.getLogger("test.ssh_proxy_router")
    app.state.ssh_proxy_enabled = True
    app.state.node_registry = node_registry
    app.state.worker_registry = worker_registry
    app.state.ssh_audit = None
    app.include_router(ssh_router.router, prefix=PREFIX)
    return app, fake_redis, node_registry.exec_node_cmd


def test_ssh_proxy_deletes_both_relay_streams_on_teardown() -> None:
    """`up`/`down` for this session must both be deleted once the WebSocket
    tears down — the relay uplink can only reliably TTL the `up` stream it
    creates itself, so the consumer (this route) owns prompt cleanup of
    both, mirroring serve.py's `_UplinkCleanup`."""
    app, fake_redis, exec_node_cmd = _make_app()
    client = TestClient(app)

    with client.websocket_connect(f"{PREFIX}/ssh/tasks/tsk-abc/proxy") as websocket:
        websocket.send_bytes(b"hello")
    # Exiting the `with` block sends a client disconnect, which ends
    # `client_to_redis` and drives the route's teardown `finally`.

    relay_token = exec_node_cmd.call_args.args[1].payload["relay_token"]
    up_key = relay_up_key(relay_token)
    down_key = relay_down_key(relay_token)

    assert fake_redis.deleted_keys == [up_key, down_key]


def test_ssh_proxy_writes_client_bytes_to_down_before_teardown() -> None:
    """Sanity check that the fake actually exercised the normal relay path
    (not just the empty-session teardown) before cleanup ran."""
    app, fake_redis, _ = _make_app()
    client = TestClient(app)

    with client.websocket_connect(f"{PREFIX}/ssh/tasks/tsk-abc/proxy") as websocket:
        websocket.send_bytes(b"hello")

    assert any(fields.get("d") for _, fields in fake_redis.down_writes)
