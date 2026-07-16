"""Tests for RelayUplinkService's worker-side TCP<->Redis relay."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.supervisor.services.relay_uplink import (
    RelayUplinkService,
    _down_key,
    _up_key,
)


@pytest.mark.anyio
async def test_connect_failure_signals_eof_to_up_stream() -> None:
    """The server-side reader blocks on the `up` stream waiting for either
    data or an eof marker. A connect failure that writes nothing there would
    otherwise leave that reader hanging until its own timeout or the client
    disconnecting; sending eof here lets it fail fast instead."""
    service = RelayUplinkService(logging.getLogger("test.relay_uplink"))
    service.start()

    rds = MagicMock()

    with patch(
        "server.supervisor.services.relay_uplink.asyncio.open_connection",
        side_effect=OSError("connection refused"),
    ):
        await service._run(rds, "tok-1", "127.0.0.1", 9001, "sess-1")

    up_key = _up_key("tok-1")
    eof_calls = [
        call
        for call in rds.xadd.call_args_list
        if call.args[0] == up_key and call.args[1].get("eof") == "1"
    ]
    assert len(eof_calls) == 1


@pytest.mark.anyio
async def test_connect_failure_does_not_touch_down_stream() -> None:
    """Only the `up` stream (the one the server reads from) needs the eof;
    there's no TCP connection for the `down` stream's writer side to
    matter."""
    service = RelayUplinkService(logging.getLogger("test.relay_uplink"))
    service.start()

    rds = MagicMock()

    with patch(
        "server.supervisor.services.relay_uplink.asyncio.open_connection",
        side_effect=OSError("connection refused"),
    ):
        await service._run(rds, "tok-2", "127.0.0.1", 9001, "sess-2")

    down_key = "relay:tok-2:down"
    down_calls = [call for call in rds.xadd.call_args_list if call.args[0] == down_key]
    assert down_calls == []


@pytest.mark.anyio
async def test_connect_failure_also_expires_both_streams() -> None:
    """The connect-failure path writes eof to `up` and returns early; it
    must still expire BOTH streams (not just skip cleanup on that exit
    path) so a shared consumer that never deletes its own keys — e.g. the
    SSH proxy, which only reads until eof — doesn't leak them. No path may
    leave a stream with no TTL, and neither path should delete outright
    (that reintroduces the lost-eof race)."""
    service = RelayUplinkService(logging.getLogger("test.relay_uplink"))
    service.start()

    rds = MagicMock()

    with patch(
        "server.supervisor.services.relay_uplink.asyncio.open_connection",
        side_effect=OSError("connection refused"),
    ):
        await service._run(rds, "tok-4", "127.0.0.1", 9001, "sess-4")

    up_key = _up_key("tok-4")
    down_key = _down_key("tok-4")
    assert rds.delete.call_count == 0
    expire_calls = {call.args[0]: call.args[1] for call in rds.expire.call_args_list}
    assert expire_calls.get(up_key) == 60
    assert expire_calls.get(down_key) == 60


@pytest.mark.anyio
async def test_normal_completion_expires_streams_instead_of_deleting() -> None:
    """A completed relay must not delete its streams immediately — the
    server-side reader may not have consumed the final eof entry yet (see
    `serve.py`'s "seen, then gone" pump guard) — but should expire them with
    a short TTL as a backstop, since the server's own cleanup deletes them
    promptly in the common path."""
    service = RelayUplinkService(logging.getLogger("test.relay_uplink"))
    service.start()

    rds = MagicMock()
    rds.xread.return_value = None

    reader = MagicMock()
    reader.read = AsyncMock(return_value=b"")

    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.wait_closed = AsyncMock()

    with patch(
        "server.supervisor.services.relay_uplink.asyncio.open_connection",
        AsyncMock(return_value=(reader, writer)),
    ):
        await service._run(rds, "tok-3", "127.0.0.1", 9001, "sess-3")

    up_key = _up_key("tok-3")
    down_key = _down_key("tok-3")
    assert rds.delete.call_count == 0
    expire_calls = {call.args[0]: call.args[1] for call in rds.expire.call_args_list}
    assert expire_calls.get(up_key) == 60
    assert expire_calls.get(down_key) == 60
