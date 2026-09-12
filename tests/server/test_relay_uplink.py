"""The supervisor half of a relay: what it waits for, and how it gives up.

The supervisor no longer dials a worker. It records a token it expects a stream
for, asks the worker to open one, and bridges whichever stream matches.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from server.supervisor.services.relay_uplink import (
    RelayRefused,
    RelayUplinkService,
    _down_key,
    _up_key,
)

TOKEN = "tok-1"
WORKER = "wkr-1"
ENDPOINT = "ssn-1"


def _service(dispatch_ok: bool = True) -> tuple[RelayUplinkService, MagicMock]:
    dispatch = MagicMock(return_value=dispatch_ok)
    service = RelayUplinkService(
        logger=logging.getLogger("test"), dispatch_relay=dispatch
    )
    service.start()
    return service, dispatch


def _redis() -> MagicMock:
    rds = MagicMock()
    rds.xread.return_value = None
    return rds


async def _empty() -> AsyncIterator[bytes]:
    return
    yield  # pragma: no cover


def _eof_writes(rds: MagicMock, key: str) -> int:
    return sum(
        1
        for call in rds.xadd.call_args_list
        if call.args[0] == key and call.args[1] == {"eof": "1"}
    )


def _expire_ttls(rds: MagicMock) -> dict[str, int]:
    return {call.args[0]: call.args[1] for call in rds.expire.call_args_list}


class TestPendingRelay:
    @pytest.mark.asyncio
    async def test_start_uplink_asks_the_worker_to_open_a_stream(self) -> None:
        service, dispatch = _service()
        service.start_uplink(_redis(), TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        dispatch.assert_called_once_with(WORKER, TOKEN, ENDPOINT)
        await service.stop()

    @pytest.mark.asyncio
    async def test_an_undeliverable_request_fails_the_relay_immediately(self) -> None:
        """A worker that is not connected must not leave the client hanging."""
        service, _ = _service(dispatch_ok=False)
        rds = _redis()
        service.start_uplink(rds, TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0.05)

        assert _eof_writes(rds, _up_key(TOKEN)) == 1
        assert _eof_writes(rds, _down_key(TOKEN)) == 0
        await service.stop()

    @pytest.mark.asyncio
    async def test_stop_fails_relays_that_never_got_a_stream(self) -> None:
        """A pending relay has no bridge whose teardown would send the eof."""
        service, _ = _service()
        rds = _redis()
        service.start_uplink(rds, TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        await service.stop()

        assert _eof_writes(rds, _up_key(TOKEN)) == 1
        assert _expire_ttls(rds) == {_up_key(TOKEN): 60, _down_key(TOKEN): 60}


class TestAttachRefusal:
    @pytest.mark.asyncio
    async def test_unknown_token_is_refused(self) -> None:
        service, _ = _service()
        with pytest.raises(RelayRefused):
            await service.attach(TOKEN, WORKER, ENDPOINT, _empty(), _noop_send)
        await service.stop()

    @pytest.mark.asyncio
    async def test_another_workers_stream_is_refused(self) -> None:
        """A token must not let one worker attach to another's relay."""
        service, _ = _service()
        service.start_uplink(_redis(), TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        with pytest.raises(RelayRefused):
            await service.attach(TOKEN, "wkr-other", ENDPOINT, _empty(), _noop_send)
        await service.stop()

    @pytest.mark.asyncio
    async def test_a_different_endpoint_is_refused(self) -> None:
        """Otherwise the endpoint named on the wire would be decorative."""
        service, _ = _service()
        service.start_uplink(_redis(), TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        with pytest.raises(RelayRefused):
            await service.attach(TOKEN, WORKER, "ssn-other", _empty(), _noop_send)
        await service.stop()

    @pytest.mark.asyncio
    async def test_a_token_attaches_only_once(self) -> None:
        """Two bridges on one token would interleave bytes on the same streams."""
        service, _ = _service()
        service.start_uplink(_redis(), TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        await service.attach(TOKEN, WORKER, ENDPOINT, _empty(), _noop_send)
        with pytest.raises(RelayRefused):
            await service.attach(TOKEN, WORKER, ENDPOINT, _empty(), _noop_send)
        await service.stop()


class TestBridge:
    @pytest.mark.asyncio
    async def test_stream_bytes_are_published_upward(self) -> None:
        service, _ = _service()
        rds = _redis()
        service.start_uplink(rds, TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        async def recv() -> AsyncIterator[bytes]:
            yield b"hello"

        await service.attach(TOKEN, WORKER, ENDPOINT, recv(), _noop_send)

        payloads = [
            call.args[1]
            for call in rds.xadd.call_args_list
            if call.args[0] == _up_key(TOKEN) and "d" in call.args[1]
        ]
        assert payloads == [{"d": "aGVsbG8="}]
        await service.stop()

    @pytest.mark.asyncio
    async def test_the_stream_ending_eofs_the_up_stream(self) -> None:
        service, _ = _service()
        rds = _redis()
        service.start_uplink(rds, TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        await service.attach(TOKEN, WORKER, ENDPOINT, _empty(), _noop_send)

        assert _eof_writes(rds, _up_key(TOKEN)) == 1
        await service.stop()

    @pytest.mark.asyncio
    async def test_a_finished_relay_expires_rather_than_deletes(self) -> None:
        """A consumer must still be able to read the final eof."""
        service, _ = _service()
        rds = _redis()
        service.start_uplink(rds, TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        await service.attach(TOKEN, WORKER, ENDPOINT, _empty(), _noop_send)

        assert _expire_ttls(rds) == {_up_key(TOKEN): 60, _down_key(TOKEN): 60}
        assert rds.delete.call_count == 0
        await service.stop()

    @pytest.mark.asyncio
    async def test_a_cancelled_bridge_still_eofs_the_up_stream(self) -> None:
        """Nothing else unblocks the halves: there is no socket to close."""
        service, _ = _service()
        rds = _redis()
        service.start_uplink(rds, TOKEN, WORKER, ENDPOINT)
        await asyncio.sleep(0)

        async def never() -> AsyncIterator[bytes]:
            await asyncio.Event().wait()
            yield b""  # pragma: no cover

        attached = asyncio.create_task(
            service.attach(TOKEN, WORKER, ENDPOINT, never(), _noop_send)
        )
        await asyncio.sleep(0.05)
        attached.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attached
        await asyncio.sleep(0.05)

        assert _eof_writes(rds, _up_key(TOKEN)) == 1
        await service.stop()


async def _noop_send(data: bytes) -> None:
    return None
