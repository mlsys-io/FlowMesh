import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from redis import Redis

from shared.utils.encoding import (
    decode_base64_text_to_bytes,
    encode_bytes_to_base64_text,
)

_STREAM_MAXLEN = 1000
_STREAM_CLEANUP_TTL_SEC = 60
_ATTACH_TIMEOUT_SEC = 10.0

SendBytes = Callable[[bytes], Awaitable[None]]
DispatchRelay = Callable[[str, str, str], bool]


def _up_key(relay_token: str) -> str:
    return f"relay:{relay_token}:up"


def _down_key(relay_token: str) -> str:
    return f"relay:{relay_token}:down"


class RelayRefused(Exception):
    """The stream does not match a relay this supervisor is waiting for."""


@dataclass
class _Pending:
    rds: Redis
    worker_id: str
    endpoint_id: str


class RelayUplinkService:
    """Bridges a relay stream a worker opened to Redis ``up``/``down`` streams.

    The supervisor never dials a worker. It asks the worker to open a stream for
    an endpoint the worker published, and bridges whichever stream arrives with
    a token it is expecting.
    """

    def __init__(
        self, logger: logging.Logger, dispatch_relay: DispatchRelay | None = None
    ) -> None:
        self._logger = logger
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dispatch_relay = dispatch_relay
        self._pending: dict[str, _Pending] = {}
        self._timeouts: dict[str, asyncio.Task[None]] = {}

    def start(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "Relay uplink service must be started inside an event loop"
            ) from exc

    def set_dispatch_relay(self, dispatch_relay: DispatchRelay) -> None:
        self._dispatch_relay = dispatch_relay

    def start_uplink(
        self,
        rds: Redis,
        relay_token: str,
        worker_id: str,
        endpoint_id: str,
    ) -> None:
        """Expect a relay stream for ``endpoint_id`` and ask the worker to open it."""
        loop = self._loop
        if loop is None:
            raise RuntimeError("Relay uplink service not started")

        def _schedule() -> None:
            if relay_token in self._pending:
                self._logger.warning(
                    "Uplink already pending for endpoint %s", endpoint_id
                )
                return
            self._pending[relay_token] = _Pending(rds, worker_id, endpoint_id)
            dispatch = self._dispatch_relay
            if dispatch is None or not dispatch(worker_id, relay_token, endpoint_id):
                self._pending.pop(relay_token, None)
                self._logger.warning(
                    "Cannot ask worker %s to relay endpoint %s", worker_id, endpoint_id
                )
                loop.create_task(self._abandon(rds, relay_token))
                return
            self._timeouts[relay_token] = loop.create_task(
                self._expire_pending(relay_token, endpoint_id)
            )

        loop.call_soon_threadsafe(_schedule)

    async def attach(
        self,
        relay_token: str,
        worker_id: str,
        endpoint_id: str,
        recv: AsyncIterator[bytes],
        send: SendBytes,
    ) -> None:
        """Bridge a stream the worker opened, if it matches a pending relay."""
        pending = self._pending.pop(relay_token, None)
        if pending is None:
            raise RelayRefused("No relay is pending for this token")
        if pending.worker_id != worker_id or pending.endpoint_id != endpoint_id:
            # Put nothing back: a mismatched claim burns the token.
            raise RelayRefused("Relay does not match the pending request")
        timeout = self._timeouts.pop(relay_token, None)
        if timeout is not None:
            timeout.cancel()

        self._logger.info("Relay attached: endpoint=%s", endpoint_id)
        try:
            await self._bridge(pending.rds, relay_token, recv, send)
        finally:
            await self._expire_streams(pending.rds, relay_token)
            self._logger.info("Relay ended: endpoint=%s", endpoint_id)

    async def stop(self) -> None:
        for task in list(self._timeouts.values()):
            task.cancel()
        if self._timeouts:
            await asyncio.gather(*self._timeouts.values(), return_exceptions=True)
        self._timeouts.clear()
        # Active bridges eof themselves; a pending relay has no bridge to do it.
        pending = list(self._pending.items())
        self._pending.clear()
        for relay_token, entry in pending:
            await self._abandon(entry.rds, relay_token)
        self._loop = None

    async def _expire_pending(self, relay_token: str, endpoint_id: str) -> None:
        await asyncio.sleep(_ATTACH_TIMEOUT_SEC)
        entry = self._pending.pop(relay_token, None)
        self._timeouts.pop(relay_token, None)
        if entry is None:
            return
        self._logger.warning(
            "No relay stream for endpoint %s within %ss; abandoning it.",
            endpoint_id,
            _ATTACH_TIMEOUT_SEC,
        )
        await self._abandon(entry.rds, relay_token)

    async def _abandon(self, rds: Redis, relay_token: str) -> None:
        """Fail a relay that never got a stream, the way a failed dial did."""
        with contextlib.suppress(Exception):
            await self._send_eof(rds, _up_key(relay_token))
        await self._expire_streams(rds, relay_token)

    async def _expire_streams(self, rds: Redis, relay_token: str) -> None:
        # Use a short TTL so consumers can still read the final eof before cleanup.
        for key in (_up_key(relay_token), _down_key(relay_token)):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(rds.expire, key, _STREAM_CLEANUP_TTL_SEC)

    async def _send_eof(self, rds: Redis, key: str) -> None:
        await asyncio.to_thread(
            rds.xadd,
            key,
            {"eof": "1"},
            maxlen=_STREAM_MAXLEN,
            approximate=True,
        )

    async def _bridge(
        self,
        rds: Redis,
        relay_token: str,
        recv: AsyncIterator[bytes],
        send: SendBytes,
    ) -> None:
        up = _up_key(relay_token)
        down = _down_key(relay_token)

        async def stream_to_redis() -> None:
            try:
                async for data in recv:
                    await asyncio.to_thread(
                        rds.xadd,
                        up,
                        {"d": encode_bytes_to_base64_text(data)},
                        maxlen=_STREAM_MAXLEN,
                        approximate=True,
                    )
            finally:
                # Shielded: this runs while the task is being cancelled, and the
                # consumer waits on this eof rather than on the stream closing.
                with contextlib.suppress(Exception):
                    await asyncio.shield(self._send_eof(rds, up))

        async def redis_to_stream() -> None:
            last_id = "0"
            while True:
                result: Any = await asyncio.to_thread(
                    rds.xread, {down: last_id}, count=10, block=5000
                )
                if not result:
                    continue
                for _, entries in result:
                    for entry_id, fields in entries:
                        last_id = entry_id
                        if b"eof" in fields or "eof" in fields:
                            return
                        raw = fields.get(b"d") or fields.get("d")
                        if raw:
                            await send(decode_base64_text_to_bytes(raw))

        t1 = asyncio.create_task(stream_to_redis())
        t2 = asyncio.create_task(redis_to_stream())
        try:
            await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Cancel *and await* both: without a socket to close, nothing else
            # unblocks them, and stream_to_redis' eof lives in its finally.
            for task in (t1, t2):
                task.cancel()
            for task in (t1, t2):
                with contextlib.suppress(BaseException):
                    await task
