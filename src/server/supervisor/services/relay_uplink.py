import asyncio
import logging
from typing import Any

from redis import Redis

from shared.utils.encoding import (
    decode_base64_text_to_bytes,
    encode_bytes_to_base64_text,
)

_READ_CHUNK = 16384
_STREAM_MAXLEN = 1000


def _up_key(relay_token: str) -> str:
    return f"relay:{relay_token}:up"


def _down_key(relay_token: str) -> str:
    return f"relay:{relay_token}:down"


class RelayUplinkService:
    """Manages relay uplinks from a worker-internal TCP endpoint to Redis Streams."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def start(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "Relay uplink service must be started inside an event loop"
            ) from exc

    def start_uplink(
        self,
        rds: Redis,
        relay_token: str,
        target_host: str,
        target_port: int,
        session_id: str,
    ) -> None:
        loop = self._loop
        if loop is None:
            raise RuntimeError("Relay uplink service not started")

        def _schedule() -> None:
            if relay_token in self._tasks:
                self._logger.warning("Uplink already active for session %s", session_id)
                return
            task = loop.create_task(
                self._run(rds, relay_token, target_host, target_port, session_id)
            )
            self._tasks[relay_token] = task
            task.add_done_callback(lambda _: self._tasks.pop(relay_token, None))

        loop.call_soon_threadsafe(_schedule)

    async def stop(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._loop = None

    async def _run(
        self,
        rds: Redis,
        relay_token: str,
        target_host: str,
        target_port: int,
        session_id: str,
    ) -> None:
        up = _up_key(relay_token)
        down = _down_key(relay_token)

        try:
            reader, writer = await asyncio.open_connection(target_host, target_port)
        except OSError as exc:
            self._logger.warning(
                "Relay uplink: cannot connect to %s:%s: %s",
                target_host,
                target_port,
                exc,
            )
            return

        self._logger.info("Relay uplink started: session=%s", session_id)
        try:

            async def tcp_to_redis() -> None:
                try:
                    while True:
                        data = await reader.read(_READ_CHUNK)
                        if not data:
                            break
                        await asyncio.to_thread(
                            rds.xadd,
                            up,
                            {"d": encode_bytes_to_base64_text(data)},
                            maxlen=_STREAM_MAXLEN,
                            approximate=True,
                        )
                finally:
                    await asyncio.to_thread(
                        rds.xadd,
                        up,
                        {"eof": "1"},
                        maxlen=_STREAM_MAXLEN,
                        approximate=True,
                    )

            async def redis_to_tcp() -> None:
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
                                writer.write(decode_base64_text_to_bytes(raw))
                                await writer.drain()

            t1 = asyncio.create_task(tcp_to_redis())
            t2 = asyncio.create_task(redis_to_tcp())
            _, pending = await asyncio.wait(
                [t1, t2], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.warning("Relay uplink error: session=%s: %s", session_id, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            # Cleanup Redis streams
            try:
                await asyncio.to_thread(rds.delete, up, down)
            except Exception:
                pass
            self._logger.info("Relay uplink ended: session=%s", session_id)
