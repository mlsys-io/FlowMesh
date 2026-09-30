import asyncio
import logging
from threading import Thread
from typing import Any, Final

from shared.schemas.command import (
    InterruptMessage,
    StopMessage,
    TaskMessage,
)

from ...clients.redis import SyncRedisClient, node_dispatch_channel
from .pubsub_reader import RebindableReader

_CLOSED: Final = None

DispatchQueue = asyncio.Queue[dict[str, Any] | None]


class DispatchStream:
    """One ``StreamTasks`` call's view of its worker's dispatch queue."""

    def __init__(self, worker_id: str, queue: DispatchQueue) -> None:
        self.worker_id = worker_id
        self._queue = queue

    async def next(self) -> dict[str, Any] | None:
        """Return the next dispatch payload, or ``None`` once the queue is closed."""
        event = await self._queue.get()
        if event is _CLOSED:
            # Keep the sentinel so every later call also returns None.
            self._queue.put_nowait(_CLOSED)
        return event


class TaskListener(RebindableReader):
    """Routes dispatch frames from the node's Redis channel to per-worker queues.

    Each registered worker id has one queue, read by the newest ``StreamTasks``
    attached to it. The queues live on the supervisor loop: every mutation and
    every delivery runs there, and other threads only schedule callbacks onto it.
    """

    _label = "Task listener"

    def __init__(
        self, redis: SyncRedisClient, node_id: str, logger: logging.Logger
    ) -> None:
        super().__init__(redis, node_id, logger)
        self._qs: dict[str, DispatchQueue] = {}
        self._attached: dict[str, DispatchStream] = {}
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _channel(self, node_id: str) -> str:
        return node_dispatch_channel(node_id)

    def start(self) -> None:
        if self._thread is not None:
            self.logger.warning("Task listener already started")
            return
        if self._pubsub is not None:
            self.logger.warning("Task listener pubsub already initialized")
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            self.logger.error(
                "Task listener must be started inside an event loop: %s", exc
            )
            return
        assert not self._running
        self._running = True
        self._subscribe()
        self._thread = Thread(
            target=self._read_loop,
            name="TaskListenerThread",
            daemon=True,
        )
        self._thread.start()
        self.logger.info("Task listener started")

    def stop(self) -> None:
        if self._thread is None or self._pubsub is None:
            self.logger.warning("Task listener not started")
            return
        assert self._running
        self._running = False
        self._thread.join()
        self._pubsub.close()
        self._thread = None
        self._pubsub = None
        self._loop = None
        self.logger.info("Task listener stopped")

    def add_worker(self, worker_id: str) -> None:
        """Create the dispatch queue for a newly registered worker id (loop only)."""
        if worker_id not in self._qs:
            self._qs[worker_id] = asyncio.Queue()

    def attach_stream(self, worker_id: str) -> DispatchStream | None:
        """Make a new stream the only reader of a worker's dispatch queue (loop only).

        The new stream takes over the pending frames in order, and every earlier
        stream on the id is closed. Returns ``None`` for an unknown worker id.
        """
        old = self._qs.get(worker_id)
        if old is None:
            return None
        new: DispatchQueue = asyncio.Queue()
        while not old.empty():
            new.put_nowait(old.get_nowait())
        old.put_nowait(_CLOSED)
        self._qs[worker_id] = new
        stream = DispatchStream(worker_id, new)
        if self._attached.get(worker_id) is not None:
            self.logger.info("Superseding task stream for worker %s", worker_id)
        self._attached[worker_id] = stream
        return stream

    def detach_stream(self, stream: DispatchStream) -> None:
        """Record that a stream stopped reading (loop only)."""
        if self._attached.get(stream.worker_id) is stream:
            del self._attached[stream.worker_id]

    def remove_worker(self, worker_id: str) -> None:
        """Detach and close a released worker id's queue. Callable from any thread."""
        loop = self._loop
        if loop is None:
            self._qs.pop(worker_id, None)
            self._attached.pop(worker_id, None)
            return
        loop.call_soon_threadsafe(self._close_queue, worker_id)

    def _close_queue(self, worker_id: str) -> None:
        self._attached.pop(worker_id, None)
        q = self._qs.pop(worker_id, None)
        if q is None:
            return
        dropped = 0
        while not q.empty():
            q.get_nowait()
            dropped += 1
        if dropped:
            self.logger.warning(
                "Dropping %d queued dispatch(es) for released worker: %s",
                dropped,
                worker_id,
            )
        q.put_nowait(_CLOSED)

    def _deliver(self, worker_id: str, payload: dict[str, Any]) -> None:
        q = self._qs.get(worker_id)
        if q is None:
            self.logger.warning(
                "Dropping dispatch for unregistered worker: %s", worker_id
            )
            return
        q.put_nowait(payload)

    def dispatch_relay(
        self, worker_id: str, relay_token: str, endpoint_id: str
    ) -> bool:
        """Queue a relay request for a worker's dispatch stream.

        Callable from any thread; returns whether the worker is connected.
        """
        loop = self._loop
        if loop is None:
            self.logger.warning("Task listener not started; dropping relay request")
            return False
        if worker_id not in self._qs:
            self.logger.warning("Cannot relay for unregistered worker: %s", worker_id)
            return False
        payload = {
            "kind": "relay",
            "relay_token": relay_token,
            "endpoint_id": endpoint_id,
        }
        loop.call_soon_threadsafe(self._deliver, worker_id, payload)
        return True

    def _handle_message(self, data: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        if "kind" not in data:
            self.logger.warning("Received dispatch message without kind: %s", data)
            return
        match data["kind"]:
            case "task":
                task_message = TaskMessage.model_validate(data)
                worker_id = task_message.worker_id
                payload = task_message.payload
            case "interrupt":
                interrupt_message = InterruptMessage.model_validate(data)
                worker_id = interrupt_message.worker_id
                payload = {
                    "kind": "interrupt",
                    "task_id": interrupt_message.task_id,
                    "reason": interrupt_message.reason,
                }
            case "stop":
                stop_message = StopMessage.model_validate(data)
                worker_id = stop_message.worker_id
                payload = {
                    "kind": "stop",
                    "task_id": stop_message.task_id,
                    "reason": stop_message.reason,
                }
            case _:
                self.logger.warning(
                    "Received dispatch message with unknown kind: %s", data
                )
                return
        loop.call_soon_threadsafe(self._deliver, worker_id, payload)
