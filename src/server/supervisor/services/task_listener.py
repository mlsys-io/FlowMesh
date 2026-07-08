import asyncio
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

from redis.client import PubSub

from shared.schemas.command import (
    InterruptMessage,
    StopMessage,
    TaskMessage,
)

from ...clients.redis import (
    SyncRedisClient,
    node_dispatch_channel,
    parse_pubsub_message,
)
from ...utils.helpers import TSQueue

_POLL_TIMEOUT_SEC = 0.25
_RECONNECT_BACKOFF_SEC = 1.0


class TaskListener:
    def __init__(
        self, redis: SyncRedisClient, node_id: str, logger: logging.Logger
    ) -> None:
        self.logger = logger
        self._redis = redis
        # node_id is mutated only by the pubsub reader thread once running.
        self._node_id = node_id

        # TODO(kaiitunnz): Consider cleaning up old queues
        self._qs: dict[str, TSQueue[dict[str, Any]]] = {}
        self._pubsub: PubSub | None = None
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running: bool = False

        self._rebind_lock = Lock()
        self._pending_node_id: str | None = None
        self._rebind_applied = Event()

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
        self._pubsub = self._redis.subscribe_control(
            node_dispatch_channel(self._node_id)
        )
        self._thread = Thread(
            target=self._run,
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

    def rebind(self, node_id: str) -> None:
        """Request moving the dispatch subscription to a new node id.

        Records the target under a lock; the reader thread applies the actual
        ``subscribe``/``unsubscribe`` between polls (mutating redis-py ``PubSub`` is not
        thread-safe). ``wait_rebound`` blocks until the switch has taken effect.
        Registered worker queues are preserved across the rebind.
        """
        if node_id == self._node_id:
            self._rebind_applied.set()
            return
        self._rebind_applied.clear()
        with self._rebind_lock:
            self._pending_node_id = node_id

    def wait_rebound(self, timeout: float) -> bool:
        return self._rebind_applied.wait(timeout)

    def _apply_pending_rebind(self, pubsub: PubSub, current_id: str) -> str:
        with self._rebind_lock:
            pending = self._pending_node_id
            self._pending_node_id = None
        if pending is None or pending == current_id:
            return current_id
        try:
            pubsub.subscribe(node_dispatch_channel(pending))
            pubsub.unsubscribe(node_dispatch_channel(current_id))
        except (ConnectionError, OSError):
            # The connection dropped mid-switch; re-arm the target (unless a newer
            # rebind already superseded it) so the reader re-applies it after
            # reconnecting, rather than silently dropping the move.
            with self._rebind_lock:
                if self._pending_node_id is None:
                    self._pending_node_id = pending
            raise
        self._node_id = pending
        self._rebind_applied.set()
        self.logger.info(
            "Task listener rebound from node %s to %s", current_id, pending
        )
        return pending

    def add_worker(self, worker_id: str) -> None:
        if worker_id not in self._qs:
            self._qs[worker_id] = TSQueue()

    def remove_worker(self, worker_id: str) -> None:
        if worker_id in self._qs:
            del self._qs[worker_id]

    async def get_event(self, worker_id: str) -> dict[str, Any]:
        if worker_id not in self._qs:
            raise RuntimeError(f"Worker {worker_id} is not registered")
        return await self._qs[worker_id].get()

    def _resubscribe(self, current_id: str) -> PubSub | None:
        """Re-establish the dispatch subscription after a dropped connection,
        retrying with backoff until it succeeds or the listener is stopped.

        A dead reader is never restarted elsewhere, so recovering the connection
        here is what guarantees a pending rebind eventually applies.
        """
        if self._pubsub is not None:
            try:
                self._pubsub.close()
            except (ConnectionError, OSError):
                pass
        while self._running:
            # Backoff before every attempt so a connection that accepts SUBSCRIBE
            # but drops on the next read can't drive a tight reconnect loop.
            time.sleep(_RECONNECT_BACKOFF_SEC)
            try:
                pubsub = self._redis.subscribe_control(
                    node_dispatch_channel(current_id)
                )
            except (ConnectionError, OSError) as exc:
                self.logger.warning(
                    "Task listener resubscribe failed (%s); retrying", exc
                )
                continue
            self._pubsub = pubsub
            self.logger.info("Task listener reconnected on node %s", current_id)
            return pubsub
        return None

    def _run(self) -> None:
        pubsub = self._pubsub
        loop = self._loop
        if pubsub is None or loop is None:
            self.logger.error("Task listener not properly initialized")
            return
        current_id = self._node_id
        while self._running:
            try:
                current_id = self._apply_pending_rebind(pubsub, current_id)
                msg = pubsub.get_message(timeout=_POLL_TIMEOUT_SEC)
                data = parse_pubsub_message(msg)
                if data is None:
                    continue
                if "kind" not in data:
                    self.logger.warning(
                        "Received dispatch message without kind: %s", data
                    )
                    continue
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
                        continue
                if worker_id not in self._qs:
                    self.logger.warning(
                        "Received dispatch for unregistered worker: %s", worker_id
                    )
                    continue
                queue = self._qs[worker_id]
                asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
            except (ConnectionError, OSError) as exc:
                if not self._running:
                    break
                self.logger.warning(
                    "Task listener connection lost (%s); reconnecting", exc
                )
                reconnected = self._resubscribe(current_id)
                if reconnected is None:
                    break
                pubsub = reconnected
            except Exception as exc:
                if self._running:
                    self.logger.exception("Task listener loop error: %s", exc)
                break
