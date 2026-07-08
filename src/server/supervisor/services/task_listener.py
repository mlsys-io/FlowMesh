import asyncio
import logging
from threading import Thread
from typing import Any

from shared.schemas.command import (
    InterruptMessage,
    StopMessage,
    TaskMessage,
)

from ...clients.redis import SyncRedisClient, node_dispatch_channel
from ...utils.helpers import TSQueue
from .pubsub_reader import RebindableReader


class TaskListener(RebindableReader):
    _label = "Task listener"

    def __init__(
        self, redis: SyncRedisClient, node_id: str, logger: logging.Logger
    ) -> None:
        super().__init__(redis, node_id, logger)
        # TODO(kaiitunnz): Consider cleaning up old queues
        self._qs: dict[str, TSQueue[dict[str, Any]]] = {}
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
        if worker_id not in self._qs:
            self._qs[worker_id] = TSQueue()

    def remove_worker(self, worker_id: str) -> None:
        if worker_id in self._qs:
            del self._qs[worker_id]

    async def get_event(self, worker_id: str) -> dict[str, Any]:
        if worker_id not in self._qs:
            raise RuntimeError(f"Worker {worker_id} is not registered")
        return await self._qs[worker_id].get()

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
        if worker_id not in self._qs:
            self.logger.warning(
                "Received dispatch for unregistered worker: %s", worker_id
            )
            return
        asyncio.run_coroutine_threadsafe(self._qs[worker_id].put(payload), loop)
