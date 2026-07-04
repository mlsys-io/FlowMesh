import asyncio
import logging
from threading import Thread
from typing import Any

from redis.client import PubSub

from shared.schemas.command import (
    InterruptMessage,
    StopMessage,
    TaskMessage,
)

from ...clients.redis import (
    SyncRedisClient,
    iter_pubsub_messages,
    node_dispatch_channel,
)
from ...utils.helpers import TSQueue


class TaskListener:
    def __init__(
        self, redis: SyncRedisClient, node_id: str, logger: logging.Logger
    ) -> None:
        self.logger = logger
        self._redis = redis
        self._node_id = node_id

        # TODO(kaiitunnz): Consider cleaning up old queues
        self._qs: dict[str, TSQueue[dict[str, Any]]] = {}
        self._pubsub: PubSub | None = None
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running: bool = False

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
        self._pubsub.close()
        self._thread.join()
        self._thread = None
        self._pubsub = None
        self._loop = None
        self.logger.info("Task listener stopped")

    def rebind(self, node_id: str) -> None:
        """Move the dispatch subscription to a new node id.

        Subscribes the new node's dispatch channel and drops the old one on the
        live pubsub connection so an already-running listener keeps receiving
        without a restart. Registered worker queues are preserved.
        """
        if node_id == self._node_id:
            return
        old_node_id = self._node_id
        self._node_id = node_id
        pubsub = self._pubsub
        if pubsub is None:
            return
        pubsub.subscribe(node_dispatch_channel(node_id))
        pubsub.unsubscribe(node_dispatch_channel(old_node_id))
        self.logger.info(
            "Task listener rebound from node %s to %s", old_node_id, node_id
        )

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

    def _run(self) -> None:
        pubsub = self._pubsub
        loop = self._loop
        if pubsub is None or loop is None:
            self.logger.error("Task listener not properly initialized")
            return
        try:
            for data in iter_pubsub_messages(pubsub):
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
        except Exception as exc:
            if self._running:
                self.logger.exception("Task listener loop error: %s", exc)
