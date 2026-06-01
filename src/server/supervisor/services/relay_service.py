import json
import logging
from queue import Queue
from threading import Thread
from typing import Any

from shared.schemas.event import TaskEvent, WorkerEvent, parse_event, serialize_event

from ...clients.redis import (
    TASK_EVENT_STREAM_KEY,
    TASK_EVENT_STREAM_MAXLEN,
    WORKER_EVENT_CHANNEL,
    SyncRedisClient,
    task_log_stream_key,
    workflow_log_stream_key,
)


class _LogData:
    __slots__ = ("data",)

    def __init__(self, data: dict[str, Any]):
        self.data = data


class RelayService:
    _KILL_SIGNAL = object()

    def __init__(self, redis: SyncRedisClient, logger: logging.Logger):
        self.logger = logger

        self._redis = redis
        self._q = Queue[Any]()
        self._thread: Thread | None = None

    def start(self):
        if self._thread is not None:
            self.logger.warning("Relay service already started")
            return
        self._thread = Thread(
            target=self._handle_event_loop,
            name="RelayServiceThread",
            daemon=True,
        )
        self._thread.start()
        self.logger.info("Relay service started")

    def stop(self):
        if self._thread is None:
            self.logger.warning("Relay service not started")
            return
        self._q.put(self._KILL_SIGNAL)
        self._thread.join()
        self._thread = None
        self.logger.info("Relay service stopped")

    def add_event(self, event_data: Any) -> None:
        self._q.put(event_data)

    def add_log(self, log_data: Any) -> None:
        self._q.put(_LogData(log_data))

    def _handle_event_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is self._KILL_SIGNAL:
                break
            if isinstance(item, _LogData):
                log_data = item.data
                if not isinstance(log_data, dict):
                    self.logger.warning("Invalid log item: %s", log_data)
                    continue
                try:
                    task_refs = log_data.get("task_refs") or [
                        {
                            "task_id": log_data.get("task_id", ""),
                            "workflow_id": log_data.get("workflow_id", ""),
                        }
                    ]
                    encoded = json.dumps(log_data, ensure_ascii=False)
                    seen_task_ids: set[str] = set()
                    seen_workflow_ids: set[str] = set()
                    for ref in task_refs:
                        if not isinstance(ref, dict):
                            continue
                        task_id = ref.get("task_id", "")
                        workflow_id = ref.get("workflow_id", "")
                        if task_id and task_id not in seen_task_ids:
                            self._redis.xadd_telemetry(
                                task_log_stream_key(task_id),
                                {
                                    "payload": encoded,
                                    "workflow_id": workflow_id,
                                    "task_id": task_id,
                                },
                            )
                            seen_task_ids.add(task_id)
                        if workflow_id and workflow_id not in seen_workflow_ids:
                            self._redis.xadd_telemetry(
                                workflow_log_stream_key(workflow_id),
                                {
                                    "payload": encoded,
                                    "workflow_id": workflow_id,
                                },
                            )
                            seen_workflow_ids.add(workflow_id)
                except Exception as exc:
                    self.logger.exception("Failed to forward task log: %s", exc)
                continue
            if not isinstance(item, dict):
                self.logger.warning("Invalid event item: %s", item)
                continue

            try:
                event = parse_event(item)
                if isinstance(event, TaskEvent):
                    self.logger.debug("Forwarding task event: %s", event)
                    self._redis.xadd_telemetry(
                        TASK_EVENT_STREAM_KEY,
                        {"payload": json.dumps(serialize_event(event))},
                        maxlen=TASK_EVENT_STREAM_MAXLEN,
                    )
                elif isinstance(event, WorkerEvent):
                    self.logger.debug("Forwarding worker event: %s", event)
                    self._redis.publish_telemetry(
                        WORKER_EVENT_CHANNEL,
                        json.dumps(serialize_event(event)),
                    )
            except Exception as exc:
                self.logger.exception("Failed to handle worker event: %s", exc)
