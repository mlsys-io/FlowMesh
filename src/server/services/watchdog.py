import json
import logging
import threading
import time

from shared.schemas.event import TaskEvent, serialize_event

from ..clients.redis import (
    TASK_EVENT_STREAM_KEY,
    TASK_EVENT_STREAM_MAXLEN,
    SyncRedisClient,
)
from ..dispatcher import Dispatcher
from ..registries.worker import WorkerRegistry
from ..task.runtime import TaskRuntime


class WorkerWatchdog:
    """Monitors worker heartbeats and emits synthetic failures when they expire."""

    def __init__(
        self,
        redis_client: SyncRedisClient,
        worker_registry: WorkerRegistry,
        runtime: TaskRuntime,
        dispatcher: Dispatcher,
        logger: logging.Logger,
        enabled: bool,
        check_interval: int,
        grace_seconds: int,
        rehydration_grace_seconds: int = 0,
    ) -> None:
        self._redis = redis_client
        self._worker_registry = worker_registry
        self._runtime = runtime
        self._dispatcher = dispatcher
        self._logger = logger
        self._enabled = enabled
        self._check_interval = max(1, check_interval)
        self._grace_seconds = max(0, grace_seconds)
        self._rehydration_grace_seconds = max(0, rehydration_grace_seconds)
        self._lock = threading.RLock()
        self._dead_marks: set[str] = set()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def is_marked_dead(self, worker_id: str) -> bool:
        if not self._enabled or not worker_id:
            return False
        with self._lock:
            return worker_id in self._dead_marks

    def clear_dead_mark(self, worker_id: str) -> None:
        if not self._enabled or not worker_id:
            return
        with self._lock:
            self._dead_marks.discard(worker_id)

    def start(self, stop_event: threading.Event) -> threading.Thread | None:
        if not self._enabled:
            return None
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self._thread
            thread = threading.Thread(
                target=self._watchdog_loop,
                args=(stop_event,),
                name="worker-watchdog",
                daemon=True,
            )
            thread.start()
            self._thread = thread
            return thread

    def _watchdog_loop(self, stop_event: threading.Event) -> None:
        stale_since: dict[str, float] = {}
        declared_dead: set[str] = set()
        while not stop_event.is_set():
            now = time.time()
            try:
                worker_ids = self._worker_registry.get_worker_ids()
            except Exception as exc:
                if stop_event.is_set():
                    break
                self._logger.debug("Worker watchdog failed to list workers: %s", exc)
                stop_event.wait(self._check_interval)
                continue

            active_workers = worker_ids
            for worker_id in worker_ids:
                if not worker_id:
                    continue
                try:
                    stale = self._worker_registry.is_worker_stale(worker_id)
                except Exception as exc:
                    self._logger.debug(
                        "Worker watchdog failed to check %s staleness: %s",
                        worker_id,
                        exc,
                    )
                    continue

                if not stale:
                    stale_since.pop(worker_id, None)
                    declared_dead.discard(worker_id)
                    self.clear_dead_mark(worker_id)
                    continue

                first_seen = stale_since.setdefault(worker_id, now)
                grace = self._grace_seconds
                if self._rehydration_grace_seconds > grace and (
                    self._runtime.has_rehydrated_in_flight(
                        worker_id, self._rehydration_grace_seconds
                    )
                ):
                    grace = self._rehydration_grace_seconds
                if now - first_seen < grace:
                    continue
                if worker_id in declared_dead:
                    continue

                declared_dead.add(worker_id)
                self._mark_dead(worker_id)
                stale_since.pop(worker_id, None)
                self._handle_worker_expired(worker_id)

            for worker_id in list(stale_since):
                if worker_id not in active_workers:
                    stale_since.pop(worker_id, None)
            declared_dead.intersection_update(active_workers)
            for worker_id in list(self._snapshot_dead_marks()):
                if worker_id not in active_workers:
                    self.clear_dead_mark(worker_id)
            stop_event.wait(self._check_interval)

    def _handle_worker_expired(self, worker_id: str) -> None:
        recovered = self._runtime.recover_tasks_for_worker(worker_id)
        if not recovered:
            self._logger.warning(
                "Worker %s heartbeat expired; no dispatched tasks to recover", worker_id
            )
            return

        self._logger.warning(
            "Worker %s heartbeat expired; emitting synthetic failures for %d task(s)",
            worker_id,
            len(recovered),
        )
        for task_id in recovered:
            payload = {
                "reason": "worker_heartbeat_expired",
                "worker": worker_id,
                "synthetic": True,
            }
            record = self._runtime.get_record(task_id)
            if record is not None:
                payload["attempt"] = record.attempts + 1
            event = TaskEvent(
                type="TASK_FAILED",
                task_id=task_id,
                worker_id=worker_id,
                error="worker_heartbeat_expired",
                payload=payload,
            )
            try:
                event_payload = json.dumps(serialize_event(event), ensure_ascii=False)
                self._redis.xadd_telemetry(
                    TASK_EVENT_STREAM_KEY,
                    {"payload": event_payload},
                    maxlen=TASK_EVENT_STREAM_MAXLEN,
                )
            except Exception as exc:
                self._logger.error(
                    "Failed to publish synthetic TASK_FAILED for %s "
                    "after worker %s expired: %s",
                    task_id,
                    worker_id,
                    exc,
                )
                try:
                    self._dispatcher._requeue_task(
                        task_id,
                        reason="worker_heartbeat_expired",
                        front=True,
                        extra_payload=payload,
                    )
                except Exception as requeue_exc:
                    self._logger.error(
                        "Failed to directly requeue %s after publish failure: %s",
                        task_id,
                        requeue_exc,
                    )

    def _mark_dead(self, worker_id: str) -> None:
        if not worker_id:
            return
        with self._lock:
            self._dead_marks.add(worker_id)

    def _snapshot_dead_marks(self) -> set[str]:
        with self._lock:
            return set(self._dead_marks)
