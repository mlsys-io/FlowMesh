import json
import logging
import threading
import time
from dataclasses import dataclass, field

from shared.schemas.event import TaskEvent, WorkerEvent, serialize_event

from ..clients.redis import (
    TASK_EVENT_STREAM_KEY,
    TASK_EVENT_STREAM_MAXLEN,
    WORKER_EVENT_CHANNEL,
    SyncRedisClient,
)
from ..dispatcher import Dispatcher
from ..registries.worker import WorkerRegistry
from ..task.runtime import TaskRuntime


@dataclass
class _WatchdogState:
    stale_since: dict[str, float] = field(default_factory=dict)
    declared_dead: set[str] = field(default_factory=set)
    dead_since: dict[str, float] = field(default_factory=dict)
    reaped: set[str] = field(default_factory=set)


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
        reap_enabled: bool = False,
        reap_grace_seconds: int = 0,
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
        self._reap_enabled = reap_enabled
        self._reap_grace_seconds = max(0, reap_grace_seconds)
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
        state = _WatchdogState()
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

            self._scan(worker_ids, state, now)
            stop_event.wait(self._check_interval)

    def _scan(self, worker_ids: set[str], state: _WatchdogState, now: float) -> None:
        active_workers = worker_ids

        for worker_id in list(state.reaped):
            try:
                self._worker_registry.unregister_workers(worker_id)
            except Exception as exc:
                self._logger.warning(
                    "Worker watchdog failed to re-delete reaped worker %s: %s",
                    worker_id,
                    exc,
                )
                continue
            if self._publish_reap_event(worker_id):
                state.reaped.discard(worker_id)

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
                state.stale_since.pop(worker_id, None)
                state.declared_dead.discard(worker_id)
                state.dead_since.pop(worker_id, None)
                self.clear_dead_mark(worker_id)
                continue

            if worker_id in state.declared_dead:
                self._maybe_reap(worker_id, state, now)
                continue

            first_seen = state.stale_since.setdefault(worker_id, now)
            grace = self._grace_seconds
            if self._rehydration_grace_seconds > grace and (
                self._runtime.has_rehydrated_in_flight(
                    worker_id, self._rehydration_grace_seconds
                )
            ):
                grace = self._rehydration_grace_seconds
            if now - first_seen < grace:
                continue

            state.declared_dead.add(worker_id)
            state.dead_since[worker_id] = now
            self._mark_dead(worker_id)
            state.stale_since.pop(worker_id, None)
            self._handle_worker_expired(worker_id)

        for worker_id in list(state.stale_since):
            if worker_id not in active_workers:
                state.stale_since.pop(worker_id, None)
        state.declared_dead.intersection_update(active_workers)
        for worker_id in list(state.dead_since):
            if worker_id not in active_workers:
                state.dead_since.pop(worker_id, None)
        for worker_id in list(self._snapshot_dead_marks()):
            if worker_id not in active_workers:
                self.clear_dead_mark(worker_id)

    def _maybe_reap(self, worker_id: str, state: _WatchdogState, now: float) -> None:
        if not self._reap_enabled:
            return
        first = state.dead_since.get(worker_id)
        if first is None or now - first < self._reap_grace_seconds:
            return
        try:
            if not self._worker_registry.is_worker_stale(worker_id):
                return
        except Exception as exc:
            self._logger.debug(
                "Worker watchdog failed to re-check %s staleness before reap: %s",
                worker_id,
                exc,
            )
            return

        try:
            self._worker_registry.unregister_workers(worker_id)
        except Exception as exc:
            self._logger.warning(
                "Worker watchdog failed to reap worker %s: %s", worker_id, exc
            )
            return

        state.dead_since.pop(worker_id, None)
        state.declared_dead.discard(worker_id)
        state.stale_since.pop(worker_id, None)
        state.reaped.add(worker_id)
        self._logger.warning(
            "Reaped stale worker %s (dead for %.0fs)",
            worker_id,
            now - first,
        )
        if not self._publish_reap_event(worker_id):
            self._logger.warning(
                "Worker watchdog reaped %s but failed to publish UNREGISTER; "
                "will retry",
                worker_id,
            )

    def _publish_reap_event(self, worker_id: str) -> bool:
        try:
            event = WorkerEvent(
                type="UNREGISTER",
                worker_id=worker_id,
                payload={"reason": "reaped", "synthetic": True},
            )
            self._redis.publish_telemetry(
                WORKER_EVENT_CHANNEL,
                json.dumps(serialize_event(event), ensure_ascii=False),
            )
            return True
        except Exception as exc:
            self._logger.warning(
                "Worker watchdog failed to publish UNREGISTER for %s: %s",
                worker_id,
                exc,
            )
            return False

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
                    self._dispatcher.requeue_task(
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
