import asyncio
import json
import logging
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from shared.schemas.command import InterruptMessage
from shared.schemas.event import (
    Event,
    NodeEvent,
    TaskEvent,
    WorkerEvent,
    parse_event,
)
from shared.schemas.result import result_file_path
from shared.schemas.worker import WorkerStatus
from shared.utils.manifest import RESULTS_NAME, sync_manifest

from ..auth import default_principal, deregister_resource, register_resource
from ..clients.redis import (
    NODE_EVENT_CHANNEL,
    REDIS_CONN_ERRORS,
    TASK_EVENT_CURSOR_KEY,
    TASK_EVENT_STREAM_KEY,
    TASK_EVENT_STREAM_MAXLEN,
    WORKER_EVENT_CHANNEL,
    SyncRedisClient,
    iter_pubsub_messages,
    task_log_closed_key,
    task_log_stream_key,
    workflow_key,
    workflow_log_closed_key,
    workflow_log_stream_key,
    workflow_tasks_key,
)
from ..dispatcher import Dispatcher
from ..hooks import (
    RESOURCE_REGISTRARS,
    USAGE_SINKS,
    PrincipalContext,
    ResourceKind,
    UsageRow,
)
from ..registries.node import NodeRegistry
from ..registries.worker import WorkerRegistry
from ..schemas.logs import LogEvent
from ..task.metadata import extract_model_dataset_names
from ..task.models import TaskRecord, TaskStatus, TaskUsage
from ..task.runtime import TaskRuntime
from ..utils.logging import log_node_event, log_worker_event
from ..utils.time import now_iso
from .metrics import MetricsRecorder
from .port_forward import PortForwardService
from .watchdog import WorkerWatchdog

TASK_EVENT_HANDLER_MAX_ATTEMPTS = 5


def _stream_id_tuple(entry_id: str) -> tuple[int, int]:
    """Parse a Redis stream id (``<ms>-<seq>``) into a comparable tuple."""
    ms, _, seq = entry_id.partition("-")
    try:
        return int(ms), int(seq or 0)
    except ValueError:
        return 0, 0


def failed_task_can_retry(record: TaskRecord | None, retryable: bool | None) -> bool:
    """Whether a failed task may be requeued: retryable and within the attempt
    budget."""
    if record is None:
        return False
    if record.status in (TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DONE):
        return False
    if retryable is False:
        return False
    max_attempts = record.max_attempts
    return max_attempts is None or max_attempts < 0 or record.attempts < max_attempts


class EventMonitor:
    """Background listeners for task, worker, and server event streams."""

    def __init__(
        self,
        redis_client: SyncRedisClient,
        logger: logging.Logger,
        runtime: TaskRuntime,
        dispatcher: Dispatcher,
        worker_registry: WorkerRegistry,
        node_registry: NodeRegistry,
        metrics_recorder: MetricsRecorder,
        watchdog: WorkerWatchdog,
        ssh_proxy_enabled: bool = False,
        port_forward: PortForwardService | None = None,
        results_dir: Path | str = ".",
        log_stream_ttl_sec: int = 0,
        server_base_url: str = "http://localhost:8000",
    ) -> None:
        self._redis_client = redis_client
        self._stop_event = threading.Event()
        self._logger = logger
        self._runtime = runtime
        self._dispatcher = dispatcher
        self._worker_registry = worker_registry
        self._node_registry = node_registry
        self._metrics = metrics_recorder
        self._watchdog = watchdog
        self._ssh_proxy_enabled = ssh_proxy_enabled
        self._port_forward = port_forward
        self._results_dir = Path(results_dir)
        self._log_stream_ttl_sec = max(0, int(log_stream_ttl_sec))
        self._server_base_url = self._validate_server_base_url(server_base_url)

        self._pending_result_clones: dict[str, list[str]] = {}
        self._pending_lock = threading.RLock()

        # Per-entry handler-failure counts backing the consumer's retry budget.
        self._event_handler_attempts: dict[str, int] = {}

        self._loop: asyncio.AbstractEventLoop | None = None
        self._threads: list[threading.Thread] | None = None
        self._pending_coros: set[Future[Any]] = set()
        self._pending_coros_lock = threading.Lock()

        # Own-node tracking to coordinate with `stop()` during lifespan teardown.
        # Set by `set_own_node()` once the supervisor handshake produces a node_id.
        self._own_node_id: str | None = None
        self._own_node_deregistered: asyncio.Event = asyncio.Event()

    def _validate_server_base_url(self, server_base_url: str) -> str:
        """Validate the server's public base URL used to advertise serve-proxy
        endpoints, falling back to a safe default instead of silently producing a broken
        advertised URL."""
        fallback = "http://localhost:8000"
        try:
            parsed = urlparse(server_base_url)
            valid = parsed.scheme in ("http", "https") and bool(parsed.hostname)
        except ValueError:
            valid = False
        if valid:
            return server_base_url
        self._logger.error(
            "Invalid server_base_url %r; falling back to %r for serve-proxy URL "
            "advertisement",
            server_base_url,
            fallback,
        )
        return fallback

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Spawn task/worker listener threads."""
        if self._threads is not None:
            self._logger.warning("Event monitor already started")
            return
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                raise RuntimeError(
                    "Event monitor must be started inside an event loop: %s", exc
                )
        self._threads = [
            threading.Thread(
                target=self._tasks_events_loop, name="tasks-events", daemon=True
            ),
            threading.Thread(
                target=self._workers_events_loop, name="workers-events", daemon=True
            ),
            threading.Thread(
                target=self._node_events_loop, name="nodes-events", daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    def set_own_node(self, node_id: str) -> None:
        """Record the node_id of the supervisor co-located with this server."""
        self._own_node_id = node_id

    async def stop(
        self,
        thread_join_timeout: float = 2.0,
        coro_timeout: float = 5.0,
        deregister_timeout: float = 5.0,
    ) -> None:
        """Drain in-flight events and pending hook coroutines, then exit."""
        if self._threads is None:
            self._logger.warning("Event monitor not started")
            return

        if self._own_node_id is not None:
            try:
                await asyncio.wait_for(
                    self._own_node_deregistered.wait(), timeout=deregister_timeout
                )
            except TimeoutError:
                self._logger.warning(
                    "EventMonitor.stop: SV_UNREGISTER for own node %s did not "
                    "fire within %ss; deregister hook may have been missed",
                    self._own_node_id,
                    deregister_timeout,
                )

        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=thread_join_timeout)
        self._threads = None

        with self._pending_coros_lock:
            pending = list(self._pending_coros)
            self._pending_coros.clear()
        if not pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(asyncio.wrap_future(f) for f in pending),
                    return_exceptions=True,
                ),
                timeout=coro_timeout,
            )
        except TimeoutError:
            self._logger.warning(
                "EventMonitor.stop: %d hook coroutine(s) did not finish within %ss",
                len(pending),
                coro_timeout,
            )

    def mirror_task_results(self, parent_task_id: str, child_ids: list[str]) -> None:
        if not child_ids:
            return
        parent_dir = result_file_path(self._results_dir, parent_task_id).parent
        if not parent_dir.exists():
            with self._pending_lock:
                pending = self._pending_result_clones.setdefault(parent_task_id, [])
                for child in child_ids:
                    if child not in pending:
                        pending.append(child)
            self._logger.debug(
                "Deferring result mirroring for %s (waiting for artifacts)",
                parent_task_id,
            )
            return

        for child_id in child_ids:
            if child_id == parent_task_id:
                continue
            dst_dir = result_file_path(self._results_dir, child_id).parent
            if dst_dir.exists() and (dst_dir / RESULTS_NAME).exists():
                continue
            try:
                if dst_dir.exists():
                    shutil.rmtree(dst_dir, ignore_errors=True)
                shutil.copytree(parent_dir, dst_dir)
                record = self._runtime.get_record(child_id)
                expected_artifacts: list[str] = []
                if record:
                    expected_artifacts = record.task.spec.get_artifacts()
                sync_manifest(dst_dir, child_id, expected_artifacts)
            except Exception as exc:
                self._logger.debug(
                    "Failed to mirror results from %s to %s: %s",
                    parent_task_id,
                    child_id,
                    exc,
                )

        with self._pending_lock:
            self._pending_result_clones.pop(parent_task_id, None)

    def pop_pending_clones(self, task_id: str) -> list[str]:
        with self._pending_lock:
            return self._pending_result_clones.pop(task_id, [])

    # ------------------------------------------------------------------ #
    # Task event handling
    # ------------------------------------------------------------------ #

    def _tasks_events_loop(self) -> None:
        """Consume task events from a durable Redis stream.

        Replay contract: a task transition is persisted to durable scheduler state
        *before* its event is emitted, and the consumer advances the persisted cursor
        only *after* an entry is handled. Delivery is therefore at-least-once — a crash
        between handling an entry and persisting the cursor replays that entry on the
        next startup. Event handlers are idempotent (terminal tasks ignore late
        dispatch/start/update events and repeated completions), so replay never
        double-applies.

        Resuming from the persisted cursor is what lets events emitted while
        the server was down (e.g. during a rolling restart) be replayed rather
        than lost. The cursor is kept on the control Redis while the stream
        lives on the telemetry Redis, so durable resume relies on both volumes
        surviving — which the rolling-restart flow already requires.

        The stream is length-bounded (``TASK_EVENT_STREAM_MAXLEN``), so a
        consumer that stays down long enough for its cursor to fall behind the
        trim horizon loses the events in between. That is detected and logged on
        resume rather than passing silently.
        """
        cursor = self._redis_client.get(TASK_EVENT_CURSOR_KEY) or "0-0"
        self._warn_if_cursor_trimmed(cursor)
        while not self._stop_event.is_set():
            try:
                rows = self._redis_client.xread_telemetry(
                    {TASK_EVENT_STREAM_KEY: cursor}, count=200, block_ms=1000
                )
                for _, entries in rows:
                    cursor = self._consume_stream_batch(entries, cursor)
                    if self._stop_event.is_set():
                        break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._logger.warning(
                    "%s listener error: %s; reconnecting...", TASK_EVENT_STREAM_KEY, exc
                )
                time.sleep(2.0)
                continue

    def _consume_stream_batch(
        self, entries: list[tuple[str, dict[str, Any]]], cursor: str
    ) -> str:
        """Handle one batch of stream entries, advancing the cursor per entry.

        Parse failures are skipped; handler failures are retried without advancing the
        cursor, then dead-lettered after ``TASK_EVENT_HANDLER_MAX_ATTEMPTS`` attempts.
        """
        for entry_id, fields in entries:
            try:
                event = self._parse_stream_event(fields)
            except Exception as exc:
                self._logger.error(
                    "Failed to parse task event %s; skipping malformed entry: %s",
                    entry_id,
                    exc,
                )
                self._event_handler_attempts.pop(entry_id, None)
                cursor = self._advance_event_cursor(entry_id)
                if self._stop_event.is_set():
                    break
                continue

            if isinstance(event, TaskEvent):
                try:
                    self._handle_task_event(event)
                except REDIS_CONN_ERRORS:
                    # Propagate so the loop backs off and replays from this cursor.
                    raise
                except Exception as exc:
                    # Don't advance past a handler failure: it may be transient,
                    # and the watchdog only reclaims dead workers (not a task stuck
                    # under a live one), so advancing would lose the transition.
                    attempts = self._event_handler_attempts.get(entry_id, 0) + 1
                    if attempts < TASK_EVENT_HANDLER_MAX_ATTEMPTS:
                        self._event_handler_attempts[entry_id] = attempts
                        self._logger.warning(
                            "Failed to apply task event %s (attempt %d/%d); "
                            "will retry: %s",
                            entry_id,
                            attempts,
                            TASK_EVENT_HANDLER_MAX_ATTEMPTS,
                            exc,
                        )
                        return cursor
                    self._logger.error(
                        "Dropping task event %s after %d failed attempts: %s",
                        entry_id,
                        attempts,
                        exc,
                    )
                self._event_handler_attempts.pop(entry_id, None)

            cursor = self._advance_event_cursor(entry_id)
            if self._stop_event.is_set():
                break
        return cursor

    def _advance_event_cursor(self, entry_id: str) -> str:
        self._redis_client.set_value(TASK_EVENT_CURSOR_KEY, entry_id)
        return entry_id

    def _warn_if_cursor_trimmed(self, cursor: str) -> None:
        """Log if the persisted cursor has fallen behind the stream's trim horizon.

        Trimming removes entries from the front of the stream, so the oldest
        surviving entry being newer than the cursor means every entry between
        them was discarded before this consumer read it.
        """
        if cursor in ("$", "0", "0-0"):
            return
        try:
            first = self._redis_client.xrange_telemetry(TASK_EVENT_STREAM_KEY, count=1)
        except Exception as exc:
            self._logger.debug(
                "Could not inspect %s head: %s", TASK_EVENT_STREAM_KEY, exc
            )
            return
        if not first:
            return
        first_id = first[0][0]
        if _stream_id_tuple(first_id) > _stream_id_tuple(cursor):
            self._logger.warning(
                "%s cursor %s fell behind the stream head %s (maxlen=%d); "
                "events in between were trimmed before they were consumed",
                TASK_EVENT_STREAM_KEY,
                cursor,
                first_id,
                TASK_EVENT_STREAM_MAXLEN,
            )

    def _parse_stream_event(self, fields: dict[str, Any]) -> Event | None:
        raw = fields.get("payload")
        if raw is None:
            return None
        try:
            return parse_event(json.loads(raw))
        except Exception as exc:
            self._logger.debug("Failed to parse task stream event: %s (%s)", raw, exc)
            return None

    def _handle_task_event(self, event: TaskEvent) -> None:
        payload = event.payload or {}
        event_type = event.type
        match event_type:
            case "TASK_STARTED":
                self._metrics.record_task_event(event)
                self._runtime.mark_started(
                    event.task_id, event.worker_id, payload, event.ts
                )
            case "TASK_UPDATE":
                payload = self._handle_ssh_task_update(
                    event.task_id, event.worker_id, payload
                )
                payload = self._handle_serve_task_update(
                    event.task_id, event.worker_id, payload
                )
                self._runtime.mark_updated(event.task_id, payload)
            case "TASK_SUCCEEDED":
                self._unregister_port_forward(event.task_id)
                self._metrics.record_task_event(event)
                merged_children = self._runtime.get_merged_children(event.task_id)
                if merged_children:
                    self.mirror_task_results(event.task_id, merged_children)
                usages = self._runtime.mark_succeeded(
                    event.task_id, event.worker_id, payload, event.ts
                )
                self._schedule_emit_usage(usages)
                self._close_task_log_stream(event.task_id)
                try:
                    queueing, dispatched, pending, done, total = (
                        self._runtime.task_status_counts()
                    )
                    summary = (
                        f"QUEUEING {queueing}, DISPATCHED {dispatched}, "
                        f"PENDING {pending}, DONE {done}, TOTAL {total}"
                    )
                except Exception:
                    summary = (
                        "QUEUEING UNKNOWN, DISPATCHED UNKNOWN, PENDING UNKNOWN, "
                        "DONE UNKNOWN, TOTAL UNKNOWN"
                    )
                self._logger.info("Task %s completed; %s", event.task_id, summary)
                if merged_children:
                    for child_id in merged_children:
                        child_payload = dict(payload)
                        child_payload["parent_task_id"] = event.task_id
                        child_payload["is_child_task"] = True
                        child_event = TaskEvent(
                            type="TASK_SUCCEEDED",
                            task_id=child_id,
                            worker_id=event.worker_id,
                            payload=child_payload,
                            ts=event.ts,
                        )
                        self._metrics.record_task_event(child_event, is_child=True)
                        self._close_task_log_stream(child_id)
                        self._maybe_close_workflow_log_stream(child_id)
                if event.worker_id:
                    try:
                        record = self._runtime.get_record(event.task_id)
                        if record:
                            models, datasets = extract_model_dataset_names(record.task)
                            if models or datasets:
                                self._worker_registry.record_worker_cache(
                                    event.worker_id,
                                    models=models,
                                    datasets=datasets,
                                )
                    except Exception as exc:
                        self._logger.debug(
                            "Failed to update cache metadata for worker %s: %s",
                            event.worker_id,
                            exc,
                        )
                    try:
                        self._worker_registry.update_worker_status(
                            event.worker_id, WorkerStatus.IDLE
                        )
                    except Exception:
                        pass
                self._maybe_close_workflow_log_stream(event.task_id)
            case "TASK_FAILED":
                record = self._runtime.get_record(event.task_id)
                if record:
                    if event.worker_id and event.worker_id not in record.failed_workers:
                        record.failed_workers.append(event.worker_id)
                    if event.error:
                        record.last_error = event.error
                    attempts = record.attempts
                    max_attempts: int | None = record.max_attempts
                else:
                    attempts = 0
                    max_attempts = None

                if failed_task_can_retry(record, event.retryable):
                    self._unregister_port_forward(event.task_id)
                    limit_display = (
                        "∞"
                        if max_attempts is None or max_attempts < 0
                        else max_attempts
                    )
                    self._logger.warning(
                        "Retrying task %s after failure (%d/%s)",
                        event.task_id,
                        attempts + 1,
                        limit_display,
                    )
                    self._dispatcher.requeue_task(
                        event.task_id,
                        reason="worker_failed",
                        front=True,
                        extra_payload={
                            "error": event.error,
                            "attempt": attempts + 1,
                            "max_attempts": max_attempts,
                        },
                    )
                    return

                self._unregister_port_forward(event.task_id)
                self._metrics.record_task_event(event)
                impacted, merged_children, usages = self._runtime.mark_failed(
                    event.task_id,
                    event.worker_id,
                    payload,
                    event.ts,
                    error=(record.last_error if record else None) or event.error,
                )
                self._schedule_emit_usage(usages)
                self._metrics.finalize_task_failure(event.task_id)
                self._close_task_log_stream(event.task_id)
                for task_id, reason in impacted:
                    derived = TaskEvent(
                        type="TASK_FAILED",
                        task_id=task_id,
                        error=reason,
                        payload={"dependency_failure": event.task_id},
                    )
                    self._metrics.record_task_event(derived)
                    self._metrics.finalize_task_failure(task_id)
                    self._close_task_log_stream(task_id)
                    self._maybe_close_workflow_log_stream(task_id)
                for child_id in merged_children:
                    child_payload = dict(payload)
                    child_payload["parent_task_id"] = event.task_id
                    child_payload["dependency_failure"] = event.task_id
                    child_payload["is_child_task"] = True
                    child_event = TaskEvent(
                        type="TASK_FAILED",
                        task_id=child_id,
                        worker_id=event.worker_id,
                        error=event.error or "parent_failed",
                        payload=child_payload,
                    )
                    self._metrics.record_task_event(child_event, is_child=True)
                    self._metrics.finalize_task_failure(child_id)
                    self._close_task_log_stream(child_id)
                    self._maybe_close_workflow_log_stream(child_id)
                self._maybe_close_workflow_log_stream(event.task_id)
            case "TASK_CANCELLED":
                self._unregister_port_forward(event.task_id)
                self._metrics.record_task_event(event)
                usages = self._runtime.mark_cancelled(
                    event.task_id,
                    event.worker_id,
                    payload,
                    event.ts,
                )
                self._schedule_emit_usage(usages)
                self._metrics.finalize_task_cancellation(event.task_id)
                self._close_task_log_stream(event.task_id)
                if event.worker_id:
                    try:
                        self._worker_registry.update_worker_status(
                            event.worker_id, WorkerStatus.IDLE
                        )
                    except Exception:
                        pass
                self._maybe_close_workflow_log_stream(event.task_id)
            case _:
                self._logger.debug(
                    "Ignoring task event type=%s payload=%s", event_type, payload
                )

    # ------------------------------------------------------------------ #
    # Node event handling
    # ------------------------------------------------------------------ #

    def _node_events_loop(self) -> None:
        event_key = NODE_EVENT_CHANNEL
        while not self._stop_event.is_set():
            try:
                for event in self._get_event_stream(event_key):
                    if isinstance(event, NodeEvent):
                        self._handle_node_event(event)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._logger.warning(
                    "%s listener error: %s; reconnecting...", event_key, exc
                )
                time.sleep(2.0)

    def _handle_node_event(self, event: NodeEvent) -> None:
        log_node_event(self._logger, event)
        event_type = event.type
        match event_type:
            case "SV_REGISTER":
                # Registry is already populated by the supervisor's lifecycle
                # Fire the register hook with the actor stamped on the event.
                self._schedule_register(
                    ResourceKind.NODE,
                    event.node_id,
                    self._actor_from_event(event),
                    {"tags": list(event.tags or [])},
                )
            case "SV_HEARTBEAT":
                ttl_sec = event.payload.get("ttl_sec", 120)
                current_gpu_count = event.payload.get("current_gpu_count")
                self._node_registry.update_node_hb(
                    event.node_id,
                    event.ts,
                    ttl_sec,
                    current_gpu_count=(
                        int(current_gpu_count)
                        if current_gpu_count is not None
                        else None
                    ),
                )
            case "SV_UNREGISTER":
                self._node_registry.unregister_node(event.node_id)
                actor = self._actor_from_event(event)
                if self._own_node_id is not None and event.node_id == self._own_node_id:
                    self._schedule_own_node_deregister(event.node_id, actor)
                else:
                    self._schedule_deregister(ResourceKind.NODE, event.node_id, actor)
            case _:
                self._logger.debug(
                    "Ignoring node event type=%s payload=%s",
                    event_type,
                    event.payload,
                )

    # ------------------------------------------------------------------ #
    # Worker event handling
    # ------------------------------------------------------------------ #

    def _workers_events_loop(self) -> None:
        event_key = WORKER_EVENT_CHANNEL
        while not self._stop_event.is_set():
            try:
                for event in self._get_event_stream(event_key):
                    if isinstance(event, WorkerEvent):
                        self._handle_worker_event(event)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._logger.warning(
                    "%s listener error: %s; reconnecting...", event_key, exc
                )
                time.sleep(2.0)

    def _handle_worker_event(self, event: WorkerEvent) -> None:
        log_worker_event(self._logger, event)
        self._metrics.record_worker_event(event)
        event_type = event.type
        match event_type:
            case "REGISTER":
                worker_id = (event.worker_id or "").strip()
                if worker_id:
                    self._schedule_register(
                        ResourceKind.WORKER,
                        worker_id,
                        self._actor_from_event(event),
                        {"tags": list(event.tags or [])},
                    )
            case "HEARTBEAT":
                worker_id = (event.worker_id or "").strip()
                ttl_sec = event.payload.get("ttl_sec", 120)
                self._worker_registry.update_worker_hb(worker_id, event.ts, ttl_sec)
            case "STATUS":
                worker_id = (event.worker_id or "").strip()
                status = event.status or WorkerStatus.UNKNOWN
                self._worker_registry.set_worker_status(
                    worker_id, status, event.ts, event.payload
                )
            case "UNREGISTER":
                worker_id = (event.worker_id or "").strip()
                self._worker_registry.unregister_workers(worker_id)
                if worker_id:
                    self._schedule_deregister(
                        ResourceKind.WORKER, worker_id, self._actor_from_event(event)
                    )
                    if self._watchdog.enabled and self._watchdog.is_marked_dead(
                        worker_id
                    ):
                        self._logger.info(
                            "Skipping direct requeue for %s unregister; watchdog "
                            "already emitted synthetic failures",
                            worker_id,
                        )
                        self._watchdog.clear_dead_mark(worker_id)
                        return
                    recovered = self._runtime.recover_tasks_for_worker(worker_id)
                    if recovered:
                        to_requeue: list[str] = []
                        ts = now_iso()
                        for task_id in recovered:
                            self._unregister_port_forward(task_id)
                            record = self._runtime.get_record(task_id)
                            if record and record.status == TaskStatus.CANCELLING:
                                self._runtime.mark_cancelled(task_id, worker_id, {}, ts)
                                self._close_task_log_stream(task_id)
                                self._maybe_close_workflow_log_stream(task_id)
                            else:
                                to_requeue.append(task_id)
                        if to_requeue:
                            self._logger.info(
                                "Requeued %d task(s) after worker %s unregistered: %s",
                                len(to_requeue),
                                worker_id,
                                ", ".join(to_requeue),
                            )
                            for task_id in to_requeue:
                                self._dispatcher.requeue_task(
                                    task_id,
                                    reason="worker_unregistered",
                                    front=True,
                                    extra_payload={"worker": worker_id},
                                )
            case _:
                self._logger.debug(
                    "Ignoring task event type=%s payload=%s", event_type, event.payload
                )

    # ------------------------------------------------------------------ #
    # SSH / serve forward task handling
    # ------------------------------------------------------------------ #

    def _handle_port_forward_update(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        *,
        key: str,
        normalize_mode: Callable[[str, str | None], str | None],
        inject_session_id: bool,
        strip_relay_target_after: bool,
        on_registration_failure: Literal["fall_back_direct", "fail_task", "drop"],
        proxy_endpoint_path: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        """Register a port-forward relay for a task update payload."""
        inner = payload.get(key)
        if not isinstance(inner, dict):
            return payload
        payload = payload.copy()

        mode = str(inner.get("mode") or "direct")
        normalized_mode = normalize_mode(mode, worker_id)
        if normalized_mode is None:
            match on_registration_failure:
                case "fail_task":
                    payload.pop(key, None)
                    self._fail_forward_task(
                        task_id,
                        worker_id,
                        f"{key} access mode {mode!r} could not be served",
                    )
                case "drop":
                    payload.pop(key, None)
            return payload
        if normalized_mode != mode:
            inner = inner.copy()
            inner["mode"] = normalized_mode
            if normalized_mode == "direct":
                inner.pop("directHost", None)
                inner.pop("directPort", None)
                inner.pop("_relay_target", None)
            payload[key] = inner

        if normalized_mode == "proxy" and proxy_endpoint_path is not None:
            inner = inner.copy()
            parsed_base = urlparse(self._server_base_url)
            inner["mode"] = "proxy"
            inner["host"] = parsed_base.hostname or self._server_base_url
            if parsed_base.port is not None:
                inner["port"] = parsed_base.port
            else:
                inner.pop("port", None)
            inner["url"] = (
                f"{self._server_base_url.rstrip('/')}{proxy_endpoint_path(task_id)}"
            )
            payload[key] = inner
            return payload

        if normalized_mode != "forward":
            return payload

        assert self._port_forward is not None
        assert worker_id is not None
        record = self._runtime.get_record(task_id)
        try:
            forward_input = inner.copy()
            if inject_session_id:
                forward_input.setdefault("session_id", task_id)
            inner = self._port_forward.register_port_forward(
                task_id,
                record.workflow_id if record is not None else None,
                worker_id,
                forward_input,
            )
            if inject_session_id:
                inner.pop("session_id", None)
            if strip_relay_target_after:
                inner.pop("_relay_target", None)
        except Exception as exc:
            self._logger.warning(
                "Failed to register forward target for task %s (%s): %s",
                task_id,
                key,
                exc,
            )
            match on_registration_failure:
                case "fall_back_direct":
                    inner = inner.copy()
                    inner["mode"] = "direct"
                    inner.pop("_relay_target", None)
                    payload[key] = inner
                    return payload
                case "fail_task":
                    payload.pop(key, None)
                    self._fail_forward_task(
                        task_id,
                        worker_id,
                        f"failed to register {key} forward target: {exc}",
                    )
                case "drop":
                    payload.pop(key, None)
            return payload

        payload[key] = inner
        return payload

    def _fail_forward_task(
        self, task_id: str, worker_id: str | None, reason: str
    ) -> None:
        """Fail a task whose forward endpoint was dropped with no fallback and stop its
        executor to free the resources it holds."""
        self._dispatcher.fail_task(
            task_id, reason, worker_id=worker_id, payload={"error": reason}
        )
        if not worker_id:
            return
        worker = self._worker_registry.get_worker(worker_id)
        if worker is None:
            return
        try:
            self._worker_registry.publish_interrupt(
                worker,
                InterruptMessage(task_id=task_id, worker_id=worker.id, reason=reason),
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to publish interrupt for task %s on worker %s: %s",
                task_id,
                worker_id,
                exc,
            )

    def _handle_ssh_task_update(
        self, task_id: str, worker_id: str | None, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle SSH forward registration for task updates."""
        return self._handle_port_forward_update(
            task_id,
            worker_id,
            payload,
            key="ssh",
            normalize_mode=self._normalize_ssh_mode,
            inject_session_id=False,
            strip_relay_target_after=False,
            on_registration_failure="fall_back_direct",
        )

    def _handle_serve_task_update(
        self, task_id: str, worker_id: str | None, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle serve forward/proxy registration for task updates."""
        return self._handle_port_forward_update(
            task_id,
            worker_id,
            payload,
            key="serve",
            normalize_mode=self._normalize_serve_mode,
            inject_session_id=True,
            strip_relay_target_after=True,
            on_registration_failure="fail_task",
            proxy_endpoint_path=lambda tid: f"/api/v1/serve/tasks/{tid}",
        )

    def _track_pending(self, fut: Future[Any]) -> None:
        fut.add_done_callback(self._untrack_pending)
        with self._pending_coros_lock:
            if not fut.done():
                self._pending_coros.add(fut)

    def _untrack_pending(self, fut: Future[Any]) -> None:
        with self._pending_coros_lock:
            self._pending_coros.discard(fut)

    def _schedule_emit_usage(self, usages: list[tuple[str, TaskUsage]]) -> None:
        if not usages or not USAGE_SINKS:
            return
        if self._loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._emit_usage_async(usages), self._loop
            )
        except RuntimeError as exc:
            self._logger.debug("Failed to schedule usage delivery: %s", exc)
            return
        self._track_pending(fut)

    def _actor_from_event(self, event: WorkerEvent | NodeEvent) -> PrincipalContext:
        return (
            default_principal()
            if event.actor is None
            else PrincipalContext.model_validate(event.actor)
        )

    def _schedule_register(
        self,
        resource_kind: ResourceKind,
        resource_id: str,
        principal: PrincipalContext,
        metadata: Mapping[str, Any],
    ) -> None:
        if not RESOURCE_REGISTRARS or not resource_id or self._loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                register_resource(
                    principal, resource_kind, resource_id, metadata, self._logger
                ),
                self._loop,
            )
        except RuntimeError as exc:
            self._logger.debug(
                "Failed to schedule %s/%s register: %s",
                resource_kind.value,
                resource_id,
                exc,
            )
            return
        self._track_pending(fut)

    def _schedule_deregister(
        self, resource_kind: ResourceKind, resource_id: str, principal: PrincipalContext
    ) -> None:
        if not RESOURCE_REGISTRARS or not resource_id or self._loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                deregister_resource(
                    principal, resource_kind, resource_id, self._logger
                ),
                self._loop,
            )
        except RuntimeError as exc:
            self._logger.debug(
                "Failed to schedule %s/%s deregister: %s",
                resource_kind.value,
                resource_id,
                exc,
            )
            return
        self._track_pending(fut)

    def _schedule_own_node_deregister(
        self, node_id: str, principal: PrincipalContext
    ) -> None:
        """Schedule the deregister for the co-located supervisor's node and
        signal `_own_node_deregistered` once the coroutine completes."""

        def on_done(*_: Any) -> None:
            self._own_node_deregistered.set()

        if self._loop is None:
            return
        if not RESOURCE_REGISTRARS:
            self._loop.call_soon_threadsafe(on_done)
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                deregister_resource(
                    principal, ResourceKind.NODE, node_id, self._logger
                ),
                self._loop,
            )
            fut.add_done_callback(on_done)
        except RuntimeError as exc:
            self._logger.debug(
                "Failed to schedule own-node %s deregister: %s", node_id, exc
            )
            return
        self._track_pending(fut)

    async def _emit_usage_async(self, usages: list[tuple[str, TaskUsage]]) -> None:
        if not usages or not USAGE_SINKS:
            return
        rows: list[UsageRow] = []
        for task_id, usage in usages:
            record = self._runtime.get_record(task_id)
            if not record:
                continue
            try:
                finished_at = usage.finished_at
                if finished_at.endswith("Z"):
                    finished_at = finished_at[:-1] + "+00:00"
                occurred_at = datetime.fromisoformat(finished_at)
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=UTC)
            except Exception:
                continue
            try:
                cost = Decimal(str(usage.total_cost))
            except (InvalidOperation, TypeError):
                continue
            rows.append(
                UsageRow(
                    org_id=record.org_id,
                    principal_id=record.owner_id,
                    supplier_id=record.supplier_id or None,
                    occurred_at=occurred_at,
                    cost=cost,
                    task_id=task_id,
                    runtime_sec=float(usage.runtime_sec),
                    cost_per_hour=float(usage.cost_per_hour),
                    task_status=str(usage.status),
                )
            )
        if not rows:
            return
        for sink in USAGE_SINKS:
            try:
                await sink.emit(rows, self._logger)
            except Exception as exc:
                self._logger.warning("Usage sink %s failed: %s", sink.name, exc)

    def _unregister_port_forward(self, task_id: str) -> None:
        if self._port_forward is None:
            return
        try:
            self._port_forward.unregister_task(task_id)
        except Exception as exc:
            self._logger.debug(
                "Failed to unregister forward target for task %s: %s", task_id, exc
            )

    def _normalize_ssh_mode(self, mode: str, worker_id: str | None) -> str:
        # `forward` -> `proxy` -> `direct` fallback
        if mode == "proxy":
            return "proxy" if self._ssh_proxy_enabled else "direct"
        if mode == "forward":
            if not worker_id:
                self._logger.warning(
                    "Cannot keep SSH forward mode without worker_id; "
                    "degrading access mode"
                )
                return "proxy" if self._ssh_proxy_enabled else "direct"
            if self._port_forward is not None:
                return "forward"
            if self._ssh_proxy_enabled:
                return "proxy"
            return "direct"
        return "direct"

    def _normalize_serve_mode(self, mode: str, worker_id: str | None) -> str | None:
        """Normalize the serve access mode.

        Returns None when the endpoint cannot be served (e.g. no forward service).
        """
        if mode == "direct":
            return "direct"
        if mode == "forward":
            if not worker_id:
                self._logger.warning(
                    "Serve task with forward mode has no worker_id; dropping endpoint"
                )
                return None
            if self._port_forward is None:
                self._logger.error(
                    "Serve task requested forward mode but no port-forward service "
                    "is configured; dropping endpoint"
                )
                return None
            return "forward"
        if mode == "proxy":
            if not worker_id:
                self._logger.warning(
                    "Serve task with proxy mode has no worker_id; dropping endpoint"
                )
                return None
            if not self._ssh_proxy_enabled:
                self._logger.error(
                    "Serve task requested proxy mode but the relay proxy is "
                    "disabled; dropping endpoint"
                )
                return None
            return "proxy"
        self._logger.warning(
            "Unsupported serve access mode %r; dropping endpoint", mode
        )
        return None

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _get_event_stream(self, topic: str) -> Iterable[Event]:
        pubsub = self._redis_client.subscribe_telemetry(topic)
        for data in iter_pubsub_messages(pubsub):
            if not isinstance(data, dict):
                self._logger.debug("Invalid %s payload: %s", topic, type(data))
                continue
            try:
                event = parse_event(data)
            except Exception as exc:
                self._logger.debug(
                    "Failed to parse %s event: %s (%s)", topic, data, exc
                )
                continue
            yield event
            # Stop AFTER yielding so a message that arrives concurrently with
            # `stop_event.set()` (e.g. SV_UNREGISTER published while the
            # supervisor subprocess is shutting down) still reaches the
            # handler and gets a chance to schedule its hook coroutine.
            if self._stop_event.is_set():
                break
        try:
            pubsub.close()
        except Exception:
            pass

    def _close_task_log_stream(self, task_id: str) -> None:
        record = self._runtime.get_record(task_id)
        if not record:
            return
        event = LogEvent(
            ts=now_iso(),
            workflow_id=record.workflow_id,
            task_id=task_id,
            level="INFO",
            stream="system",
            source="server",
            message="Task log stream closed.",
        )
        payload = event.model_dump(exclude_none=True)
        payload["type"] = "LOG_STREAM_CLOSED"
        encoded = json.dumps(payload, ensure_ascii=False)
        try:
            self._redis_client.xadd_telemetry(
                task_log_stream_key(task_id),
                {
                    "payload": encoded,
                    "workflow_id": record.workflow_id,
                    "task_id": task_id,
                },
            )
            self._redis_client.set_value(task_log_closed_key(task_id), "1")
            if self._log_stream_ttl_sec:
                self._redis_client.expire_telemetry(
                    task_log_stream_key(task_id), self._log_stream_ttl_sec
                )
                self._redis_client.expire(
                    task_log_closed_key(task_id), self._log_stream_ttl_sec
                )
        except Exception as exc:
            self._logger.debug(
                "Failed to append log sentinel for task %s: %s", task_id, exc
            )

    def _maybe_close_workflow_log_stream(self, task_id: str) -> None:
        record = self._runtime.get_record(task_id)
        if not record:
            return
        workflow_id = record.workflow_id
        try:
            if self._redis_client.exists(workflow_log_closed_key(workflow_id)):
                return
            remaining = self._redis_client.set_members(workflow_tasks_key(workflow_id))
            if remaining:
                return
            if not self._redis_client.exists(workflow_key(workflow_id)):
                return
        except Exception as exc:
            self._logger.debug(
                "Failed to evaluate workflow completion for %s: %s", workflow_id, exc
            )
            return

        event = LogEvent(
            ts=now_iso(),
            workflow_id=workflow_id,
            level="INFO",
            stream="system",
            source="server",
            message="Workflow log stream closed.",
        )
        payload = event.model_dump(exclude_none=True)
        payload["type"] = "LOG_STREAM_CLOSED"
        encoded = json.dumps(payload, ensure_ascii=False)
        try:
            self._redis_client.xadd_telemetry(
                workflow_log_stream_key(workflow_id),
                {"payload": encoded, "workflow_id": workflow_id},
            )
            self._redis_client.set_value(workflow_log_closed_key(workflow_id), "1")
            if self._log_stream_ttl_sec:
                self._redis_client.expire_telemetry(
                    workflow_log_stream_key(workflow_id), self._log_stream_ttl_sec
                )
                self._redis_client.expire(
                    workflow_log_closed_key(workflow_id), self._log_stream_ttl_sec
                )
        except Exception as exc:
            self._logger.debug(
                "Failed to append log sentinel for workflow %s: %s", workflow_id, exc
            )
