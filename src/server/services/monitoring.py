import asyncio
import json
import logging
import shutil
import threading
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import Future
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from shared.schemas.event import (
    Event,
    NodeEvent,
    TaskEvent,
    WorkerEvent,
    parse_event,
)
from shared.schemas.worker import WorkerStatus
from shared.utils.manifest import RESULTS_NAME, sync_manifest

from ..auth import default_principal, deregister_resource, register_resource
from ..clients.redis import (
    NODE_EVENT_CHANNEL,
    TASK_EVENT_CHANNEL,
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
    ResourceType,
    UsageRow,
)
from ..registries.node import NodeRegistry
from ..registries.worker import WorkerRegistry
from ..schemas.logs import LogEvent
from ..schemas.result import result_file_path
from ..task.metadata import extract_model_dataset_names
from ..task.models import TaskStatus, TaskUsage
from ..task.runtime import TaskRuntime
from ..utils.logging import log_node_event, log_worker_event
from ..utils.time import now_iso
from .metrics import MetricsRecorder
from .ssh_forward import SshForwardService
from .watchdog import WorkerWatchdog


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
        ssh_forward: SshForwardService | None = None,
        results_dir: Path | str = ".",
        log_stream_ttl_sec: int = 0,
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
        self._ssh_forward = ssh_forward
        self._results_dir = Path(results_dir)
        self._log_stream_ttl_sec = max(0, int(log_stream_ttl_sec))

        self._pending_result_clones: dict[str, list[str]] = {}
        self._pending_lock = threading.RLock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._threads: list[threading.Thread] | None = None
        self._pending_coros: set[Future[Any]] = set()
        self._pending_coros_lock = threading.Lock()

        # Own-node tracking to coordinate with `stop()` during lifespan teardown.
        # Set by `set_own_node()` once the supervisor handshake produces a node_id.
        self._own_node_id: str | None = None
        self._own_node_deregistered: asyncio.Event = asyncio.Event()

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
        event_key = TASK_EVENT_CHANNEL
        while not self._stop_event.is_set():
            try:
                for event in self._get_event_stream(event_key):
                    if isinstance(event, TaskEvent):
                        self._handle_task_event(event)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._logger.warning(
                    "%s listener error: %s; reconnecting...", event_key, exc
                )
                time.sleep(2.0)

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
                self._runtime.mark_updated(event.task_id, payload)
            case "TASK_SUCCEEDED":
                self._unregister_forward_task(event.task_id)
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
                attempts = record.attempts if record else 0
                max_attempts = record.max_attempts if record else None
                can_retry = record and (
                    max_attempts is None or max_attempts < 0 or attempts < max_attempts
                )
                if can_retry:
                    self._unregister_forward_task(event.task_id)
                    limit_display = (
                        "∞"
                        if max_attempts is None or max_attempts < 0
                        else max_attempts
                    )
                    if event.worker_id and record:
                        record.last_failed_worker = event.worker_id
                    self._logger.warning(
                        "Retrying task %s after worker failure (%d/%s)",
                        event.task_id,
                        attempts + 1,
                        limit_display,
                    )
                    self._dispatcher._requeue_task(  # noqa: SLF001
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

                self._unregister_forward_task(event.task_id)
                self._metrics.record_task_event(event)
                impacted, merged_children, usages = self._runtime.mark_failed(
                    event.task_id,
                    event.worker_id,
                    payload,
                    event.ts,
                    error=event.error,
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
                self._unregister_forward_task(event.task_id)
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
                    ResourceType.NODE,
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
                    self._schedule_deregister(ResourceType.NODE, event.node_id, actor)
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
                        ResourceType.WORKER,
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
                        ResourceType.WORKER, worker_id, self._actor_from_event(event)
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
                            self._unregister_forward_task(task_id)
                            record = self._runtime.get_record(task_id)
                            if record and record.status == TaskStatus.CANCELLING:
                                self._runtime.mark_cancelled(task_id, worker_id, {}, ts)
                                self._close_task_log_stream(task_id)
                                self._maybe_close_workflow_log_stream(task_id)
                            else:
                                if record:
                                    record.last_failed_worker = worker_id
                                to_requeue.append(task_id)
                        if to_requeue:
                            self._logger.info(
                                "Requeued %d task(s) after worker %s unregistered: %s",
                                len(to_requeue),
                                worker_id,
                                ", ".join(to_requeue),
                            )
                            for task_id in to_requeue:
                                self._dispatcher._requeue_task(
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
    # SSH task handling
    # ------------------------------------------------------------------ #

    def _handle_ssh_task_update(
        self, task_id: str, worker_id: str | None, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle SSH forward registration for task updates."""
        ssh_payload = payload.get("ssh")
        if not isinstance(ssh_payload, dict):
            return payload
        payload = payload.copy()

        mode = str(ssh_payload.get("mode") or "direct")
        normalized_mode = self._normalize_ssh_mode(mode, worker_id)
        if normalized_mode != mode:
            ssh_payload = ssh_payload.copy()
            ssh_payload["mode"] = normalized_mode
            if normalized_mode == "direct":
                ssh_payload.pop("directHost", None)
                ssh_payload.pop("directPort", None)
                ssh_payload.pop("_relay_target", None)
            payload["ssh"] = ssh_payload

        if normalized_mode != "forward":
            return payload
        assert self._ssh_forward is not None
        assert worker_id is not None
        record = self._runtime.get_record(task_id)

        try:
            ssh_payload = self._ssh_forward.register_forward_task(
                task_id,
                record.workflow_id if record is not None else None,
                worker_id,
                ssh_payload,
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to register SSH forward target for task %s: %s",
                task_id,
                exc,
            )
            return payload

        payload["ssh"] = ssh_payload
        return payload

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
        resource_type: ResourceType,
        resource_id: str,
        principal: PrincipalContext,
        metadata: Mapping[str, Any],
    ) -> None:
        if not RESOURCE_REGISTRARS or not resource_id or self._loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                register_resource(
                    principal, resource_type, resource_id, metadata, self._logger
                ),
                self._loop,
            )
        except RuntimeError as exc:
            self._logger.debug(
                "Failed to schedule %s/%s register: %s",
                resource_type.value,
                resource_id,
                exc,
            )
            return
        self._track_pending(fut)

    def _schedule_deregister(
        self, resource_type: ResourceType, resource_id: str, principal: PrincipalContext
    ) -> None:
        if not RESOURCE_REGISTRARS or not resource_id or self._loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                deregister_resource(
                    principal, resource_type, resource_id, self._logger
                ),
                self._loop,
            )
        except RuntimeError as exc:
            self._logger.debug(
                "Failed to schedule %s/%s deregister: %s",
                resource_type.value,
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
                    principal, ResourceType.NODE, node_id, self._logger
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

    def _unregister_forward_task(self, task_id: str) -> None:
        if self._ssh_forward is None:
            return
        try:
            self._ssh_forward.unregister_task(task_id)
        except Exception as exc:
            self._logger.debug(
                "Failed to unregister SSH forward target for task %s: %s", task_id, exc
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
            if self._ssh_forward is not None:
                return "forward"
            if self._ssh_proxy_enabled:
                return "proxy"
            return "direct"
        return "direct"

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
