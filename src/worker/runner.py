"""Task runner that executes assignments relayed by the server."""

import json
import logging
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

from shared.tasks import MergedChildTaskStrict
from shared.tasks.specs import InferenceSpecStrict, TaskSpecStrictBase
from shared.tasks.worker_message import (
    HardwareUsage,
    WorkerHardware,
    WorkerTaskMessage,
)
from shared.utils.manifest import prepare_output_dir, sync_manifest
from shared.utils.time import now_iso

from .executors.base_executor import Executor, TaskCancelledError
from .executors.utils.checkpoints import (
    get_http_destination,
    write_executor_result,
)
from .lifecycle import Lifecycle
from .utils.logging import TaskLogEmitter


class Runner:
    def __init__(
        self,
        lifecycle: Lifecycle,
        task_stream: Iterable[WorkerTaskMessage],
        results_dir: Path,
        hardware: WorkerHardware,
        executors: dict[str, Executor],
        default_executor: Executor,
        logger: logging.Logger,
        network_bandwidth_bytes_per_sec: float | None = None,
        executor_idle_cleanup_sec: float | None = None,
    ):
        self.lifecycle = lifecycle
        self.task_stream = task_stream
        self.results_dir = results_dir
        self.hardware = hardware
        self.executors = executors
        self.logger = logger
        self.default_executor = default_executor
        self.network_bandwidth_bytes_per_sec = network_bandwidth_bytes_per_sec
        # How long to keep an executor alive (seconds) after its last use
        # before calling `cleanup_after_run()`. None or <=0 disables delayed cleanup.
        assert (
            executor_idle_cleanup_sec is None or executor_idle_cleanup_sec >= 0
        ), "executor_idle_cleanup_sec must be None or non-negative"
        self.executor_idle_cleanup_sec = executor_idle_cleanup_sec

        # Track a single active executor instance for reuse and its metadata
        self._active_executor: Executor | None = None
        self._active_executor_key: str | None = None
        self._active_executor_last_used_at: float | None = None
        # Lock to protect concurrent access to active executor state
        self._active_executor_lock = threading.Lock()

        # Background idle checker thread and control event
        self._idle_checker_thread: threading.Thread | None = None
        self._idle_checker_stop_event: threading.Event | None = None
        # Poll interval in seconds for the idle checker
        self._idle_check_interval: float = 1.0

        # Interrupt monitoring thread and control event
        self._interrupt_thread: threading.Thread | None = None
        self._interrupt_stop_event: threading.Event | None = None
        self._current_task_id: str | None = None
        self._pending_cancels: set[str] = set()
        self._pending_stops: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._shutdown_requested = threading.Event()

    def _cancel_active_executor(self) -> None:
        with self._active_executor_lock:
            executor = self._active_executor
            if executor is None:
                return
            current_task_id = self._current_task_id
            if current_task_id is not None:
                try:
                    executor.cancel(current_task_id)
                except Exception as exc:
                    self.logger.debug(
                        "Error cancelling active executor during shutdown: %s", exc
                    )

    def _cleanup_active_executor(self) -> None:
        self._cancel_active_executor()
        with self._active_executor_lock:
            executor = self._active_executor
            if executor is None:
                return
            try:
                executor.cleanup_after_run()
            except Exception as exc:
                self.logger.debug(
                    "Error cleaning up active executor during shutdown: %s", exc
                )
            finally:
                self._active_executor = None
                self._active_executor_key = None
                self._active_executor_last_used_at = None

    def stop(self) -> None:
        self._shutdown_requested.set()
        self.lifecycle.stop()
        self._cancel_active_executor()

    def _resolve_output_dir(self, task_id: str) -> Path:
        """Prepare and return the canonical output directory for a task's results."""
        out_dir = self.results_dir / task_id
        prepare_output_dir(out_dir)
        return out_dir

    def _write_results(
        self,
        task_id: str,
        spec: TaskSpecStrictBase,
        merged_children: list[MergedChildTaskStrict],
        out_dir: Path,
        result: dict[str, Any] | None,
    ):
        if result is None:
            return
        self._write_single_result(task_id, spec, out_dir, result)

        child_lookup = {entry.task_id: entry for entry in merged_children}
        children_payload = (
            result.get("children") if isinstance(result, dict) else {}
        ) or {}
        for child_id, child_result in children_payload.items():
            child_info = child_lookup.get(child_id)
            if child_info is None:
                continue
            child_out_dir = self._resolve_output_dir(child_id)
            self._write_single_result(
                child_id, child_info.spec, child_out_dir, child_result
            )

    def _write_single_result(
        self,
        task_id: str,
        spec: TaskSpecStrictBase,
        out_dir: Path,
        payload: dict[str, Any] | None,
    ):
        if payload is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        write_executor_result(out_dir / "results.json", task_id, spec, payload)
        sync_manifest(out_dir, task_id, spec.get_artifacts())
        self._maybe_emit_http(task_id, spec, payload)

    def _simulate_bandwidth_delay(self, payload_bytes: int, destination: str) -> None:
        if not self.network_bandwidth_bytes_per_sec:
            return
        if payload_bytes <= 0:
            return
        delay = payload_bytes / self.network_bandwidth_bytes_per_sec
        if delay <= 0:
            return
        self.logger.debug(
            "HTTP delivery to %s throttled for %.3f sec (payload=%d bytes, "
            "bandwidth=%.0f B/s)",
            destination,
            delay,
            payload_bytes,
            self.network_bandwidth_bytes_per_sec,
        )
        time.sleep(delay)

    def _maybe_emit_http(
        self, task_id: str, spec: TaskSpecStrictBase, result: dict[str, Any]
    ) -> None:
        """Send task results to an HTTP endpoint when requested by the spec."""
        destination = get_http_destination(spec)
        if destination is None:
            return

        url = destination.url
        ignore_error = destination.ignore_error
        payload = {
            "task_id": task_id,
            "result": result,
            "worker_id": self.lifecycle.worker_id,
        }
        payload_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self._simulate_bandwidth_delay(payload_size, destination=url)

        try:
            response = requests.request(
                destination.method,
                url,
                json=payload,
                headers=destination.headers,
                timeout=destination.timeout,
            )
        except requests.RequestException as exc:
            if ignore_error:
                self.logger.warning(
                    "Task %s failed to upload results to %s (%s); result still on disk",
                    task_id,
                    url,
                    exc,
                )
                return
            raise RuntimeError(
                f"Failed to deliver task {task_id} result to {url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            snippet = response.text[:200]
            if ignore_error:
                self.logger.warning(
                    "Task %s upload to %s returned %s: %s; result still on disk",
                    task_id,
                    url,
                    response.status_code,
                    snippet,
                )
                return
            raise RuntimeError(
                f"HTTP delivery for task {task_id} returned status "
                f"{response.status_code}: {snippet}"
            )

        self.logger.info(
            "Task %s result delivered to %s (%s)", task_id, url, response.status_code
        )

    def _select_inference_executor_key(self, spec: InferenceSpecStrict) -> str:
        enforce_cpu = spec.enforce_cpu
        if enforce_cpu:
            return "default"

        model_cfg = spec.model
        if model_cfg is not None and model_cfg.has_lora_adapter():
            return "vllm_lora"
        adapters = model_cfg.adapters if model_cfg else None
        if adapters:
            return "vllm_lora"
        if model_cfg is not None and model_cfg.transformers and not model_cfg.vllm:
            return "default"
        return "vllm"

    def _maybe_expire_active_executor(self) -> None:
        """Expire the active executor if it has been idle past the configured
        timeout.
        """
        with self._active_executor_lock:
            if not self._active_executor:
                return
            if not self.executor_idle_cleanup_sec:
                return
            if not self._active_executor_last_used_at:
                return
            idle = time.time() - self._active_executor_last_used_at
            if idle >= self.executor_idle_cleanup_sec:
                self.logger.info(
                    "Executor %s idle for %.1f sec, calling cleanup_after_run()",
                    self._active_executor_key,
                    idle,
                )
                self._active_executor.cleanup_after_run()
                self._active_executor = None
                self._active_executor_key = None
                self._active_executor_last_used_at = None

    def _idle_check_loop(self, stop_event: threading.Event) -> None:
        """Background loop that periodically checks for idle executors.

        The loop waits on `stop_event` with a timeout equal to
        `self._idle_check_interval` and calls `_maybe_expire_active_executor`
        each tick.
        """
        try:
            while not stop_event.wait(self._idle_check_interval):
                try:
                    self._maybe_expire_active_executor()
                except Exception as exc:
                    self.logger.debug("Idle checker encountered error: %s", exc)
        except Exception:
            # Ensure background thread doesn't propagate exceptions
            pass

    def _start_idle_checker(self) -> None:
        """Start the idle checker background thread if not already running.

        Only starts if `executor_idle_cleanup_sec` is configured and positive.
        """
        if not self.executor_idle_cleanup_sec or self.executor_idle_cleanup_sec <= 0:
            return
        if self._idle_checker_thread and self._idle_checker_thread.is_alive():
            return
        self._idle_checker_stop_event = threading.Event()
        # Use a small poll interval; ensure it's not larger than the cleanup
        # timeout so expiration happens reasonably soon after timed out.
        self._idle_check_interval = min(
            1.0, max(0.5, float(self.executor_idle_cleanup_sec) / 10.0)
        )
        t = threading.Thread(
            target=self._idle_check_loop, args=(self._idle_checker_stop_event,)
        )
        t.daemon = True
        t.name = "flowmesh-idle-checker"
        self._idle_checker_thread = t
        t.start()

    def _stop_idle_checker(self) -> None:
        """Signal the idle checker thread to stop and wait briefly for join."""
        if not self._idle_checker_thread:
            return
        if self._idle_checker_stop_event:
            self._idle_checker_stop_event.set()
        try:
            self._idle_checker_thread.join(timeout=2.0)
        except Exception:
            pass
        finally:
            self._idle_checker_thread = None
            self._idle_checker_stop_event = None

    def _interrupt_monitor_loop(self, stop_event: threading.Event) -> None:
        try:
            while not stop_event.wait(0.5):
                try:
                    for task_id, reason in self.lifecycle.client.iter_interrupts():
                        with self._cancel_lock:
                            self._pending_cancels.add(task_id)
                        if self._current_task_id != task_id:
                            continue
                        self.logger.info(
                            "Interrupt for running task %s (reason=%s)", task_id, reason
                        )
                        with self._active_executor_lock:
                            executor = self._active_executor
                        if executor is not None:
                            try:
                                executor.cancel(task_id)
                            except Exception as exc:
                                self.logger.warning("Executor cancel() raised: %s", exc)
                    for task_id, reason in self.lifecycle.client.iter_stops():
                        if self._current_task_id != task_id:
                            with self._cancel_lock:
                                self._pending_stops.add(task_id)
                            continue
                        self.logger.info(
                            "Graceful stop for running task %s (reason=%s)",
                            task_id,
                            reason,
                        )
                        with self._active_executor_lock:
                            executor = self._active_executor
                        if executor is not None:
                            try:
                                executor.stop(task_id)
                            except Exception as exc:
                                self.logger.warning("Executor stop() raised: %s", exc)
                except Exception as exc:
                    self.logger.warning("Interrupt monitor encountered error: %s", exc)
        except Exception:
            pass

    def _start_interrupt_monitor(self) -> None:
        if self._interrupt_thread and self._interrupt_thread.is_alive():
            return
        self._interrupt_stop_event = threading.Event()
        thread = threading.Thread(
            target=self._interrupt_monitor_loop,
            args=(self._interrupt_stop_event,),
            daemon=True,
            name="flowmesh-interrupt-monitor",
        )
        self._interrupt_thread = thread
        thread.start()

    def _stop_interrupt_monitor(self) -> None:
        if not self._interrupt_thread:
            return
        if self._interrupt_stop_event is not None:
            self._interrupt_stop_event.set()
        try:
            self._interrupt_thread.join(timeout=2.0)
        except Exception:
            pass
        finally:
            self._interrupt_thread = None
            self._interrupt_stop_event = None

    def start(self) -> None:
        self._start_idle_checker()
        self._start_interrupt_monitor()
        try:
            for msg in self.task_stream:
                if self._shutdown_requested.is_set():
                    break
                assigned_worker = msg.assigned_worker
                if assigned_worker and assigned_worker != self.lifecycle.worker_id:
                    self.logger.info(
                        "Skipping task %s assigned to %s (this worker id=%s)",
                        msg.task_id,
                        assigned_worker,
                        self.lifecycle.worker_id,
                    )
                    continue

                task_id = msg.task_id
                spec = msg.spec
                task_type = spec.taskType
                merged_children = msg.merged_children or []

                parent_task_id = msg.parent_task_id
                shard_index = msg.shard_index
                shard_total = msg.shard_total
                dispatched_at = msg.dispatched_at

                extra = []
                if parent_task_id:
                    extra.append(f"parent={parent_task_id}")
                if shard_index is not None and shard_total is not None:
                    extra.append(f"shard={shard_index}/{shard_total}")
                extra_info = ", ".join(extra) if extra else "no-parent"

                self.logger.info(
                    "Received task %s (type=%s, %s) with spec keys %s",
                    task_id,
                    task_type or "unknown",
                    extra_info,
                    sorted(key for key, value in spec if value is not None),
                )

                task_log_emitter: TaskLogEmitter | None = None
                log_handler_attached: bool = False
                prev_root_log_level: int | None = None
                out_dir = self._resolve_output_dir(task_id)
                self.lifecycle.set_busy(task_id)
                # Set task start time in case of early failure
                start_iso = now_iso()
                start_wall = time.time()
                notified_task_started: bool = False
                try:
                    with self._cancel_lock:
                        cancelled_before_start = task_id in self._pending_cancels
                        if cancelled_before_start:
                            self._pending_cancels.discard(task_id)
                        stop_before_start = task_id in self._pending_stops
                        if stop_before_start:
                            self._pending_stops.discard(task_id)
                    if cancelled_before_start:
                        raise TaskCancelledError(
                            f"Task {task_id} was cancelled before execution"
                        )
                    self._current_task_id = task_id
                    desired_key: str
                    if task_type == "inference":
                        assert isinstance(spec, InferenceSpecStrict)
                        desired_key = self._select_inference_executor_key(spec)
                    elif task_type == "diffusion":
                        desired_key = "diffusers"
                    elif task_type == "embedding":
                        desired_key = "default"
                    else:
                        desired_key = "default" if task_type is None else task_type
                    # Acquire lock before accessing/modifying active executor
                    with self._active_executor_lock:
                        if (
                            self._active_executor
                            and self._active_executor_key != desired_key
                        ):
                            self.logger.info(
                                "New task requests executor %s, shutting down active "
                                "executor %s",
                                desired_key,
                                self._active_executor_key,
                            )
                            self._active_executor.cleanup_after_run()
                            self._active_executor = None
                            self._active_executor_key = None

                        if not self._active_executor:
                            self._active_executor = self.executors.get(
                                desired_key, self.default_executor
                            )
                            self._active_executor_key = desired_key

                        (
                            task_log_emitter,
                            log_handler_attached,
                            prev_root_log_level,
                        ) = self._create_task_logger(task_id, msg, out_dir)

                        # Notify task started just before execution
                        start_iso = now_iso()
                        start_wall = time.time()
                        self.lifecycle.notify_task_started(
                            task_id,
                            task_type=task_type,
                            dispatched_at=dispatched_at,
                            started_at=start_iso,
                        )
                        notified_task_started = True

                        # Disable idle checker during execution
                        self._active_executor_last_used_at = None
                        executor_to_run = self._active_executor
                        if stop_before_start:
                            executor_to_run.stop(task_id)
                    out = executor_to_run.run(msg, out_dir)
                    self._write_results(task_id, spec, merged_children, out_dir, out)
                    metadata = self._build_task_metadata(
                        task_type,
                        dispatched_at,
                        start_iso,
                        start_wall,
                        shard_index=shard_index,
                        shard_total=shard_total,
                    )
                    self.lifecycle.set_succeeded(task_id, metadata=metadata)
                    self.logger.info("Task %s completed successfully", task_id)
                except TaskCancelledError as e:
                    if not notified_task_started:
                        self.lifecycle.notify_task_started(
                            task_id,
                            task_type=task_type,
                            dispatched_at=dispatched_at,
                            started_at=start_iso,
                        )
                        notified_task_started = True
                    metadata = self._build_task_metadata(
                        task_type,
                        dispatched_at,
                        start_iso,
                        start_wall,
                        shard_index=shard_index,
                        shard_total=shard_total,
                    )
                    self.lifecycle.set_cancelled(task_id, metadata=metadata)
                    self.logger.info("Task %s cancelled: %s", task_id, e)
                except Exception as e:
                    if not notified_task_started:
                        self.lifecycle.notify_task_started(
                            task_id,
                            task_type=task_type,
                            dispatched_at=dispatched_at,
                            started_at=start_iso,
                        )
                        notified_task_started = True
                    metadata = self._build_task_metadata(
                        task_type,
                        dispatched_at,
                        start_iso,
                        start_wall,
                        shard_index=shard_index,
                        shard_total=shard_total,
                    )
                    self.lifecycle.set_failed(task_id, str(e), metadata=metadata)
                    self.logger.exception("Task %s failed", task_id)
                finally:
                    self._current_task_id = None
                    with self._active_executor_lock:
                        self._active_executor_last_used_at = time.time()
                    self.lifecycle.set_idle(task_id)
                    if task_log_emitter is not None:
                        if log_handler_attached:
                            logging.getLogger().removeHandler(task_log_emitter)
                            self.logger.removeHandler(task_log_emitter)
                        try:
                            task_log_emitter.close()
                        except Exception:
                            pass
                    if prev_root_log_level is not None:
                        logging.getLogger().setLevel(prev_root_log_level)
        except KeyboardInterrupt:
            self.logger.info("Runner interrupted by user; shutting down task loop")
        finally:
            self._cleanup_active_executor()
            self._stop_interrupt_monitor()
            self._stop_idle_checker()

    def _create_task_logger(
        self, task_id: str, msg: WorkerTaskMessage, out_dir: Path
    ) -> tuple[TaskLogEmitter | None, bool, int | None]:
        task_log_emitter: TaskLogEmitter | None = None
        log_handler_attached = False
        prev_root_log_level: int | None = None
        try:
            owner_mismatch = False
            log_paths: dict[str, Path] = {task_id: self._get_log_path(out_dir)}
            task_refs: list[dict[str, str]] = [
                {"task_id": task_id, "workflow_id": msg.workflow_id}
            ]
            if msg.merged_children:
                for entry in msg.merged_children:
                    child_id = entry.task_id
                    child_workflow_id = entry.workflow_id or msg.workflow_id
                    task_refs.append(
                        {"task_id": child_id, "workflow_id": child_workflow_id}
                    )
                    child_out_dir = self._resolve_output_dir(child_id)
                    log_paths[child_id] = self._get_log_path(child_out_dir)
                    if entry.owner_id != msg.owner_id:
                        owner_mismatch = True
                        continue

            if owner_mismatch:
                task_log_emitter = self.lifecycle.client.create_task_log_emitter(
                    task_id=task_id,
                    workflow_id=msg.workflow_id,
                    owner_id=msg.owner_id,
                    task_refs=task_refs,
                    log_paths=log_paths,
                )
                if task_log_emitter is not None:
                    task_log_emitter.emit_warning_only(
                        "Task-level logs are not supported for merged tasks "
                        "with different owners."
                    )
                else:
                    self.logger.warning(
                        "Task-level logs are not supported for merged tasks "
                        "with different owners."
                    )
                task_log_emitter = None
            else:
                task_log_emitter = self.lifecycle.client.create_task_log_emitter(
                    task_id=task_id,
                    workflow_id=msg.workflow_id,
                    owner_id=msg.owner_id,
                    task_refs=task_refs,
                    log_paths=log_paths,
                )
                if task_log_emitter is not None:
                    root_logger = logging.getLogger()
                    root_logger.addHandler(task_log_emitter)
                    self.logger.addHandler(task_log_emitter)
                    log_handler_attached = True
                    prev_root_log_level = root_logger.level
                    desired_level = self.logger.level
                    if prev_root_log_level > desired_level:
                        root_logger.setLevel(desired_level)
        except Exception:
            task_log_emitter = None
            log_handler_attached = False
            prev_root_log_level = None

        return task_log_emitter, log_handler_attached, prev_root_log_level

    @staticmethod
    def _get_log_path(out_dir: Path) -> Path:
        return out_dir / "logs" / "logs.jsonl"

    def _build_task_metadata(
        self,
        task_type: str | None,
        dispatched_at: str | None,
        started_at: str,
        start_wall: float,
        shard_index: int | None = None,
        shard_total: int | None = None,
    ) -> dict[str, Any]:
        finished_at = now_iso()
        runtime_sec = max(0.0, time.time() - start_wall)
        hw_usage = HardwareUsage.from_hw(self.hardware).model_dump()
        cost_per_hour = self.lifecycle.cost_per_hour
        metadata = {
            "taskType": task_type,
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_sec": runtime_sec,
            "hardware": hw_usage,
            "cost_per_hour": cost_per_hour,
            "total_cost": (cost_per_hour / 3600.0) * runtime_sec,
        }
        if dispatched_at:
            metadata["dispatched_at"] = dispatched_at
        if shard_index is not None:
            metadata["shard_index"] = shard_index
        if shard_total is not None:
            metadata["shard_total"] = shard_total
        return metadata
