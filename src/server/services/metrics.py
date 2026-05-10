import datetime
import json
import logging
import time
from pathlib import Path
from threading import RLock
from typing import Any

from shared.schemas.event import Event, TaskEvent, WorkerEvent, serialize_event
from shared.tasks.worker_message import WorkerHardware

from ..task.models import categorize_task_type
from ..utils.time import now_iso, parse_iso_ts


def _fresh_timing_bucket() -> dict[str, dict[str, float]]:
    return {
        "queue": {"sum": 0.0, "count": 0},
        "execution": {"sum": 0.0, "count": 0},
        "total": {"sum": 0.0, "count": 0},
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MetricsRecorder:
    """Aggregates task/worker metrics and persists snapshots to disk."""

    def __init__(
        self,
        base_dir: Path,
        logger: logging.Logger,
        *,
        enable_density_plot: bool = False,
        density_bucket_seconds: int = 60,
    ) -> None:
        self._dir = Path(base_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._dir / "metrics.json"
        self._events_log = self._dir / "events.log"
        self._final_report_path = self._dir / "final_report.json"
        self._density_data_path = self._dir / "task_worker_density.json"
        self._logger = logger
        self._lock = RLock()

        self._active_workers: set[str] = set()
        self._counters: dict[str, int] = {
            "tasks_submitted": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "tasks_requeued": 0,
            "workers_registered": 0,
            "workers_unregistered": 0,
            "worker_heartbeats": 0,
        }

        self._task_meta: dict[str, dict[str, Any]] = {}
        self._completed_tasks: set[str] = set()

        self._timings: dict[str, dict[str, dict[str, float]]] = {
            "overall": _fresh_timing_bucket(),
            "inference": _fresh_timing_bucket(),
            "training": _fresh_timing_bucket(),
            "other": _fresh_timing_bucket(),
        }

        self._worker_meta: dict[str, dict[str, Any]] = {}

        self._density_plot_enabled = bool(enable_density_plot)
        self._density_bucket_sec = max(1, int(density_bucket_seconds))
        self._task_density_buckets: dict[int, int] = {}
        self._task_completion_buckets: dict[int, int] = {}
        self._worker_count_series: list[tuple[float, int]] = []
        if self._density_plot_enabled:
            self._logger.info(
                "Task density tracking enabled (bucket=%ds)",
                self._density_bucket_sec,
            )

        self._write_metrics()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_event(self, event: Event) -> None:
        if isinstance(event, TaskEvent):
            self.record_task_event(event)
        elif isinstance(event, WorkerEvent):
            self.record_worker_event(event)

    def record_task_event(self, event: TaskEvent, *, is_child: bool = False) -> None:
        treat_as_child = is_child or self._is_child_task(event)
        with self._lock:
            ev_type = event.type
            if ev_type == "TASK_SUCCEEDED":
                self._counters["tasks_succeeded"] += 1
            elif ev_type == "TASK_FAILED":
                self._counters["tasks_failed"] += 1
            elif ev_type == "TASK_CANCELLED":
                self._counters["tasks_cancelled"] += 1
            elif ev_type == "TASK_REQUEUED":
                self._counters["tasks_requeued"] += 1
            elif ev_type == "TASK_SUBMITTED":
                self._counters["tasks_submitted"] += 1

            meta = self._ensure_task_meta(event)
            if ev_type == "TASK_SUBMITTED":
                self._on_task_submitted(meta, event)
            elif ev_type == "TASK_STARTED":
                self._on_task_started(meta, event)
            elif ev_type == "TASK_SUCCEEDED":
                self._on_task_succeeded(meta, event)
            elif ev_type == "TASK_FAILED":
                self._on_task_failed(meta, event)
            elif ev_type == "TASK_CANCELLED":
                self._on_task_cancelled(meta, event)
            elif ev_type == "TASK_REQUEUED":
                self._on_task_requeued(meta, event)

            if treat_as_child:
                meta["is_child_task"] = True
            self._append_event(event)
        self._write_metrics()

    def _is_child_task(self, event: TaskEvent) -> bool:
        payload = event.payload or {}
        if payload.get("is_child_task") is True:
            return True
        if payload.get("parent_task_id"):
            return True
        shard_index = payload.get("shard_index")
        if shard_index is None:
            shard_index = (
                payload.get("shard", {}).get("index")
                if isinstance(payload.get("shard"), dict)
                else None
            )
        try:
            shard_index = int(shard_index) if shard_index is not None else None
        except (TypeError, ValueError):
            shard_index = None
        return shard_index not in (None, 0)

    def record_worker_event(self, event: WorkerEvent) -> None:
        with self._lock:
            ev_type = event.type
            worker_id = (event.worker_id or "").strip()
            ts_float = parse_iso_ts(event.ts)

            if ev_type == "REGISTER" and worker_id:
                self._active_workers.add(worker_id)
                self._counters["workers_registered"] += 1
                self._on_worker_register(worker_id, event)
                self._record_worker_count(ts_float)
            elif ev_type == "UNREGISTER" and worker_id:
                self._active_workers.discard(worker_id)
                self._counters["workers_unregistered"] += 1
                self._on_worker_unregister(worker_id, event)
                self._record_worker_count(ts_float)
            elif ev_type == "HEARTBEAT":
                self._counters["worker_heartbeats"] += 1
                self._on_worker_heartbeat(worker_id, event)
                self._record_worker_count(ts_float)
            elif ev_type == "STATUS":
                self._on_worker_status(worker_id, event)

            self._append_event(event)
            self._write_metrics()

    def finalize_task_failure(self, task_id: str) -> None:
        self._finalize_task_terminal(task_id, "failed")

    def finalize_task_cancellation(self, task_id: str) -> None:
        self._finalize_task_terminal(task_id, "cancelled")

    def _finalize_task_terminal(self, task_id: str, status: str) -> None:
        with self._lock:
            meta = self._task_meta.get(task_id)
            if not meta or meta.get("finalized"):
                return

            failure = meta.get("pending_failure")
            finished_ts = meta.get("finished_ts")
            if failure:
                queue_time = failure.get("queue")
                exec_time = failure.get("execution")
                total_time = failure.get("total")
            else:
                queue_time, exec_time, total_time = self._compute_durations(
                    meta,
                    finished_ts or time.time(),
                    None,
                )

            self._record_timing(meta, queue_time, exec_time, total_time)

            if queue_time is not None:
                meta["queue_wait_accum"] = queue_time
            if exec_time is not None:
                meta["execution_time_accum"] = exec_time

            meta["finalized"] = True
            meta["pending_failure"] = None
            meta["last_durations"] = {
                "queue": queue_time,
                "execution": exec_time,
                "total": total_time,
                "status": status,
            }
            self._completed_tasks.add(task_id)
            completion_anchor = finished_ts if finished_ts is not None else time.time()
            self._record_task_completion(completion_anchor)
            self._write_metrics()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._build_snapshot()

    def export_final_report(self) -> dict[str, Any]:
        with self._lock:
            report = self._build_snapshot()
            report["finalized_at"] = now_iso()
            density_data_path: str | None = None
            if self._density_plot_enabled and self._density_data_path.exists():
                density_data_path = str(self._density_data_path)
            try:
                self._final_report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                path = str(self._final_report_path)
            except Exception as exc:
                path = ""
                self._logger.warning("Failed to write final metrics report: %s", exc)
            result = {"path": path, "report": report}
            if density_data_path:
                report.setdefault("artifacts", {})[
                    "task_worker_density_data"
                ] = density_data_path
                result["density_data_path"] = density_data_path
            return result

    def finalize_density_series(self) -> None:
        if not self._density_plot_enabled:
            return
        with self._lock:
            bucket_sources = [self._task_density_buckets, self._task_completion_buckets]
            bucket_candidates = [max(store) for store in bucket_sources if store]
            last_bucket_ts = max(bucket_candidates) if bucket_candidates else None
            if self._worker_count_series:
                final_ts = self._worker_count_series[-1][0]
            elif last_bucket_ts is not None:
                final_ts = last_bucket_ts + self._density_bucket_sec
            else:
                final_ts = time.time()
            bucket_ts = (
                int(final_ts // self._density_bucket_sec) * self._density_bucket_sec
            )
            if last_bucket_ts is not None and bucket_ts <= last_bucket_ts:
                bucket_ts = last_bucket_ts + self._density_bucket_sec
            for store in bucket_sources:
                if store.get(bucket_ts) is None:
                    store[bucket_ts] = 0
        self._write_density_snapshot()

    def format_report(self, report: dict[str, Any]) -> str:
        counters = report.get("counters", {})
        timings = report.get("task_timings", {})
        workers = report.get("workers", {})

        def _fmt(avg: float | None) -> str:
            return f"{avg:.2f}s" if isinstance(avg, (int, float)) else "n/a"

        overall = timings.get("overall", {})
        lines = [
            "==== Final Metrics ====",
            f"Tasks: submitted={counters.get('tasks_submitted', 0)}, "
            f"succeeded={counters.get('tasks_succeeded', 0)}, "
            f"failed={counters.get('tasks_failed', 0)}, "
            f"requeued={counters.get('tasks_requeued', 0)}",
            "Average Durations (overall): "
            f"queue={_fmt(overall.get('queue_avg_sec'))}, "
            f"execution={_fmt(overall.get('execution_avg_sec'))}, "
            f"total={_fmt(overall.get('total_avg_sec'))}",
        ]

        totals = workers.get("totals", {})
        lines.append(
            f"Workers: active={report.get('active_workers_count', 0)}, "
            f"total_cost=${totals.get('accrued_cost_usd', 0.0) or 0.0:.2f}"
        )
        avg_submit = report.get("avg_submission_rate_per_min")
        avg_completion = report.get("avg_completion_rate_per_min")

        def _fmt_rate(value: float | None) -> str:
            return f"{value:.2f}/min" if isinstance(value, (int, float)) else "n/a"

        if avg_submit is not None or avg_completion is not None:
            lines.append(
                f"Rates: submit={_fmt_rate(avg_submit)}, "
                f"complete={_fmt_rate(avg_completion)}"
            )
        energy_total = totals.get("estimated_energy_kwh")
        cpu_energy = totals.get("cpu_energy_kwh")
        gpu_energy = totals.get("gpu_energy_kwh")
        reporters = int(totals.get("workers_with_energy", 0) or 0)
        uptime_hours = (totals.get("uptime_sec") or 0.0) / 3600.0

        def _fmt_energy(value: float | None) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.3f}kWh"
            return "n/a"

        lines.append(
            f"Energy: total={_fmt_energy(energy_total)}, "
            f"cpu={_fmt_energy(cpu_energy)}, gpu={_fmt_energy(gpu_energy)}, "
            f"uptime={uptime_hours:.2f}h, reporters={reporters}"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Task helpers
    # ------------------------------------------------------------------ #

    def _ensure_task_meta(self, event: TaskEvent) -> dict[str, Any]:
        task_id = event.task_id
        meta = self._task_meta.get(task_id)
        if not meta:
            meta = {
                "task_type": None,
                "category": "other",
                "submitted_ts": None,
                "last_queue_ts": None,
                "queue_wait_accum": 0.0,
                "execution_time_accum": 0.0,
                "dispatched_ts": None,
                "start_ts": None,
                "finished_ts": None,
                "attempts": 0,
                "pending_failure": None,
                "finalized": False,
                "last_durations": None,
            }
            self._task_meta[task_id] = meta
        else:
            if meta.get("queue_wait_accum") is None:
                meta["queue_wait_accum"] = 0.0
            if meta.get("execution_time_accum") is None:
                meta["execution_time_accum"] = 0.0

        payload = event.payload or {}
        if not meta["task_type"] and payload.get("taskType"):
            meta["task_type"] = payload["taskType"]
            meta["category"] = categorize_task_type(meta["task_type"])
            self._ensure_category(meta["category"])
        return meta

    def _ensure_category(self, category: str) -> None:
        if category not in self._timings:
            self._timings[category] = _fresh_timing_bucket()

    def _on_task_submitted(self, meta: dict[str, Any], event: TaskEvent) -> None:
        ts = parse_iso_ts(event.ts)
        meta["submitted_ts"] = ts
        meta["last_queue_ts"] = ts
        meta["queue_wait_accum"] = 0.0
        meta["execution_time_accum"] = 0.0
        self._record_task_submission(ts)
        payload = event.payload or {}
        task_type = payload.get("taskType")
        if task_type:
            meta["task_type"] = task_type
            meta["category"] = categorize_task_type(task_type)
            self._ensure_category(meta["category"])
        meta["finalized"] = False
        meta["attempts"] = 0

    def _on_task_started(self, meta: dict[str, Any], event: TaskEvent) -> None:
        payload = event.payload or {}
        start_iso = payload.get("started_at") or event.ts
        start_ts = parse_iso_ts(start_iso)
        dispatched_iso = payload.get("dispatched_at")

        meta["start_ts"] = start_ts
        if dispatched_iso:
            meta["dispatched_ts"] = parse_iso_ts(dispatched_iso)
        elif meta["dispatched_ts"] is None:
            meta["dispatched_ts"] = start_ts

        meta["attempts"] = int(meta.get("attempts") or 0) + 1
        meta["pending_failure"] = None

        task_type = payload.get("taskType")
        if task_type and not meta.get("task_type"):
            meta["task_type"] = task_type
            meta["category"] = categorize_task_type(task_type)
            self._ensure_category(meta["category"])

    def _on_task_succeeded(self, meta: dict[str, Any], event: TaskEvent) -> None:
        payload = event.payload or {}
        finish_iso = payload.get("finished_at") or event.ts
        finished_ts = parse_iso_ts(finish_iso)
        meta["finished_ts"] = finished_ts

        runtime_override = _safe_float(payload.get("runtime_sec"))
        queue_time, exec_time, total_time = self._compute_durations(
            meta,
            finished_ts,
            runtime_override,
        )
        if queue_time is not None:
            meta["queue_wait_accum"] = queue_time
        if exec_time is not None:
            meta["execution_time_accum"] = exec_time

        self._record_timing(meta, queue_time, exec_time, total_time)

        meta["finalized"] = True
        meta["last_durations"] = {
            "queue": queue_time,
            "execution": exec_time,
            "total": total_time,
            "status": "succeeded",
        }
        self._completed_tasks.add(event.task_id)
        self._record_task_completion(finished_ts)

    def _on_task_failed(self, meta: dict[str, Any], event: TaskEvent) -> None:
        self._on_task_terminal(meta, event)

    def _on_task_cancelled(self, meta: dict[str, Any], event: TaskEvent) -> None:
        self._on_task_terminal(meta, event)

    def _on_task_terminal(self, meta: dict[str, Any], event: TaskEvent) -> None:
        payload = event.payload or {}
        finish_iso = payload.get("finished_at") or event.ts
        finished_ts = parse_iso_ts(finish_iso)
        meta["finished_ts"] = finished_ts
        runtime_override = _safe_float(payload.get("runtime_sec"))
        queue_time, exec_time, total_time = self._compute_durations(
            meta,
            finished_ts,
            runtime_override,
        )
        if queue_time is not None:
            meta["queue_wait_accum"] = queue_time
        if exec_time is not None:
            meta["execution_time_accum"] = exec_time
        meta["pending_failure"] = {
            "queue": queue_time,
            "execution": exec_time,
            "total": total_time,
        }

    def _on_task_requeued(self, meta: dict[str, Any], event: TaskEvent) -> None:
        ts = parse_iso_ts(event.ts)
        previous_anchor = meta.get("last_queue_ts") or meta.get("submitted_ts")
        if previous_anchor is not None and meta.get("start_ts") is None:
            wait = max(0.0, ts - previous_anchor)
            if wait > 0.0:
                meta["queue_wait_accum"] = (meta.get("queue_wait_accum") or 0.0) + wait
        meta["last_queue_ts"] = ts
        meta["start_ts"] = None
        meta["dispatched_ts"] = None
        meta["pending_failure"] = None

    def _compute_durations(
        self,
        meta: dict[str, Any],
        finished_ts: float,
        runtime_override: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        if finished_ts is None:
            finished_ts = time.time()

        queue_accum = float(meta.get("queue_wait_accum") or 0.0)
        exec_accum = float(meta.get("execution_time_accum") or 0.0)
        queue_anchor = (
            meta.get("last_queue_ts") or meta.get("submitted_ts") or finished_ts
        )
        start_ts = meta.get("start_ts")

        queue_segment = None
        if start_ts is not None and queue_anchor is not None:
            queue_segment = max(0.0, start_ts - queue_anchor)
        elif queue_anchor is not None:
            queue_segment = max(0.0, finished_ts - queue_anchor)

        queue_time = None
        if queue_segment is not None:
            queue_time = queue_accum + queue_segment
        elif queue_accum > 0.0:
            queue_time = queue_accum

        exec_time = None
        exec_segment = None
        if runtime_override is not None:
            exec_segment = max(0.0, runtime_override)
        elif start_ts is not None:
            exec_segment = max(0.0, finished_ts - start_ts)

        if exec_segment is not None:
            exec_time = exec_accum + exec_segment
        elif exec_accum > 0.0:
            exec_time = exec_accum

        total_time = None
        if queue_time is not None and exec_time is not None:
            total_time = queue_time + exec_time
        elif queue_anchor is not None:
            total_time = max(0.0, finished_ts - queue_anchor)

        return queue_time, exec_time, total_time

    def _record_timing(
        self,
        meta: dict[str, Any],
        queue_time: float | None,
        exec_time: float | None,
        total_time: float | None,
    ) -> None:
        category = meta.get("category") or "other"
        self._ensure_category(category)
        for bucket_name in {"overall", category}:
            bucket = self._timings[bucket_name]
            if queue_time is not None:
                bucket["queue"]["sum"] += queue_time
                bucket["queue"]["count"] += 1
            if exec_time is not None:
                bucket["execution"]["sum"] += exec_time
                bucket["execution"]["count"] += 1
            if total_time is not None:
                bucket["total"]["sum"] += total_time
                bucket["total"]["count"] += 1

    # ------------------------------------------------------------------ #
    # Worker helpers
    # ------------------------------------------------------------------ #

    def _normalize_hardware(self, hardware: Any) -> dict[str, Any] | None:
        if isinstance(hardware, WorkerHardware):
            return hardware.model_dump(mode="python")
        if isinstance(hardware, dict):
            return hardware
        return None

    def _on_worker_register(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._worker_meta.setdefault(worker_id, {})
        meta["registered_at"] = event.ts
        meta["last_heartbeat_ts"] = event.ts
        payload = event.payload or {}
        meta["cost_per_hour"] = _safe_float(payload.get("cost_per_hour"))
        hardware = self._normalize_hardware(payload.get("hardware"))
        if hardware is not None:
            meta["hardware"] = hardware
        power = payload.get("power_metrics") or {}
        if isinstance(power, dict):
            meta["power_samples"] = [power]

    def _on_worker_unregister(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._worker_meta.setdefault(worker_id, {})
        meta["unregistered_at"] = event.ts
        meta["last_heartbeat_ts"] = event.ts
        payload = event.payload or {}
        meta["cost_per_hour"] = meta.get("cost_per_hour") or _safe_float(
            payload.get("cost_per_hour")
        )
        meta["uptime_sec"] = _safe_float(payload.get("uptime_sec"))
        meta["accrued_cost_usd"] = _safe_float(payload.get("accrued_cost_usd"))
        summary = payload.get("power_summary")
        if isinstance(summary, dict):
            meta["power_summary"] = summary

    def _on_worker_heartbeat(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._worker_meta.setdefault(worker_id, {})
        meta["last_heartbeat_ts"] = event.ts
        metrics = event.metrics or {}
        uptime = _safe_float(metrics.get("uptime_sec"))
        if uptime is not None:
            meta["uptime_sec"] = uptime
        accrued_cost = _safe_float(metrics.get("accrued_cost_usd"))
        if accrued_cost is not None:
            meta["accrued_cost_usd"] = accrued_cost
        hardware = self._normalize_hardware(metrics.get("hardware"))
        if hardware is not None:
            meta["hardware"] = hardware
        energy_total = _safe_float(metrics.get("estimated_energy_kwh"))
        if energy_total is not None:
            meta["estimated_energy_kwh"] = energy_total
        power_summary = metrics.get("power_summary")
        if isinstance(power_summary, dict):
            meta["power_summary"] = power_summary
        power = metrics.get("power")
        if power and isinstance(power, dict):
            samples = meta.setdefault("power_samples", [])
            samples.append(power)

    def _on_worker_status(self, worker_id: str, event: WorkerEvent) -> None:
        meta = self._worker_meta.setdefault(worker_id, {})
        meta["status"] = event.status

    # ------------------------------------------------------------------ #
    # Density tracking helpers
    # ------------------------------------------------------------------ #

    def _record_task_submission(self, ts: float) -> None:
        if not self._density_plot_enabled:
            return
        bucket = int(ts // self._density_bucket_sec) * self._density_bucket_sec
        self._task_density_buckets[bucket] = (
            self._task_density_buckets.get(bucket, 0) + 1
        )

    def _record_task_completion(self, ts: float) -> None:
        if not self._density_plot_enabled:
            return
        bucket = int(ts // self._density_bucket_sec) * self._density_bucket_sec
        self._task_completion_buckets[bucket] = (
            self._task_completion_buckets.get(bucket, 0) + 1
        )

    def _record_worker_count(self, ts: float) -> None:
        if not self._density_plot_enabled:
            return
        active_workers = len(self._active_workers)
        if self._worker_count_series:
            last_ts, last_value = self._worker_count_series[-1]
            if last_value == active_workers and ts <= last_ts + 1:
                return
        self._worker_count_series.append((ts, active_workers))

    def _serialize_task_buckets(self, buckets: dict[int, int]) -> list[dict[str, Any]]:
        if not self._density_plot_enabled:
            return []
        entries = []
        for bucket_ts in sorted(buckets):
            count = buckets[bucket_ts]
            density = count * (60.0 / self._density_bucket_sec)
            entries.append(
                {
                    "bucket_start": datetime.datetime.fromtimestamp(
                        bucket_ts, tz=datetime.UTC
                    ).isoformat(),
                    "count": count,
                    "tasks_per_minute": density,
                }
            )
        return entries

    def _serialize_task_density(self) -> list[dict[str, Any]]:
        return self._serialize_task_buckets(self._task_density_buckets)

    def _serialize_task_completion_density(self) -> list[dict[str, Any]]:
        return self._serialize_task_buckets(self._task_completion_buckets)

    def _serialize_worker_series(self) -> list[dict[str, Any]]:
        if not self._density_plot_enabled:
            return []
        return [
            {
                "ts": datetime.datetime.fromtimestamp(ts, tz=datetime.UTC).isoformat(),
                "active_workers": count,
            }
            for ts, count in self._worker_count_series
        ]

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def _append_event(self, event: Event) -> None:
        try:
            serialized = serialize_event(event)
            with self._events_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(serialized, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._logger.debug("Failed to log event %s: %s", event, exc)

    def _write_metrics(self) -> None:
        snapshot = self._build_snapshot()
        try:
            self._metrics_path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            self._logger.debug("Failed to write metrics snapshot: %s", exc)
        self._write_density_snapshot()

    def _write_density_snapshot(self) -> None:
        if not self._density_plot_enabled:
            return
        rates = self._compute_density_rates()
        payload = {
            "generated_at": now_iso(),
            "bucket_seconds": self._density_bucket_sec,
            "task_density_buckets": self._serialize_task_density(),
            "task_completion_buckets": self._serialize_task_completion_density(),
            "worker_count_series": self._serialize_worker_series(),
            "avg_submission_rate_per_min": rates.get("submission_per_min"),
            "avg_completion_rate_per_min": rates.get("completion_per_min"),
        }
        try:
            self._density_data_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            self._logger.debug("Failed to write density snapshot: %s", exc)

    def _build_snapshot(self) -> dict[str, Any]:
        counters = dict(self._counters)
        timings = {
            bucket: self._summarize_timing(values)
            for bucket, values in self._timings.items()
        }
        workers = self._summarize_workers()
        snapshot = {
            "generated_at": now_iso(),
            "counters": counters,
            "task_timings": timings,
            "workers": workers,
            "active_workers_count": len(self._active_workers),
            "completed_tasks": sorted(self._completed_tasks),
        }
        if self._density_plot_enabled:
            snapshot["density_bucket_seconds"] = self._density_bucket_sec
            snapshot["task_density_buckets"] = self._serialize_task_density()
            snapshot["task_completion_buckets"] = (
                self._serialize_task_completion_density()
            )
            snapshot["worker_count_series"] = self._serialize_worker_series()
            rates = self._compute_density_rates()
            snapshot["avg_submission_rate_per_min"] = rates.get("submission_per_min")
            snapshot["avg_completion_rate_per_min"] = rates.get("completion_per_min")
            snapshot["density_rates"] = rates
        return snapshot

    def _summarize_timing(self, bucket: dict[str, dict[str, float]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for name, payload in bucket.items():
            count = payload.get("count", 0)
            total = payload.get("sum", 0.0)
            summary[f"{name}_avg_sec"] = (total / count) if count else None
            summary[f"{name}_count"] = count
            summary[f"{name}_total_sec"] = total
        return summary

    def _compute_density_rates(self) -> dict[str, float | None]:
        return {
            "submission_per_min": self._compute_average_rate(
                self._task_density_buckets
            ),
            "completion_per_min": self._compute_average_rate(
                self._task_completion_buckets
            ),
        }

    def _compute_average_rate(self, buckets: dict[int, int]) -> float | None:
        if not self._density_plot_enabled or not buckets:
            return None
        total = sum(buckets.values())
        bucket_count = len(buckets)
        total_seconds = bucket_count * self._density_bucket_sec
        if total_seconds <= 0:
            return None
        return (total / total_seconds) * 60.0

    def _summarize_workers(self) -> dict[str, Any]:
        totals: dict[str, Any] = {
            "accrued_cost_usd": 0.0,
            "uptime_sec": 0.0,
            "estimated_energy_kwh": 0.0,
            "cpu_energy_kwh": 0.0,
            "gpu_energy_kwh": 0.0,
            "workers_with_energy": 0,
            "gpu_memory_total_bytes": 0,
            "gpu_shared_memory_total_bytes": 0,
        }
        detailed = {}
        for worker_id, meta in self._worker_meta.items():
            info = self._prepare_worker_summary(worker_id, meta)
            gpu_mem_bytes = self._extract_gpu_memory_bytes(info)
            gpu_shared_mem_bytes = self._extract_gpu_shared_memory_bytes(info)
            if gpu_mem_bytes is not None:
                totals["gpu_memory_total_bytes"] += gpu_mem_bytes
                info["gpu_memory_total_bytes"] = gpu_mem_bytes
            else:
                info["gpu_memory_total_bytes"] = None
            if gpu_shared_mem_bytes is not None:
                totals["gpu_shared_memory_total_bytes"] += gpu_shared_mem_bytes
                info["gpu_shared_memory_total_bytes"] = gpu_shared_mem_bytes
            else:
                info["gpu_shared_memory_total_bytes"] = None
            breakdown: dict[str, Any] | None = None
            cost = _safe_float(info.get("cost_per_hour"))
            uptime = _safe_float(info.get("uptime_sec"))
            accrued_cost = _safe_float(info.get("accrued_cost_usd"))
            if uptime is not None:
                totals["uptime_sec"] += uptime
            if accrued_cost is not None:
                totals["accrued_cost_usd"] += accrued_cost
            elif cost is not None and uptime is not None:
                totals["accrued_cost_usd"] += (cost / 3600.0) * uptime
            summary = info.get("power_summary")
            if isinstance(summary, dict):
                energy_total = _safe_float(summary.get("estimated_energy_kwh"))
                breakdown = summary.get("estimated_energy_breakdown") or {}
                if energy_total is not None:
                    totals["estimated_energy_kwh"] += energy_total
                    totals["workers_with_energy"] += 1
                    info["estimated_energy_kwh"] = energy_total
                if isinstance(breakdown, dict):
                    cpu_energy = _safe_float(breakdown.get("cpu_kwh"))
                    gpu_energy = _safe_float(breakdown.get("gpu_kwh"))
                    if cpu_energy is not None:
                        totals["cpu_energy_kwh"] += cpu_energy
                    if gpu_energy is not None:
                        totals["gpu_energy_kwh"] += gpu_energy
            info["estimated_energy_breakdown"] = (
                breakdown if isinstance(breakdown, dict) else None
            )
            detailed[worker_id] = info
        if totals["workers_with_energy"] == 0:
            totals["estimated_energy_kwh"] = None
            totals["cpu_energy_kwh"] = None
            totals["gpu_energy_kwh"] = None
        if totals["gpu_memory_total_bytes"] == 0:
            totals["gpu_memory_total_bytes"] = None
        if totals["gpu_shared_memory_total_bytes"] == 0:
            totals["gpu_shared_memory_total_bytes"] = None
        return {
            "totals": totals,
            "workers": detailed,
        }

    def _prepare_worker_summary(
        self, worker_id: str, meta: dict[str, Any]
    ) -> dict[str, Any]:
        info = dict(meta)

        registered_at = info.get("registered_at")
        start_ts: float | None = None
        if registered_at:
            start_ts = parse_iso_ts(registered_at)

        end_ts = None
        unregistered_at = info.get("unregistered_at")
        if unregistered_at:
            end_ts = parse_iso_ts(unregistered_at)
        if end_ts is None:
            last_hb = info.get("last_heartbeat_ts")
            if last_hb:
                end_ts = parse_iso_ts(last_hb)
        if end_ts is None:
            end_ts = time.time()

        if start_ts is not None:
            inferred_uptime = max(0.0, end_ts - start_ts)
            current_uptime = _safe_float(info.get("uptime_sec"))
            if current_uptime is None or inferred_uptime > current_uptime:
                info["uptime_sec"] = inferred_uptime
                meta["uptime_sec"] = inferred_uptime

        uptime = _safe_float(info.get("uptime_sec"))
        cost_per_hour = _safe_float(info.get("cost_per_hour"))
        inferred_cost = None
        if uptime is not None and cost_per_hour is not None:
            inferred_cost = (cost_per_hour / 3600.0) * uptime
        current_cost = _safe_float(info.get("accrued_cost_usd"))
        if inferred_cost is not None and (
            current_cost is None or inferred_cost > current_cost
        ):
            info["accrued_cost_usd"] = inferred_cost
            meta["accrued_cost_usd"] = inferred_cost

        return info

    def _extract_gpu_memory_bytes(self, info: dict[str, Any]) -> int | None:
        hardware_dict = info.get("hardware")
        if not isinstance(hardware_dict, dict):
            return None
        hardware = WorkerHardware.model_validate(hardware_dict)
        gpu_info = hardware.gpu
        total = 0
        found = False
        gpus = gpu_info.gpus
        for entry in gpus:
            mem_bytes = entry.memory_total_bytes
            if mem_bytes is not None:
                if mem_bytes > 0:
                    total += int(mem_bytes)
                    found = True
        if found:
            return total
        return None

    def _extract_gpu_shared_memory_bytes(self, info: dict[str, Any]) -> int | None:
        hardware_dict = info.get("hardware")
        if not isinstance(hardware_dict, dict):
            return None
        hardware = WorkerHardware.model_validate(hardware_dict)
        if not hardware.gpu.memory_is_unified:
            return None
        shared_total = hardware.gpu.shared_memory_total_bytes
        if shared_total is None or shared_total == 0:
            return None
        if shared_total < 0:
            self._logger.debug(
                "Ignoring negative shared GPU memory total: %s", shared_total
            )
            return None
        return int(shared_total)
