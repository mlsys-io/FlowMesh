# worker/lifecycle.py
"""Lifecycle manager for the Worker process.

Responsible for registration, periodic heartbeats, transitions between
RUNNING and IDLE, and graceful shutdown/unregister.
"""

import os
import threading
import time
from pathlib import Path
from typing import Any

from shared.schemas.worker import SSHLimits
from shared.tasks.worker_message import WorkerHardware, WorkerStatus
from shared.utils.time import now_iso

from .power import PowerMonitor
from .supervisor_client import SupervisorClient


class Lifecycle:
    def __init__(
        self,
        client: SupervisorClient,
        hb_sec: int,
        hb_ttl_sec: int,
        hb_file: Path,
        cost_per_hour: float,
        power_monitor: PowerMonitor | None = None,
    ):
        self.client = client
        self.hb_sec = hb_sec
        self.hb_ttl_sec = hb_ttl_sec
        self.hb_file = hb_file
        self.cost_per_hour = cost_per_hour
        self.power_monitor = power_monitor or PowerMonitor()
        self._stop_event = threading.Event()
        self._started_ts: float | None = None

    @property
    def worker_id(self) -> str:
        return self.client.worker_id

    def _metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        uptime = None
        if self._started_ts is not None:
            uptime = max(0.0, time.time() - self._started_ts)
            metrics["uptime_sec"] = uptime
            metrics["accrued_cost_usd"] = (self.cost_per_hour / 3600.0) * uptime
        try:
            la = os.getloadavg()
            metrics["loadavg"] = {"1m": la[0], "5m": la[1], "15m": la[2]}
        except Exception:
            pass
        try:
            power_sample = self.power_monitor.sample()
        except Exception:
            power_sample = None
        if power_sample:
            metrics["power"] = power_sample
        try:
            power_summary = self.power_monitor.summary()
        except Exception:
            power_summary = None
        if power_summary:
            metrics["power_summary"] = power_summary
            energy_total = power_summary.get("estimated_energy_kwh")
            if isinstance(energy_total, (int, float)):
                metrics["estimated_energy_kwh"] = energy_total
        return metrics

    def start(
        self,
        env: dict[str, Any],
        hardware: WorkerHardware,
        ssh_limits: SSHLimits | None,
        tags: list[str],
    ):
        self._started_ts = time.time()
        try:
            initial_power = self.power_monitor.sample()
        except Exception:
            initial_power = None
        self.client.register(
            status=WorkerStatus.STARTING,
            started_at=now_iso(),
            pid=os.getpid(),
            env=env,
            hardware=hardware,
            ssh_limits=ssh_limits,
            tags=tags,
            cost_per_hour=self.cost_per_hour,
            power_metrics=initial_power,
        )
        self.client.start()
        self.client.set_status(WorkerStatus.IDLE)
        self._touch_hb_file()
        threading.Thread(target=self._hb_loop, daemon=True).start()

    def _hb_loop(self):
        while not self._stop_event.is_set():
            try:
                self.client.heartbeat(ttl_sec=self.hb_ttl_sec, metrics=self._metrics())
            except Exception:
                pass
            self._touch_hb_file()
            self._stop_event.wait(self.hb_sec)

    def set_busy(self, task_id: str):
        try:
            self.client.set_status(WorkerStatus.BUSY, {"task_id": task_id})
        except Exception:
            pass

    def set_idle(self, task_id: str):
        try:
            self.client.set_status(WorkerStatus.IDLE, {"last_task": task_id})
        except Exception:
            pass

    def set_failed(
        self,
        task_id: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        retryable: bool = True,
    ):
        try:
            self.client.task_failed(
                task_id, error=error, metadata=metadata, retryable=retryable
            )
        except Exception:
            pass

    def set_succeeded(self, task_id: str, metadata: dict[str, Any] | None = None):
        try:
            self.client.task_succeeded(task_id, metadata=metadata)
        except Exception:
            pass

    def set_cancelled(self, task_id: str, metadata: dict[str, Any] | None = None):
        try:
            self.client.task_cancelled(task_id, metadata=metadata)
        except Exception:
            pass

    def notify_task_update(self, task_id: str, payload: dict[str, Any]) -> None:
        try:
            self.client.task_update(task_id, payload)
        except Exception:
            pass

    def notify_task_started(
        self,
        task_id: str,
        task_type: str | None,
        dispatched_at: str | None,
        started_at: str,
    ) -> None:
        try:
            self.client.task_started(
                task_id,
                task_type=task_type,
                dispatched_at=dispatched_at,
                started_at=started_at,
            )
        except Exception:
            pass

    def stop(self) -> None:
        self.client.stop()

    def shutdown(self):
        self._stop_event.set()
        try:
            self.power_monitor.sample()
        except Exception:
            pass
        uptime = None
        if self._started_ts is not None:
            uptime = max(0.0, time.time() - self._started_ts)
        accrued_cost = (
            (self.cost_per_hour / 3600.0) * uptime if uptime is not None else None
        )
        summary = self.power_monitor.summary()
        try:
            self.client.unregister(
                cost_per_hour=self.cost_per_hour,
                uptime_sec=uptime,
                accrued_cost_usd=accrued_cost,
                power_summary=summary,
            )
        except Exception:
            pass
        self.client.shutdown()
        self._remove_hb_file()

    def _touch_hb_file(self) -> None:
        hb_file = self.hb_file
        hb_file.parent.mkdir(parents=True, exist_ok=True)
        hb_file.touch()

    def _remove_hb_file(self) -> None:
        self.hb_file.unlink(missing_ok=True)
