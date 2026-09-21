# worker/lifecycle.py
"""Lifecycle manager for the Worker process.

Responsible for registration, periodic heartbeats, transitions between
RUNNING and IDLE, and graceful shutdown/unregister.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from shared.schemas.worker import SSHLimits, WorkerCapabilities
from shared.tasks.worker_message import WorkerHardware, WorkerStatus
from shared.utils.time import now_iso

from .gpu_occupancy import DeviceOccupancy, GpuOccupancyMonitor
from .power import PowerMonitor
from .relay import EndpointRegistry, RelayClient
from .supervisor_client import SupervisorClient

logger = logging.getLogger(__name__)


class Lifecycle:
    def __init__(
        self,
        client: SupervisorClient,
        hb_sec: int,
        hb_ttl_sec: int,
        hb_file: Path,
        cost_per_hour: float,
        power_monitor: PowerMonitor | None = None,
        endpoints: EndpointRegistry | None = None,
        relay_client: RelayClient | None = None,
        gpu_monitor: GpuOccupancyMonitor | None = None,
    ):
        self.client = client
        self.endpoints = endpoints or EndpointRegistry()
        self.relay_client = relay_client
        self.hb_sec = hb_sec
        self.hb_ttl_sec = hb_ttl_sec
        self.hb_file = hb_file
        self.cost_per_hour = cost_per_hour
        self.power_monitor = power_monitor or PowerMonitor()
        self._stop_event = threading.Event()
        self._started_ts: float | None = None
        # _status_lock serialises status sends between the heartbeat thread and the
        # task runner, so neither can overwrite the other's report.
        self._status_lock = threading.Lock()
        self._active_task: str | None = None
        self._reported: WorkerStatus = WorkerStatus.STARTING
        self._last_task_end: float = 0.0
        self._gpu_monitor = gpu_monitor
        self._gpu_executor_probe: Callable[[], bool] | None = None
        if gpu_monitor is not None:
            cfg = gpu_monitor.config
            logger.info(
                "foreign-GPU detection on: a device is reported unavailable above "
                "%d MiB used with nothing of ours loaded (%d consecutive checks, "
                "%.0fs grace after a task)",
                cfg.threshold_mib,
                cfg.consecutive,
                cfg.grace_sec,
            )

    def set_gpu_executor_probe(self, probe: Callable[[], bool]) -> None:
        """Register a probe reporting whether a GPU-using executor is loaded.

        A reading taken while one is warm would include our own model, so the
        monitor must not treat it as another tenant's.
        """
        self._gpu_executor_probe = probe

    def live_gpu_occupancy(self) -> dict[str, DeviceOccupancy]:
        """Per-device occupancy, but only from a reading just taken.

        Empty when the last observation was suppressed or the probe failed, so a
        caller that refuses work on this never refuses on a stale latch.
        """
        monitor = self._gpu_monitor
        if monitor is None or not monitor.measured:
            return {}
        return monitor.snapshot()

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
        if (monitor := self._gpu_monitor) is not None:
            occupancy = {
                uuid: {
                    "unavailable": device.unavailable,
                    "free_bytes": device.free_bytes,
                }
                for uuid, device in monitor.snapshot().items()
            }
            if occupancy:
                metrics["gpu_occupancy"] = occupancy
        return metrics

    def start(
        self,
        env: dict[str, Any],
        hardware: WorkerHardware,
        capabilities: WorkerCapabilities,
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
            capabilities=capabilities,
            ssh_limits=ssh_limits,
            tags=tags,
            cost_per_hour=self.cost_per_hour,
            power_metrics=initial_power,
        )
        self.client.start()
        if self.relay_client is not None:
            self.relay_client.start()
        with self._status_lock:
            self.client.set_status(WorkerStatus.IDLE)
            self._reported = WorkerStatus.IDLE
        self._touch_hb_file()
        threading.Thread(target=self._hb_loop, daemon=True).start()

    def _hb_loop(self):
        while not self._stop_event.is_set():
            # Observe before reporting so the heartbeat carries this beat's
            # reading rather than the previous one's.
            try:
                self._observe_gpu()
            except Exception:
                logger.debug("GPU occupancy observation failed", exc_info=True)
            try:
                self.client.heartbeat(ttl_sec=self.hb_ttl_sec, metrics=self._metrics())
            except Exception:
                pass
            self._touch_hb_file()
            self._stop_event.wait(self.hb_sec)

    def _observe_gpu(self) -> None:
        """Feed the occupancy monitor one observation per heartbeat.

        A reading is only trustworthy when nothing of ours could be in it: no task
        running, no GPU-using executor still warm, and past the grace window in
        which a finished task's subprocess may still be releasing memory. Before
        the runner registers its probe we cannot know, so we do not measure.
        """
        monitor = self._gpu_monitor
        if monitor is None:
            return
        probe = self._gpu_executor_probe
        with self._status_lock:
            idle = self._active_task is None
            past_grace = time.time() - self._last_task_end >= monitor.config.grace_sec
        monitor.observe(idle and past_grace and probe is not None and not probe())

    def set_busy(self, task_id: str):
        with self._status_lock:
            self._active_task = task_id
            try:
                self.client.set_status(WorkerStatus.BUSY, {"task_id": task_id})
                self._reported = WorkerStatus.BUSY
            except Exception:
                pass

    def set_idle(self, task_id: str):
        with self._status_lock:
            self._active_task = None
            self._last_task_end = time.time()
            try:
                self.client.set_status(WorkerStatus.IDLE, {"last_task": task_id})
                self._reported = WorkerStatus.IDLE
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

    def publish_endpoint(self, endpoint_id: str, port: int) -> None:
        self.endpoints.publish(endpoint_id, port)

    def withdraw_endpoint(self, endpoint_id: str) -> None:
        self.endpoints.withdraw(endpoint_id)

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
        if self.relay_client is not None:
            self.relay_client.shutdown()
        self.client.shutdown()
        self._remove_hb_file()

    def _touch_hb_file(self) -> None:
        hb_file = self.hb_file
        hb_file.parent.mkdir(parents=True, exist_ok=True)
        hb_file.touch()

    def _remove_hb_file(self) -> None:
        self.hb_file.unlink(missing_ok=True)
