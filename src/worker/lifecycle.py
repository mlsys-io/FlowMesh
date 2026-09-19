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

from .gpu_occupancy import GateConfig, GateState, decide
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
        # Foreign-GPU gate (see gpu_occupancy.py). Off until configure_gpu_gate().
        # _status_lock serialises status sends between the heartbeat thread and
        # the task runner, so a gate flip can never overwrite a BUSY that raced it.
        self._status_lock = threading.Lock()
        self._active_task: str | None = None
        self._reported: WorkerStatus = WorkerStatus.STARTING
        self._last_task_end: float = 0.0
        self._gate_cfg: GateConfig | None = None
        self._gate_probe: Callable[[], float | None] | None = None
        self._gate_state = GateState()

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
            try:
                self.client.heartbeat(ttl_sec=self.hb_ttl_sec, metrics=self._metrics())
            except Exception:
                pass
            self._touch_hb_file()
            try:
                self._evaluate_gpu_gate()
            except Exception:
                logger.debug("foreign-GPU gate evaluation failed", exc_info=True)
            self._stop_event.wait(self.hb_sec)

    def configure_gpu_gate(
        self, cfg: GateConfig, probe: Callable[[], float | None]
    ) -> None:
        """Enable the foreign-GPU gate. Call once, before start(), on GPU workers."""
        if not cfg.enabled:
            return
        self._gate_cfg = cfg
        self._gate_probe = probe
        logger.info(
            "foreign-GPU gate on: UNAVAILABLE when >%d MiB is used with no task "
            "(%d consecutive checks, %.0fs grace after a task)",
            cfg.threshold_mib,
            cfg.consecutive,
            cfg.grace_sec,
        )

    def _evaluate_gpu_gate(self) -> None:
        cfg, probe = self._gate_cfg, self._gate_probe
        if cfg is None or probe is None:
            return
        with self._status_lock:
            if self._active_task is not None or self._reported not in (
                WorkerStatus.IDLE,
                WorkerStatus.UNAVAILABLE,
            ):
                return
            if time.time() - self._last_task_end < cfg.grace_sec:
                return
        used = probe()  # NVML read, outside the lock
        with self._status_lock:
            if self._active_task is not None or self._reported not in (
                WorkerStatus.IDLE,
                WorkerStatus.UNAVAILABLE,
            ):
                return  # a task arrived while we were reading
            prev = self._gate_state
            nxt = decide(used, False, cfg.threshold_mib, cfg.consecutive, prev)
            self._gate_state = nxt
            if nxt.unavailable == prev.unavailable:
                return
            used_mib = round(used or 0.0)
            if nxt.unavailable:
                logger.warning(
                    "foreign GPU occupancy detected: %d MiB used with no active "
                    "task -> UNAVAILABLE",
                    used_mib,
                )
                status = WorkerStatus.UNAVAILABLE
                payload = {"reason": "foreign_gpu_occupancy", "gpu_used_mib": used_mib}
            else:
                logger.info(
                    "foreign GPU occupancy cleared (%d MiB used) -> IDLE", used_mib
                )
                status = WorkerStatus.IDLE
                payload = {"reason": "foreign_gpu_released", "gpu_used_mib": used_mib}
            try:
                self.client.set_status(status, payload)
                self._reported = status
            except Exception:
                # Could not tell the supervisor; retry on the next heartbeat.
                self._gate_state = prev

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
            # Back to IDLE; the gate re-checks after its grace period.
            self._gate_state = GateState()
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
