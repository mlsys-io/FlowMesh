"""Lightweight power sampling utilities for worker heartbeats."""

import logging
import os
import time
from typing import Any

import pynvml

from shared.utils.time import now_iso

logger = logging.getLogger(__name__)


def _detect_cpu_energy_file() -> str | None:
    """Return the first available RAPL energy counter."""
    candidates = [
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/amd-rapl:0/energy_uj",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None
    for root, _dirs, files in os.walk(base):
        if "energy_uj" in files:
            return os.path.join(root, "energy_uj")
    return None


def _parse_cuda_visible_devices(value: str | None) -> set[int] | None:
    """Parse CUDA_VISIBLE_DEVICES into a set of GPU indices."""
    if value is None:
        return None

    stripped = value.strip()
    if stripped == "":
        return set()

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return set()

    indices: set[int] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered == "nodevfiles":
            return set()
        try:
            indices.add(int(token))
        except ValueError:
            # Non-numeric tokens (e.g. GPU UUIDs/MIG identifiers) are not handled;
            # fall back.
            return None
    return indices


class PowerMonitor:
    """Tracks CPU/GPU power draw samples and aggregates averages."""

    def __init__(self) -> None:
        self._cpu_energy_path = _detect_cpu_energy_file()
        self._cpu_prev_energy: float | None = None
        self._cpu_prev_ts: float | None = None
        self._start_ts = time.time()

        self._cpu_sum = 0.0
        self._cpu_samples = 0
        self._gpu_total_sum = 0.0
        self._gpu_total_samples = 0
        self._per_gpu: dict[str, dict[str, float]] = {}
        self._visible_gpu_indices = _parse_cuda_visible_devices(
            os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        self._nvml_initialized: bool | None = None
        self._nvml_handles: dict[int, Any] = {}

    def sample(self) -> dict[str, Any]:
        """Collect a single power sample."""
        ts = time.time()
        cpu_power = self._read_cpu_power(ts)
        gpu_entries = self._read_gpu_power()
        per_gpu_payload: list[dict[str, Any]] = []
        valid_gpu_totals: list[float] = []

        if cpu_power is not None:
            self._cpu_sum += cpu_power
            self._cpu_samples += 1

        for entry in gpu_entries:
            idx = str(entry["index"])
            power = entry["power_w"]
            per_gpu_payload.append(entry)
            if power is None:
                continue
            valid_gpu_totals.append(power)
            bucket = self._per_gpu.setdefault(idx, {"sum": 0.0, "count": 0})
            bucket["sum"] += power
            bucket["count"] += 1

        gpu_total_value: float | None = None
        if valid_gpu_totals:
            gpu_total_value = sum(valid_gpu_totals)
            self._gpu_total_sum += gpu_total_value
            self._gpu_total_samples += 1

        return {
            "timestamp": now_iso(),
            "cpu_watts": cpu_power,
            "gpu_watts": {
                "total": gpu_total_value,
                "per_gpu": per_gpu_payload,
            },
        }

    def summary(self) -> dict[str, Any]:
        """Return aggregated averages and uptime."""
        uptime_sec = max(0.0, time.time() - self._start_ts)
        avg_cpu = self._cpu_sum / self._cpu_samples if self._cpu_samples else None
        avg_gpu_total = (
            self._gpu_total_sum / self._gpu_total_samples
            if self._gpu_total_samples
            else None
        )
        per_gpu_avg = {
            idx: (stats["sum"] / stats["count"] if stats["count"] else None)
            for idx, stats in self._per_gpu.items()
        }
        hours = uptime_sec / 3600.0 if uptime_sec else 0.0
        cpu_energy_kwh = (
            (avg_cpu * hours / 1000.0) if avg_cpu is not None and hours > 0 else None
        )
        gpu_energy_kwh = (
            (avg_gpu_total * hours / 1000.0)
            if avg_gpu_total is not None and hours > 0
            else None
        )
        total_energy_components = [
            value for value in (cpu_energy_kwh, gpu_energy_kwh) if value is not None
        ]
        total_energy_kwh = (
            sum(total_energy_components) if total_energy_components else None
        )
        return {
            "uptime_sec": uptime_sec,
            "avg_cpu_watts": avg_cpu,
            "avg_gpu_watts": avg_gpu_total,
            "per_gpu_avg_watts": per_gpu_avg,
            "estimated_energy_kwh": total_energy_kwh,
            "estimated_energy_breakdown": {
                "cpu_kwh": cpu_energy_kwh,
                "gpu_kwh": gpu_energy_kwh,
            },
            "samples": {
                "cpu": self._cpu_samples,
                "gpu": self._gpu_total_samples,
            },
        }

    def _read_cpu_power(self, ts: float) -> float | None:
        path = self._cpu_energy_path
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                micro_joules = float(fh.read().strip())
        except (FileNotFoundError, ValueError, OSError):
            return None

        if self._cpu_prev_energy is None:
            self._cpu_prev_energy = micro_joules
            self._cpu_prev_ts = ts
            return None

        delta = micro_joules - self._cpu_prev_energy
        if delta < 0:
            # Counter wrapped; reset baseline.
            self._cpu_prev_energy = micro_joules
            self._cpu_prev_ts = ts
            return None

        prev_ts = self._cpu_prev_ts or ts
        dt = ts - prev_ts
        self._cpu_prev_energy = micro_joules
        self._cpu_prev_ts = ts
        if dt <= 0:
            return None
        watts = (
            delta / 1_000_000.0
        ) / dt  # convert microjoules to joules, then divide by seconds
        return watts

    def _ensure_nvml(self) -> bool:
        if self._nvml_initialized is not None:
            return self._nvml_initialized
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:
            logger.debug("NVML init failed; skipping GPU power sampling: %s", exc)
            self._nvml_initialized = False
            return False
        self._nvml_initialized = True
        return True

    def _read_gpu_power(self) -> list[dict[str, Any]]:
        visible = self._visible_gpu_indices
        if visible is not None and not visible:
            return []

        if not self._ensure_nvml():
            return []

        try:
            count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return []

        entries: list[dict[str, Any]] = []
        for idx in range(count):
            if visible is not None and idx not in visible:
                continue
            handle = self._nvml_handles.get(idx)
            if handle is None:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                except pynvml.NVMLError:
                    continue
                self._nvml_handles[idx] = handle
            try:
                milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
                power: float | None = milliwatts / 1000.0
            except pynvml.NVMLError:
                power = None
            entries.append({"index": idx, "power_w": power})
        return entries
