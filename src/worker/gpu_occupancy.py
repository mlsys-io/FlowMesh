"""Detect GPU memory held by processes outside this worker.

A worker's GPU can be claimed by things FlowMesh cannot see: a Kubernetes pod
handed the same card by the NVIDIA device plugin, a bare ``vllm serve``
container, an ad-hoc training script. The worker kept reporting ``IDLE``
throughout, so the dispatcher sent it tasks that then failed to fit (on
2026-09-19 a sandbox pod held 44.7 GiB of a 48 GiB card while its worker
advertised itself as free).

Signal: device memory in use while this worker runs no task. NVML's
per-process list is not usable for attribution here — inside a container it
reports host PIDs that cannot be mapped back to the worker's own processes —
but the worker tears its executors down after every task (``mp_executor``
runs one subprocess per task), so an idle worker's own footprint is only a
CUDA context at most. Anything well above that belongs to someone else.

The check fails OPEN: if NVML cannot be read, the worker keeps its current
status rather than taking itself out of the pool.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIB = 1024 * 1024


@dataclass(frozen=True)
class GateConfig:
    """Tuning for the foreign-occupancy gate.

    ``threshold_mib`` sits above an idle CUDA context (a few hundred MiB) and
    below anything that can actually serve or train. ``consecutive`` checks
    must agree before the state flips, in either direction, so a transient
    allocation or a slow teardown does not make the worker flap.
    ``grace_sec`` skips checks right after this worker's own task ends, while
    its executor subprocess may still be releasing memory.
    """

    enabled: bool = True
    threshold_mib: int = 1024
    consecutive: int = 2
    grace_sec: float = 90.0


@dataclass(frozen=True)
class GateState:
    unavailable: bool = False
    streak: int = 0


def decide(
    used_mib: float | None,
    has_active_task: bool,
    threshold_mib: int,
    consecutive: int,
    previous: GateState,
) -> GateState:
    """Next gate state from one observation. Pure, for testing.

    ``used_mib`` is the largest per-device usage across this worker's GPUs, or
    ``None`` when it could not be read — which never changes the state.
    While a task is active the reading is this worker's own, so the gate
    neither enters nor leaves ``unavailable`` on it; the streak resets.
    """
    if used_mib is None or has_active_task:
        return GateState(unavailable=previous.unavailable, streak=0)
    occupied = used_mib > threshold_mib
    if occupied == previous.unavailable:
        # Observation agrees with the current state: nothing pending.
        return GateState(unavailable=previous.unavailable, streak=0)
    streak = previous.streak + 1
    if streak >= max(1, consecutive):
        return GateState(unavailable=occupied, streak=0)
    return GateState(unavailable=previous.unavailable, streak=streak)


class NvmlUsageProbe:
    """Largest per-device memory usage (MiB) across the GPUs this process sees.

    A worker container is only given its own GPU(s), so every device NVML
    enumerates here belongs to this worker. Unified-memory devices (e.g.
    GB10) are skipped: their "used" figure is system RAM, not a card
    another tenant is holding.
    """

    def __init__(self, is_unified: Callable[[int, str], bool] | None = None) -> None:
        self._is_unified = is_unified
        self._warned = False
        self._initialised = False

    def __call__(self) -> float | None:
        try:
            import pynvml

            if not self._initialised:
                pynvml.nvmlInit()
                self._initialised = True
            peak: float | None = None
            for idx in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                if self._is_unified is not None:
                    name_raw = pynvml.nvmlDeviceGetName(handle)
                    name = (
                        name_raw.decode() if isinstance(name_raw, bytes) else name_raw
                    )
                    if self._is_unified(idx, name):
                        continue
                used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / MIB
                peak = used if peak is None else max(peak, used)
            return peak
        except Exception as exc:  # NVML missing, driver wedged, no GPU
            if not self._warned:
                logger.warning(
                    "foreign-GPU gate: cannot read GPU memory (%s); "
                    "keeping current status (fail open)",
                    exc,
                )
                self._warned = True
            return None
