"""Detect GPU memory held by processes outside this worker, per device.

A worker's GPU can be claimed by a process FlowMesh cannot see: a Kubernetes
pod handed the same card by the NVIDIA device plugin, a bare ``vllm serve``
container, an ad-hoc training script. The worker keeps running CPU work either
way, so occupancy is reported as a per-device resource fact rather than a
worker status, and the dispatcher skips only the held devices.

Signal: device memory in use while the worker runs nothing of its own —
neither a task nor a warm GPU executor, which stays resident between tasks and
may still hold VRAM. The caller decides when that holds (see
``Lifecycle._observe_gpu``) and passes it as ``measurable``; any usage seen then
belongs to another tenant. NVML's per-process list cannot attribute memory here
— inside a container it reports host PIDs that do not map back to the worker's
own processes — hence the coarse per-device signal.

Two kinds of "cannot read" are deliberately distinguished:

* **Suppressed** — a task is running, a GPU executor is warm, or we are inside
  the post-task grace window. The last reading latches, because the device is
  very likely still in whatever state we last saw it in.
* **Probe failure** — NVML is unreadable. The reading clears to "no opinion".
  Latching here would be a trap: only a fresh clear reading releases a latch,
  and a broken probe never produces one, so a worker would be excluded from GPU
  placement forever.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pynvml

logger = logging.getLogger(__name__)

MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuGateConfig:
    """Tuning for foreign-occupancy detection.

    ``threshold_mib`` sits above an idle CUDA context (a few hundred MiB) and below
    anything that can actually serve or train. ``consecutive`` checks must agree before
    a device's state flips, in either direction, so a transient allocation or a slow
    teardown does not make it flap. ``grace_sec`` suppresses readings right after this
    worker's own task ends, while its executor subprocess may still be releasing memory.
    """

    enabled: bool = True
    threshold_mib: int = 1024
    consecutive: int = 2
    grace_sec: float = 90.0


@dataclass(frozen=True)
class DeviceReading:
    """One device's memory as NVML reported it."""

    used_mib: float
    free_bytes: int


@dataclass(frozen=True)
class DeviceState:
    """Hysteresis state for one device."""

    unavailable: bool = False
    used_mib: float | None = None
    streak: int = 0


@dataclass(frozen=True)
class DeviceOccupancy:
    """What the worker reports about one device."""

    unavailable: bool
    free_bytes: int | None


def decide(
    used_mib: float,
    threshold_mib: int,
    consecutive: int,
    previous: DeviceState,
) -> DeviceState:
    """Next state for one device from one observation. Pure, for testing.

    Only called for a device that was actually read; a device the probe could not
    read keeps its previous state by not being passed here at all.
    """
    occupied = used_mib > threshold_mib
    if occupied == previous.unavailable:
        # Observation agrees with the current state: nothing pending.
        return DeviceState(
            unavailable=previous.unavailable, used_mib=used_mib, streak=0
        )
    streak = previous.streak + 1
    if streak >= max(1, consecutive):
        return DeviceState(unavailable=occupied, used_mib=used_mib, streak=0)
    return DeviceState(
        unavailable=previous.unavailable, used_mib=used_mib, streak=streak
    )


class NvmlDeviceProbe:
    """Per-UUID memory readings for the GPUs this process can see.

    A worker container is only given its own GPU(s), so every device NVML enumerates
    here belongs to this worker. Unified-memory devices (e.g. GB10) are omitted: their
    "used" figure is system RAM, not a card another tenant is holding.

    Returns ``{}`` when NVML itself cannot be reached, which the monitor reads as total
    probe failure. A device that individually fails to read is simply absent from the
    result, leaving whatever the monitor last knew about it untouched.
    """

    def __init__(self, is_unified: Callable[[int, str], bool] | None = None) -> None:
        self._is_unified = is_unified
        self._warned = False
        self._initialised = False

    def __call__(self) -> dict[str, DeviceReading]:
        try:
            if not self._initialised:
                pynvml.nvmlInit()
                self._initialised = True
            readings: dict[str, DeviceReading] = {}
            for idx in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = _decode(pynvml.nvmlDeviceGetName(handle))
                if self._is_unified is not None and self._is_unified(idx, name):
                    continue
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
                except pynvml.NVMLError:
                    continue
                readings[uuid] = DeviceReading(
                    used_mib=float(info.used) / MIB, free_bytes=int(info.free)
                )
            return readings
        except Exception as exc:  # NVML missing, driver wedged, no GPU
            if not self._warned:
                logger.warning(
                    "foreign-GPU gate: cannot read GPU memory (%s); reporting no "
                    "occupancy",
                    exc,
                )
                self._warned = True
            return {}


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


class GpuOccupancyMonitor:
    """Track per-device occupancy across heartbeats.

    Owned by the heartbeat thread, which calls ``observe`` once per beat. Other
    threads may call ``snapshot`` freely; the dicts are rebound rather than mutated,
    so a reader never sees a half-written one. A reader can pair new occupancy with
    a previous ``free_bytes``, which is informational only and never gates placement.
    """

    def __init__(
        self,
        config: GpuGateConfig,
        probe: Callable[[], dict[str, DeviceReading]],
    ) -> None:
        self._config = config
        self._probe = probe
        self._states: dict[str, DeviceState] = {}
        self._free_bytes: dict[str, int] = {}
        self._measured_uuids: frozenset[str] = frozenset()

    @property
    def config(self) -> GpuGateConfig:
        return self._config

    @property
    def measured(self) -> bool:
        """Whether the most recent observation read any device at all."""
        return bool(self._measured_uuids)

    def observe(self, measurable: bool) -> None:
        if not self._config.enabled:
            self._states, self._free_bytes = {}, {}
            self._measured_uuids = frozenset()
            return
        if not measurable:
            self._measured_uuids = frozenset()
            return
        readings = self._probe()
        if not readings:
            # Total probe failure: drop every latch. Only a fresh clear reading
            # releases one, and a broken probe never produces one.
            self._states, self._free_bytes = {}, {}
            self._measured_uuids = frozenset()
            return
        states = dict(self._states)
        free = dict(self._free_bytes)
        for uuid, reading in readings.items():
            states[uuid] = decide(
                reading.used_mib,
                self._config.threshold_mib,
                self._config.consecutive,
                self._states.get(uuid, DeviceState()),
            )
            free[uuid] = reading.free_bytes
        self._states, self._free_bytes = states, free
        self._measured_uuids = frozenset(readings)

    def live_snapshot(self) -> dict[str, DeviceOccupancy]:
        """Only the devices read on the most recent observation.

        A device absent from the last reading keeps its latched state, which is good
        enough to advise the dispatcher but not to refuse a task outright: a partial
        probe failure would otherwise let one unreadable device hard-refuse work
        forever on a reading nobody has confirmed since.
        """
        measured = self._measured_uuids
        return {
            uuid: occupancy
            for uuid, occupancy in self.snapshot().items()
            if uuid in measured
        }

    def snapshot(self) -> dict[str, DeviceOccupancy]:
        """Per-UUID occupancy, latched from the last usable reading of each device."""
        states, free = self._states, self._free_bytes
        return {
            uuid: DeviceOccupancy(
                unavailable=state.unavailable, free_bytes=free.get(uuid)
            )
            for uuid, state in states.items()
        }
