"""Detect GPU memory held by processes outside this worker, per device.

A worker's GPU can be claimed by a process FlowMesh cannot see. The worker keeps
running CPU work either way, so availability is reported as a per-device resource
fact rather than a worker status, and the dispatcher skips only the held devices.

Signal: device memory in use while the worker runs nothing of its own --
neither a task nor a warm GPU executor, which stays resident between tasks and
may still hold VRAM. The caller decides when that holds (see
``Lifecycle._observe_gpu``) and passes it as ``measurable``; any usage seen then
belongs to another tenant. NVML's per-process list cannot attribute memory here
-- inside a container it reports host PIDs that do not map back to the worker's
own processes -- hence the coarse per-device signal.

Two kinds of "cannot read" are deliberately distinguished:

* **Suppressed** -- a task is running, a GPU executor is warm, or we are inside
  the post-task grace window. The last reading latches, because the device is
  very likely still in whatever state we last saw it in.
* **Probe failure** -- NVML is unreadable. The reading clears to "no opinion".
  Latching here would be a trap: only a fresh clear reading releases a latch,
  and a broken probe never produces one, so a worker would be excluded from GPU
  placement forever.
"""

import logging
from collections.abc import Callable, Mapping
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
class DeviceAvailability:
    """What the worker reports about one device."""

    available: bool = True
    free_bytes: int | None = None


@dataclass(frozen=True)
class DeviceState:
    """A device's availability, plus the streak of readings disagreeing with it.

    Nothing but ``streak`` is internal, so reporting is a projection rather than a
    reconstruction.
    """

    availability: DeviceAvailability = DeviceAvailability()
    streak: int = 0


def decide_availability(
    reading: DeviceReading,
    threshold_mib: int,
    consecutive: int,
    previous: DeviceState,
) -> DeviceState:
    """Next state for one device from one observation. Pure, for testing.

    Only called for a device that was actually read; a device the probe could not
    read keeps its previous state by not being passed here at all.
    """
    available = reading.used_mib <= threshold_mib
    observed = DeviceAvailability(available=available, free_bytes=reading.free_bytes)
    if available == previous.availability.available:
        return DeviceState(observed)
    streak = previous.streak + 1
    if streak >= max(1, consecutive):
        return DeviceState(observed)
    latched = DeviceAvailability(
        available=previous.availability.available, free_bytes=reading.free_bytes
    )
    return DeviceState(latched, streak)


class NvmlDeviceProbe:
    """Per-UUID memory readings for the worker's own GPUs.

    NVML lists every GPU the process can reach, including ones
    `CUDA_VISIBLE_DEVICES` hides from it, so given ``devices`` (the worker's
    reported devices, UUID to CUDA ordinal) only those are read. Without it every
    NVML device is read, under its NVML index. Unified-memory devices (e.g. GB10)
    are omitted: their "used" figure is system RAM, not a card another tenant is
    holding.

    Returns ``{}`` when NVML itself cannot be reached, which the monitor reads as total
    probe failure. A device that individually fails to read is simply absent from the
    result, leaving whatever the monitor last knew about it untouched.
    """

    def __init__(
        self,
        is_unified: Callable[[int, str], bool] | None = None,
        devices: Mapping[str, int] | None = None,
    ) -> None:
        self._is_unified = is_unified
        self._devices = devices
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
                try:
                    uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
                except pynvml.NVMLError:
                    continue
                ordinal = idx if self._devices is None else self._devices.get(uuid)
                if ordinal is None:
                    continue
                if self._is_unified is not None and self._is_unified(ordinal, name):
                    continue
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                except pynvml.NVMLError:
                    continue
                readings[uuid] = DeviceReading(
                    used_mib=float(info.used) / MIB, free_bytes=int(info.free)
                )
            return readings
        except Exception as exc:  # NVML missing, driver wedged, no GPU
            if not self._warned:
                logger.warning(
                    "foreign-GPU gate: cannot read GPU memory (%s); reporting every "
                    "device as available",
                    exc,
                )
                self._warned = True
            return {}


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


class GpuAvailabilityMonitor:
    """Track per-device availability across heartbeats.

    Owned by the heartbeat thread, which calls ``observe`` once per beat. Other
    threads may call ``snapshot`` freely; the dict is rebound rather than mutated,
    so a reader never sees a half-written one.

    ``GpuGateConfig.enabled`` is checked here as well as at construction: the kill
    switch must not depend on one call site staying correct.
    """

    def __init__(
        self,
        config: GpuGateConfig,
        probe: Callable[[], dict[str, DeviceReading]],
    ) -> None:
        self._config = config
        self._probe = probe
        self._devices: dict[str, DeviceState] = {}
        self._measured_uuids: frozenset[str] = frozenset()

    @property
    def config(self) -> GpuGateConfig:
        return self._config

    def observe(self, measurable: bool) -> None:
        if not self._config.enabled or not measurable:
            self._measured_uuids = frozenset()
            return
        readings = self._probe()
        if not readings:
            # Total probe failure: drop every latch. Only a fresh clear reading
            # releases one, and a broken probe never produces one.
            self._devices = {}
            self._measured_uuids = frozenset()
            return
        devices = self._devices.copy()
        for uuid, reading in readings.items():
            devices[uuid] = decide_availability(
                reading,
                self._config.threshold_mib,
                self._config.consecutive,
                self._devices.get(uuid) or DeviceState(),
            )
        self._devices = devices
        self._measured_uuids = frozenset(readings)

    def live_snapshot(self) -> dict[str, DeviceAvailability]:
        """Only the devices read on the most recent observation.

        A device absent from the last reading keeps its latched state, which is good
        enough to advise the dispatcher but not to refuse a task outright: a partial
        probe failure would otherwise let one unreadable device hard-refuse work
        forever on a reading nobody has confirmed since.
        """
        measured = self._measured_uuids
        return {
            uuid: availability
            for uuid, availability in self.snapshot().items()
            if uuid in measured
        }

    def snapshot(self) -> dict[str, DeviceAvailability]:
        """Per-UUID availability, latched from each device's last usable reading."""
        return {uuid: state.availability for uuid, state in self._devices.items()}
