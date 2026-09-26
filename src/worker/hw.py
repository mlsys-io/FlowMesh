# worker/hw.py
"""Hardware introspection helpers.

Collects lightweight CPU/memory/GPU/network information for registration.
"""

import logging
import os
import platform
import re
import socket
import sys
from collections.abc import Callable
from ctypes import CDLL, POINTER, byref, c_int
from ctypes.util import find_library
from dataclasses import dataclass
from functools import cache
from typing import Any

import pynvml

from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)

logger = logging.getLogger(__name__)

_UNIFIED_GPU_NAME_PATTERN = re.compile(r"\b(?:gb10|tegra|thor)\b", re.IGNORECASE)
_CUDA_DEV_ATTR_INTEGRATED = 18


def _is_unified_memory_gpu(name: str) -> bool:
    return bool(_UNIFIED_GPU_NAME_PATTERN.search(name))


@cache
def _load_cudart() -> CDLL | None:
    """Load the CUDA runtime library for capability probes when available."""
    library_name = find_library("cudart")
    candidates = [library_name, "libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return CDLL(candidate)
        except OSError:
            continue
    return None


@cache
def _cuda_device_is_integrated(device_index: int) -> bool | None:
    """Return CUDA's integrated-device flag as the primary UMA signal.

    References:
    - cudaDeviceGetAttribute / cudaGetDeviceCount:
      https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    - cudaDeviceProp.integrated ("Device is integrated as opposed to discrete"):
      https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    - Unified/system memory model background:
      https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/understanding-memory.html
    """
    cudart = _load_cudart()
    if cudart is None:
        return None

    cuda_get_device_count = getattr(cudart, "cudaGetDeviceCount", None)
    cuda_device_get_attribute = getattr(cudart, "cudaDeviceGetAttribute", None)
    if cuda_get_device_count is None or cuda_device_get_attribute is None:
        return None

    cuda_get_device_count.argtypes = [POINTER(c_int)]
    cuda_get_device_count.restype = c_int
    cuda_device_get_attribute.argtypes = [POINTER(c_int), c_int, c_int]
    cuda_device_get_attribute.restype = c_int

    count = c_int()
    if cuda_get_device_count(byref(count)) != 0:
        return None
    if device_index < 0 or device_index >= count.value:
        return None

    value = c_int()
    if (
        cuda_device_get_attribute(byref(value), _CUDA_DEV_ATTR_INTEGRATED, device_index)
        != 0
    ):
        return None
    return bool(value.value)


def device_uses_unified_memory(device_index: int, name: str) -> bool:
    """Return whether a visible GPU uses the host's shared memory pool.

    CUDA's integrated-device attribute takes precedence over the device-name fallback.
    """
    integrated = _cuda_device_is_integrated(device_index)
    if integrated is not None:
        return integrated
    return _is_unified_memory_gpu(name)


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _mig_uuids(nvml_index: int) -> dict[int, str]:
    """UUIDs of the MIG devices carved out of an NVML device, by MIG slot."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
    except pynvml.NVMLError:
        return {}
    uuids: dict[int, str] = {}
    for slot in range(count):
        try:
            mig = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, slot)
            uuids[slot] = _decode(pynvml.nvmlDeviceGetUUID(mig))
        except pynvml.NVMLError:
            continue
    return uuids


def visible_device_order(
    devices: list[tuple[str, str]],
    mig_uuids: Callable[[int], dict[int, str]] = _mig_uuids,
) -> list[tuple[int, int | None]]:
    """Return the devices `CUDA_VISIBLE_DEVICES` leaves visible, in CUDA order.

    `devices` is each NVML device's (uuid, name), in NVML order; NVML itself
    ignores the variable. Each result is an NVML index and, for a MIG slice of
    that GPU, its MIG slot. Entries follow CUDA's rules: a `GPU-` entry names a
    device by a unique UUID prefix, an integer by its position, and the first
    entry that names no device ends the list. Integers are read in PCI bus
    order, which is NVML's and matches CUDA's only under
    `CUDA_DEVICE_ORDER=PCI_BUS_ID` or on identical GPUs. A `MIG-` entry names a
    slice of a GPU (`mig_uuids` lists an NVML device's MIG device UUIDs by
    slot); it also ends the list, since a CUDA process enumerates at most one
    MIG device.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return [(i, None) for i in range(len(devices))]
    uuids = [uuid.lower() for uuid, _ in devices]
    order: list[tuple[int, int | None]] = []
    by_position = False
    for entry in (token.strip() for token in value.split(",")):
        if entry.upper().startswith("MIG-"):
            slices = [
                (i, slot)
                for i in range(len(devices))
                for slot, mig in mig_uuids(i).items()
                if mig.lower().startswith(entry.lower())
            ]
            if len(slices) != 1:
                logger.warning(
                    "CUDA_VISIBLE_DEVICES entry %s names no single MIG device; "
                    "reporting no GPUs from it on",
                    entry,
                )
            elif all(i != slices[0][0] for i, _ in order):
                order.append(slices[0])
            break
        if entry.upper().startswith("GPU-"):
            matches = [
                i for i, uuid in enumerate(uuids) if uuid.startswith(entry.lower())
            ]
            index = matches[0] if len(matches) == 1 else None
        else:
            try:
                index = int(entry)
            except ValueError:
                index = None
            if index is not None and not 0 <= index < len(devices):
                index = None
            by_position = by_position or index is not None
        if index is None or any(i == index for i, _ in order):
            break
        order.append((index, None))
    if (
        by_position
        and len({name for _, name in devices}) > 1
        and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        logger.warning(
            "CUDA_VISIBLE_DEVICES lists GPUs by position on a host with mixed GPU "
            "models; reading positions in PCI bus order, which CUDA uses only under "
            "CUDA_DEVICE_ORDER=PCI_BUS_ID"
        )
    return order


@dataclass(frozen=True)
class VisibleGpu:
    """A GPU this process's CUDA calls can use, located in NVML."""

    ordinal: int
    """The device's CUDA ordinal."""
    nvml_index: int
    uuid: str
    """The physical GPU's UUID, also when the process sees a MIG slice of it."""
    name: str
    mig_slot: int | None = None
    """The MIG slot of the slice the process sees, if it sees only a slice."""


@cache
def visible_gpus() -> tuple[VisibleGpu, ...]:
    """The GPUs `CUDA_VISIBLE_DEVICES` leaves visible, in CUDA order.

    Needs NVML initialised, and raises `pynvml.NVMLError` when it cannot be read.
    Resolved once per process, as CUDA does, so an executor rewriting the variable
    later does not move the worker's devices.
    """
    devices: list[tuple[str, str]] = []
    for idx in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        devices.append(
            (
                _decode(pynvml.nvmlDeviceGetUUID(handle)),
                _decode(pynvml.nvmlDeviceGetName(handle)),
            )
        )
    return tuple(
        VisibleGpu(ordinal, nvml_index, *devices[nvml_index], mig_slot)
        for ordinal, (nvml_index, mig_slot) in enumerate(visible_device_order(devices))
    )


def nvml_memory_handle(gpu: VisibleGpu) -> Any:
    """The NVML handle whose memory is the process's: its MIG slice, if any."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.nvml_index)
    if gpu.mig_slot is None:
        return handle
    return pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, gpu.mig_slot)


def collect_hw(*, bandwidth_bytes_per_sec: float | None = None) -> WorkerHardware:
    # CPU
    cpu = CPUInfo(
        logical_cores=os.cpu_count() or 0,
        model=platform.processor() or platform.machine(),
    )
    # Memory
    mem = MemoryInfo(total_bytes=None)
    if sys.platform.startswith("linux") and os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem.total_bytes = int(line.split()[1]) * 1024
                    break
    # GPU (NVIDIA)
    driver_version: str | None = None
    cuda_version: str | None = None
    devices: list[GpuInfo] = []
    unified_memory = False
    try:
        pynvml.nvmlInit()
        raw = pynvml.nvmlSystemGetDriverVersion()
        driver_version = raw.decode() if isinstance(raw, bytes) else raw
        cuda_raw = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_version = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}"
        for visible in visible_gpus():
            gpu_uses_unified_memory = device_uses_unified_memory(
                visible.ordinal, visible.name
            )
            unified_memory = unified_memory or gpu_uses_unified_memory
            mem_total: int | None = None
            if not gpu_uses_unified_memory:
                try:
                    mem_total_raw = pynvml.nvmlDeviceGetMemoryInfo(
                        nvml_memory_handle(visible)
                    ).total
                except pynvml.NVMLError:
                    mem_total_raw = None
                if mem_total_raw:
                    mem_total = int(mem_total_raw)
            devices.append(
                GpuInfo(
                    index=visible.ordinal,
                    name=visible.name,
                    uuid=visible.uuid,
                    memory_total_bytes=mem_total,
                )
            )
    except pynvml.NVMLError:
        pass
    gpu = GpuPlatformInfo(
        driver_version=driver_version,
        cuda_version=cuda_version,
        devices=devices,
        memory_is_unified=unified_memory,
        shared_memory_total_bytes=mem.total_bytes if unified_memory else None,
    )
    # Network
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = None
    network = NetworkInfo(ip=ip, bandwidth_bytes_per_sec=bandwidth_bytes_per_sec)

    return WorkerHardware(cpu=cpu, memory=mem, gpu=gpu, network=network)
