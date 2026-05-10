# worker/hw.py
"""Hardware introspection helpers.

Collects lightweight CPU/memory/GPU/network information for registration.
"""

import os
import platform
import re
import socket
import sys
from ctypes import CDLL, POINTER, byref, c_int
from ctypes.util import find_library
from functools import cache

import pynvml

from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)

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


def _device_uses_unified_memory(device_index: int, name: str) -> bool:
    integrated = _cuda_device_is_integrated(device_index)
    if integrated is not None:
        return integrated
    return _is_unified_memory_gpu(name)


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
    gpus: list[GpuInfo] = []
    unified_memory = False
    try:
        pynvml.nvmlInit()
        raw = pynvml.nvmlSystemGetDriverVersion()
        driver_version = raw.decode() if isinstance(raw, bytes) else raw
        cuda_raw = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_version = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}"
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            uuid_raw = pynvml.nvmlDeviceGetUUID(handle)
            name = name_raw.decode() if isinstance(name_raw, bytes) else name_raw
            uuid = uuid_raw.decode() if isinstance(uuid_raw, bytes) else uuid_raw
            gpu_uses_unified_memory = _device_uses_unified_memory(idx, name)
            unified_memory = unified_memory or gpu_uses_unified_memory
            mem_total: int | None = None
            if not gpu_uses_unified_memory:
                try:
                    mem_total_raw = pynvml.nvmlDeviceGetMemoryInfo(handle).total
                except pynvml.NVMLError:
                    mem_total_raw = None
                if mem_total_raw:
                    mem_total = int(mem_total_raw)
            gpus.append(
                GpuInfo(
                    index=idx,
                    name=name,
                    uuid=uuid,
                    memory_total_bytes=mem_total,
                )
            )
    except pynvml.NVMLError:
        pass
    gpu = GpuPlatformInfo(
        driver_version=driver_version,
        cuda_version=cuda_version,
        gpus=gpus,
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
