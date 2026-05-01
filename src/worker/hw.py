# worker/hw.py
"""Hardware introspection helpers.

Collects lightweight CPU/memory/GPU/network information for registration.
"""

import logging
import os
import platform
import socket
import sys
from typing import Any

from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)

logger = logging.getLogger(__name__)


def _collect_gpu_info() -> GpuPlatformInfo:
    try:
        import pynvml
    except ImportError:
        return GpuPlatformInfo(driver_version=None, cuda_version=None, gpus=[])

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        logger.debug("NVML init failed; no GPU info collected: %s", exc)
        return GpuPlatformInfo(driver_version=None, cuda_version=None, gpus=[])

    try:
        driver_version = _safe_str(pynvml.nvmlSystemGetDriverVersion)
        cuda_version = _format_cuda_version(pynvml.nvmlSystemGetCudaDriverVersion)
        gpus: list[GpuInfo] = []
        try:
            count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            count = 0
        for idx in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = _decode(pynvml.nvmlDeviceGetName(handle))
                uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
                mem_total: int | None
                try:
                    mem_total = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
                except pynvml.NVMLError:
                    mem_total = None
            except pynvml.NVMLError:
                continue
            gpus.append(
                GpuInfo(
                    index=idx,
                    name=name,
                    uuid=uuid,
                    memory_total_bytes=mem_total,
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    return GpuPlatformInfo(
        driver_version=driver_version, cuda_version=cuda_version, gpus=gpus
    )


def _safe_str(fn: Any) -> str | None:
    try:
        return _decode(fn())
    except Exception:
        return None


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _format_cuda_version(fn: Any) -> str | None:
    try:
        raw = int(fn())
    except Exception:
        return None
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def collect_hw(*, bandwidth_bytes_per_sec: float | None = None) -> WorkerHardware:
    cpu = CPUInfo(
        logical_cores=os.cpu_count() or 0,
        model=platform.processor() or platform.machine(),
    )
    mem = MemoryInfo(total_bytes=None)
    if sys.platform.startswith("linux") and os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem.total_bytes = int(line.split()[1]) * 1024
                    break

    gpu = _collect_gpu_info()

    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = None
    network = NetworkInfo(ip=ip, bandwidth_bytes_per_sec=bandwidth_bytes_per_sec)

    return WorkerHardware(cpu=cpu, memory=mem, gpu=gpu, network=network)
