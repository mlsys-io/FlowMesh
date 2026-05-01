# worker/hw.py
"""Hardware introspection helpers.

Collects lightweight CPU/memory/GPU/network information for registration.
"""

import logging
import os
import platform
import socket
import sys

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


def _collect_gpu_info() -> GpuPlatformInfo:
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        logger.debug("NVML init failed; no GPU info collected: %s", exc)
        return GpuPlatformInfo(driver_version=None, cuda_version=None, gpus=[])

    try:
        try:
            driver_version_raw = pynvml.nvmlSystemGetDriverVersion()
            driver_version: str | None = (
                driver_version_raw.decode()
                if isinstance(driver_version_raw, bytes)
                else driver_version_raw
            )
        except pynvml.NVMLError:
            driver_version = None
        try:
            cuda_raw = int(pynvml.nvmlSystemGetCudaDriverVersion())
            cuda_version: str | None = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}"
        except pynvml.NVMLError:
            cuda_version = None
        try:
            count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            count = 0
        gpus: list[GpuInfo] = []
        for idx in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name_raw = pynvml.nvmlDeviceGetName(handle)
                uuid_raw = pynvml.nvmlDeviceGetUUID(handle)
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
                    name=name_raw.decode() if isinstance(name_raw, bytes) else name_raw,
                    uuid=uuid_raw.decode() if isinstance(uuid_raw, bytes) else uuid_raw,
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
