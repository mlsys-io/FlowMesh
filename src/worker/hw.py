# worker/hw.py
"""Hardware introspection helpers.

Collects lightweight CPU/memory/GPU/network information for registration.
"""

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
            mem_total = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
            gpus.append(
                GpuInfo(
                    index=idx,
                    name=name_raw.decode() if isinstance(name_raw, bytes) else name_raw,
                    uuid=uuid_raw.decode() if isinstance(uuid_raw, bytes) else uuid_raw,
                    memory_total_bytes=mem_total,
                )
            )
    except pynvml.NVMLError:
        pass
    gpu = GpuPlatformInfo(
        driver_version=driver_version, cuda_version=cuda_version, gpus=gpus
    )
    # Network
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = None
    network = NetworkInfo(ip=ip, bandwidth_bytes_per_sec=bandwidth_bytes_per_sec)

    return WorkerHardware(cpu=cpu, memory=mem, gpu=gpu, network=network)
