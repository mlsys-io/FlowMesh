from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from shared.schemas.node import NodeInfo


class NodeRegisterResponse(BaseModel):
    node_id: str = Field(description="Registered node identifier.")


class WorkerRegisterResponse(BaseModel):
    worker_id: str = Field(description="Registered worker identifier.")


class NodeWorkerStatus(StrEnum):
    STARTING = "STARTING"
    IDLE = "IDLE"
    BUSY = "BUSY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class CPUInfo(BaseModel):
    logical_cores: int | None = Field(
        default=None, description="Number of logical CPU cores."
    )
    model: str | None = Field(
        default=None, description="CPU model name or architecture."
    )

    arch: str | None = Field(default=None, description="CPU architecture.")
    name: str | None = Field(default=None, description="CPU model name.")
    has_avx: bool | None = Field(default=None, description="CPU supports AVX.")


class MemoryInfo(BaseModel):
    total_bytes: int | None = Field(default=None, description="Total memory in bytes.")


class GpuInfo(BaseModel):
    index: int | None = Field(default=None, description="GPU index.")
    name: str | None = Field(default=None, description="GPU name.")
    uuid: str | None = Field(default=None, description="GPU UUID.")
    memory_total_bytes: int | None = Field(
        default=None, description="Total GPU memory in bytes."
    )


class GpuPlatformInfo(BaseModel):
    driver_version: str | None = Field(default=None, description="GPU driver version.")
    cuda_version: str | None = Field(default=None, description="CUDA version.")
    devices: list[GpuInfo] = Field(
        default_factory=list, description="List of GPU devices."
    )
    memory_is_unified: bool = Field(
        default=False,
        description="Whether GPU memory is a unified/shared system memory pool.",
    )
    shared_memory_total_bytes: int | None = Field(
        default=None,
        description="Total shared GPU/system memory pool in bytes when unified.",
    )

    gpu_arch: str | None = Field(default=None, description="GPU architecture.")
    compute_cap: int | None = Field(
        default=None, description="CUDA compute capability (e.g. 890 -> 8.9)."
    )
    bw_nvlink: float | None = Field(default=None, description="NVLink bandwidth.")
    gpu_lanes: int | None = Field(default=None, description="Number of GPU lanes.")
    gpu_mem_bw: float | None = Field(default=None, description="GPU memory bandwidth.")
    pci_gen: float | None = Field(default=None, description="PCIe generation.")
    pcie_bw: float | None = Field(default=None, description="PCIe bandwidth.")
    cost_per_hour: float | None = Field(
        default=None, description="GPU cost per hour in USD."
    )
    total_flops: float | None = Field(default=None, description="Total FLOPs.")


class NetworkInfo(BaseModel):
    ip: str | None = Field(default=None, description="Network IP address.")
    bandwidth_bytes_per_sec: float | None = Field(
        default=None, description="Network bandwidth in bytes per second."
    )

    public_ipaddr: str | None = Field(default=None, description="Public IP address.")
    local_ipaddrs: list[str] | None = Field(
        default=None, description="Local IP addresses."
    )
    geolocation: str | None = Field(default=None, description="Geolocation string.")
    inet_down: float | None = Field(default=None, description="Internet download rate.")
    inet_up: float | None = Field(default=None, description="Internet upload rate.")


class StorageInfo(BaseModel):
    disk_name: str | None = Field(default=None, description="Disk name.")
    disk_space: float | None = Field(default=None, description="Disk space in GB.")
    disk_usage: float | None = Field(default=None, description="Disk usage.")
    disk_util: float | None = Field(default=None, description="Disk utilization.")


class HostInfo(BaseModel):
    os_version: str | None = Field(default=None, description="OS version.")
    reliability: float | None = Field(default=None, description="Reliability score.")


class WorkerHardware(BaseModel):
    cpu: CPUInfo = Field(default_factory=CPUInfo, description="CPU information.")
    memory: MemoryInfo = Field(
        default_factory=MemoryInfo, description="Memory information."
    )
    gpu: GpuPlatformInfo = Field(
        default_factory=GpuPlatformInfo, description="GPU information."
    )
    network: NetworkInfo = Field(
        default_factory=NetworkInfo, description="Network information."
    )
    storage: StorageInfo = Field(
        default_factory=StorageInfo, description="Storage information."
    )
    host: HostInfo = Field(default_factory=HostInfo, description="Host information.")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific metadata."
    )


class NodeWorkerInfo(BaseModel):
    id: str | None = Field(..., description="Worker ID")
    name: str = Field(..., description="Worker name")
    namespace: str = Field(..., description="Worker namespace")
    cluster: str = Field(..., description="Worker cluster")
    node_id: str = Field(..., description="Associated node ID")
    node_alias: str = Field(..., description="Associated node alias")
    provider: str = Field(..., description="Worker provider")
    status: NodeWorkerStatus = Field(..., description="Current worker status")
    hardware: WorkerHardware | None = Field(
        default=None, description="Hardware metadata"
    )


__all__ = [
    "NodeInfo",
    "NodeRegisterResponse",
    "NodeWorkerInfo",
    "NodeWorkerStatus",
    "WorkerRegisterResponse",
]
