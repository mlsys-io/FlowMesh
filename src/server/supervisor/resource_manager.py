import logging
import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from docker.types import DeviceRequest
from pydantic import BaseModel, Field

from .. import env
from ..utils.helpers import get_docker_client

logger = logging.getLogger("supervisor")


class GpuArch(StrEnum):
    BLACKWELL = "blackwell"
    HOPPER = "hopper"
    UNKNOWN = "unknown"

    @classmethod
    def from_name(cls, name: str) -> "GpuArch":
        name = name.strip().lower()
        blackwell_pattern = r"(rtx50|5090|5080|5070|b100|b200|gb200|gb100|blackwell)"
        if re.search(blackwell_pattern, name):
            return cls.BLACKWELL
        hopper_pattern = r"(h100|h800|h200|hopper)"
        if re.search(hopper_pattern, name):
            return cls.HOPPER
        return cls.UNKNOWN


class MachineEnv(BaseModel):
    cpu_count: int
    gpu_families: dict[int, GpuArch]
    available_gpus: set[int]
    gpu_uuids: dict[str, int]
    """Host GPU index by device UUID."""
    hold_counts: dict[int, int] = Field(default_factory=dict)
    """Number of workers holding each reserved GPU."""

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_families)


class ResourceManager:
    _instance: "ResourceManager | None" = None

    def __init__(self) -> None:
        self._docker_client = get_docker_client()
        self._env = self._detect_machine_env()

    @classmethod
    def get_instance(cls) -> "ResourceManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def total_gpu_count(self) -> int:
        return self._env.gpu_count

    def available_gpu_count(self) -> int:
        return len(self._env.available_gpus)

    def reserve_gpus(
        self, devices: list[int] | None = None, n: int | None = None
    ) -> tuple[list[int], GpuArch]:
        """Atomically reserve GPUs and return their indices and architecture.

        Either ``devices`` (validate that the explicit set is free) or ``n``
        (auto-pick the lowest N free indices) must be given, not both.
        Selection, arch-consistency check, and removal from the available set
        run as one synchronous block, so concurrent callers on the same event
        loop never observe the same indices.

        Note: the atomicity guarantee is per event loop, not across threads —
        callers must invoke this from the supervisor's main loop thread (do
        not wrap in ``asyncio.to_thread`` or call from a worker thread).
        """
        if (n is None) == (devices is None):
            raise ValueError("Provide exactly one of n or devices")

        available_gpus = self._env.available_gpus
        # Pick devices
        if devices is not None:
            if not devices:
                raise ValueError("Empty device list")
            invalid = [d for d in devices if d not in available_gpus]
            if invalid:
                raise ValueError(f"Requested GPUs are not available: {invalid}")
            picked = devices.copy()
        else:
            assert n is not None
            if n <= 0:
                raise ValueError("Invalid number of GPUs")
            if n > len(available_gpus):
                raise ValueError("Not enough available GPUs")
            picked = [min(available_gpus)] if n == 1 else sorted(available_gpus)[:n]

        # Check architecture consistency
        archs = {self._env.gpu_families[d] for d in picked}
        if len(archs) != 1:
            raise ValueError("Selected CUDA devices have different architectures.")
        arch = archs.pop()

        available_gpus.difference_update(picked)
        self._hold(picked)
        return picked, arch

    def claim_gpus_by_uuid(self, uuids: Iterable[str]) -> tuple[list[int], list[int]]:
        """Hold the host GPUs with the given UUIDs, whoever else holds them.

        UUIDs that are not GPUs of this host are ignored. Returns the host
        indices that were free before the claim and those that were already
        held; the caller holds both and releases both with `deallocate_gpus`.
        Never raises. The same event-loop atomicity rule as `reserve_gpus`
        applies.
        """
        gpu_uuids = self._env.gpu_uuids
        devices = sorted({gpu_uuids[u] for u in uuids if u in gpu_uuids})
        available_gpus = self._env.available_gpus
        claimed = [d for d in devices if d in available_gpus]
        overlapping = [d for d in devices if d not in available_gpus]
        available_gpus.difference_update(devices)
        self._hold(devices)
        return claimed, overlapping

    def deallocate_gpus(self, devices: list[int]) -> None:
        """Drop one hold on each device; a device with no holds left is free."""
        hold_counts = self._env.hold_counts
        for device in devices:
            remaining = hold_counts.get(device, 0) - 1
            if remaining > 0:
                hold_counts[device] = remaining
            else:
                hold_counts.pop(device, None)
                self._env.available_gpus.add(device)

    def _hold(self, devices: list[int]) -> None:
        hold_counts = self._env.hold_counts
        for device in devices:
            hold_counts[device] = hold_counts.get(device, 0) + 1

    def _detect_machine_env(self) -> MachineEnv:
        info = self._docker_client.info()
        cpu_count = info.get("NCPU", 0)

        gpu_families: dict[int, GpuArch] = {}
        available_gpus: set[int] = set()
        gpu_uuids: dict[str, int] = {}

        visible_devices: set[int] | None
        if env.CUDA_VISIBLE_DEVICES is None:
            visible_devices = None
        else:
            try:
                visible_devices = {
                    int(dev.strip()) for dev in env.CUDA_VISIBLE_DEVICES.split(",")
                }
            except Exception:
                visible_devices = set()

        if visible_devices is None or len(visible_devices) > 0:
            # Detect GPUs using nvidia-smi if available
            try:
                optional_kwargs: dict[str, Any] = {}
                if env.DOCKER_GPU_RUNTIME is not None:
                    optional_kwargs["runtime"] = env.DOCKER_GPU_RUNTIME
                nvidia_smi_output = self._docker_client.containers.run(
                    image=env.SERVER_CUDA_PROBE_IMAGE,
                    device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                    command=(
                        "nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader"
                    ),
                    remove=True,
                    **optional_kwargs,
                )
                gpu_families, gpu_uuids = _parse_gpu_query(
                    nvidia_smi_output.decode("utf-8"), visible_devices
                )
                available_gpus = set(gpu_families)
            except Exception:
                pass

        return MachineEnv(
            cpu_count=cpu_count,
            gpu_families=gpu_families,
            available_gpus=available_gpus,
            gpu_uuids=gpu_uuids,
        )


def _parse_gpu_query(
    output: str, visible_devices: set[int] | None
) -> tuple[dict[int, GpuArch], dict[str, int]]:
    """Parse `nvidia-smi --query-gpu=index,name[,uuid]` CSV rows.

    Each row stands alone: one malformed row is skipped rather than costing
    the rest, and a row without a trailing `GPU-` UUID still yields its GPU.
    """
    gpu_families: dict[int, GpuArch] = {}
    gpu_uuids: dict[str, int] = {}
    for line in output.strip().splitlines():
        index_str, _, rest = line.partition(",")
        try:
            index = int(index_str.strip())
        except ValueError:
            logger.debug("Skipping unparsable GPU probe row: %r", line)
            continue
        if visible_devices is not None and index not in visible_devices:
            continue
        name, uuid = rest, ""
        head, sep, tail = rest.rpartition(",")
        if sep and tail.strip().startswith("GPU-"):
            name, uuid = head, tail.strip()
        gpu_families[index] = GpuArch.from_name(name)
        if uuid:
            gpu_uuids[uuid] = index
    return gpu_families, gpu_uuids
