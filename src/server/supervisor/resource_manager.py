import re
from enum import StrEnum
from typing import Any

from docker.types import DeviceRequest
from pydantic import BaseModel

from .. import env
from ..utils.helpers import get_docker_client


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

    @property
    def total_gpu_count(self) -> int:
        return self._env.gpu_count

    @property
    def available_gpu_count(self) -> int:
        return len(self._env.available_gpus)

    def reserve_gpus(
        self, n: int | None = None, devices: list[int] | None = None
    ) -> tuple[list[int], GpuArch]:
        """Atomically reserve GPUs and return their indices and architecture.

        Either ``n`` (auto-pick the lowest N free indices) or ``devices``
        (validate that the explicit set is free) must be given, not both.
        Selection, arch-consistency check, and removal from the available set
        run as one synchronous block, so concurrent callers on the same event
        loop never observe the same indices.
        """
        if (n is None) == (devices is None):
            raise ValueError("Provide exactly one of n or devices")

        available_gpus = self._env.available_gpus
        if n is not None:
            if n <= 0:
                raise ValueError("Invalid number of GPUs")
            if n > len(available_gpus):
                raise ValueError("Not enough available GPUs")
            picked = (
                [min(available_gpus)] if n == 1 else sorted(available_gpus)[:n]
            )
        else:
            assert devices is not None
            if not devices:
                raise ValueError("Empty device list")
            invalid = [d for d in devices if d not in available_gpus]
            if invalid:
                raise ValueError(f"Requested GPUs are not available: {invalid}")
            picked = list(devices)

        archs = {self._env.gpu_families[d] for d in picked}
        if len(archs) != 1:
            raise ValueError("Selected CUDA devices have different architectures.")
        arch = archs.pop()

        available_gpus.difference_update(picked)
        return picked, arch

    def deallocate_gpus(self, devices: list[int]) -> None:
        self._env.available_gpus.update(devices)

    def _detect_machine_env(self) -> MachineEnv:
        info = self._docker_client.info()
        cpu_count = info.get("NCPU", 0)

        gpu_families: dict[int, GpuArch] = {}
        available_gpus: set[int] = set()

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
                    command="nvidia-smi --query-gpu=index,name --format=csv,noheader",
                    remove=True,
                    **optional_kwargs,
                )
                output_str = nvidia_smi_output.decode("utf-8").strip()
                for line in output_str.split("\n"):
                    index_str, name = line.split(",", maxsplit=1)
                    index = int(index_str.strip())
                    if visible_devices is None or index in visible_devices:
                        available_gpus.add(index)
                        gpu_families[index] = GpuArch.from_name(name)
            except Exception:
                pass

        return MachineEnv(
            cpu_count=cpu_count,
            gpu_families=gpu_families,
            available_gpus=available_gpus,
        )
