import json

from shared.tasks.components.resources import GPURequirements
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)
from shared.utils.hardware import (
    available_devices,
    gpu_device_matches,
    select_matching_gpu_indices,
)


def _hardware_json_without_availability_keys() -> str:
    """A hardware blob exactly as a pre-upgrade worker registered it."""
    hw = WorkerHardware(
        cpu=CPUInfo(logical_cores=16, model="AMD EPYC 7543"),
        memory=MemoryInfo(total_bytes=64 * 1024**3),
        gpu=GpuPlatformInfo(
            driver_version="550.0",
            cuda_version="12.4",
            devices=[
                GpuInfo(
                    index=0,
                    name="NVIDIA RTX 6000 Ada Generation",
                    uuid="GPU-da09fb0d",
                    memory_total_bytes=48 * 1024**3,
                )
            ],
        ),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )
    payload = hw.model_dump(mode="json")
    for device in payload["gpu"]["devices"]:
        device.pop("gpu_available")
        device.pop("memory_free_bytes")
    return json.dumps(payload)


class TestBackwardCompatibleDeserialization:
    def test_hardware_written_before_the_fields_existed_still_parses(self) -> None:
        # Every worker registered before this change has a hardware_json blob
        # without these keys, and the registry parses it with no try/except --
        # so a missing default would break worker listing fleet-wide.
        hw = WorkerHardware.model_validate_json(
            _hardware_json_without_availability_keys()
        )
        device = hw.gpu.devices[0]
        assert device.gpu_available is None
        assert device.memory_free_bytes is None


class TestIsAvailable:
    def _device(self, **kwargs) -> GpuInfo:
        return GpuInfo(
            index=0,
            name="NVIDIA RTX 6000 Ada Generation",
            uuid="GPU-da09fb0d",
            memory_total_bytes=48 * 1024**3,
            **kwargs,
        )

    def test_an_unreported_device_is_available(self) -> None:
        # The trap the property exists to close: reading gpu_available for truth
        # would withhold every device on a cluster that has never taken a reading.
        assert self._device().gpu_available is None
        assert self._device().is_available is True

    def test_a_device_reported_free_is_available(self) -> None:
        assert self._device(gpu_available=True).is_available is True

    def test_only_an_explicit_false_withholds_a_device(self) -> None:
        assert self._device(gpu_available=False).is_available is False

    def test_available_devices_drops_only_held_ones(self) -> None:
        unreported = self._device()
        free = self._device(gpu_available=True)
        held = self._device(gpu_available=False)
        assert available_devices([unreported, free, held]) == [unreported, free]


class TestFreeMemoryIsInformationalOnly:
    def _device(self, **kwargs) -> GpuInfo:
        return GpuInfo(
            index=0,
            name="NVIDIA RTX 6000 Ada Generation",
            uuid="GPU-da09fb0d",
            memory_total_bytes=48 * 1024**3,
            **kwargs,
        )

    def test_exhausted_free_memory_alone_does_not_reject_a_device(self) -> None:
        # 19 MiB free -- the measured value from the 2026-09-19 incident. Only
        # gpu_available may gate; wiring free bytes into the matcher would
        # also reject a worker holding its own warm model.
        starved = self._device(memory_free_bytes=19 * 1024**2)
        assert gpu_device_matches(starved) is True
        assert select_matching_gpu_indices([starved], GPURequirements(count=1)) == [0]

    def test_free_memory_does_not_override_a_declared_size(self) -> None:
        starved = self._device(memory_free_bytes=19 * 1024**2)
        assert (
            gpu_device_matches(starved, min_memory_bytes=40 * 1024**3) is True
        ), "sizing still compares against total; free-vs-total is a separate change"
