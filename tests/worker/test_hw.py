"""Tests for worker hardware detection."""

from unittest.mock import mock_open, patch

from worker import hw


class _FakeNvmlError(Exception):
    pass


class _FakePynvml:
    NVMLError = _FakeNvmlError

    @staticmethod
    def nvmlInit() -> None:
        return None

    @staticmethod
    def nvmlSystemGetDriverVersion() -> bytes:
        return b"580.95.05"

    @staticmethod
    def nvmlSystemGetCudaDriverVersion() -> int:
        return 13000

    @staticmethod
    def nvmlDeviceGetCount() -> int:
        return 1

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> int:
        return index

    @staticmethod
    def nvmlDeviceGetName(handle: int) -> bytes:
        return b"NVIDIA GB10"

    @staticmethod
    def nvmlDeviceGetUUID(handle: int) -> bytes:
        return b"GPU-GB10"

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle: int) -> object:
        raise AssertionError("Unified-memory GPU probe should not query NVML VRAM")


def test_collect_hw_marks_gb10_as_unified_memory() -> None:
    meminfo = "MemTotal:       131072000 kB\n"
    with (
        patch.object(hw, "pynvml", _FakePynvml),
        patch("worker.hw._cuda_device_is_integrated", return_value=True),
        patch("worker.hw.os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=meminfo)),
    ):
        hardware = hw.collect_hw()

    assert hardware.memory.total_bytes == 131072000 * 1024
    assert hardware.gpu.driver_version == "580.95.05"
    assert hardware.gpu.cuda_version == "13.0"
    assert hardware.gpu.memory_is_unified is True
    assert hardware.gpu.shared_memory_total_bytes == 131072000 * 1024
    assert len(hardware.gpu.gpus) == 1
    assert hardware.gpu.gpus[0].name == "NVIDIA GB10"
    assert hardware.gpu.gpus[0].uuid == "GPU-GB10"
    assert hardware.gpu.gpus[0].memory_total_bytes is None


class _FallbackNamePynvml(_FakePynvml):
    @staticmethod
    def nvmlDeviceGetName(handle: int) -> bytes:
        return b"NVIDIA Tegra Thor"


def test_collect_hw_falls_back_to_name_heuristic_for_integrated_families() -> None:
    meminfo = "MemTotal:       131072000 kB\n"
    with (
        patch.object(hw, "pynvml", _FallbackNamePynvml),
        patch("worker.hw._cuda_device_is_integrated", return_value=None),
        patch("worker.hw.os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=meminfo)),
    ):
        hardware = hw.collect_hw()

    assert hardware.gpu.memory_is_unified is True
    assert hardware.gpu.shared_memory_total_bytes == 131072000 * 1024
    assert hardware.gpu.gpus[0].name == "NVIDIA Tegra Thor"
    assert hardware.gpu.gpus[0].memory_total_bytes is None
