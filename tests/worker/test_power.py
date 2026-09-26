"""GPU power sampling covers the worker's own GPUs."""

from collections.abc import Iterator

import pytest

from worker.hw import visible_gpus
from worker.power import PowerMonitor

_DEVICES = [("GPU-aaaa", 100_000), ("GPU-bbbb", 250_000), ("GPU-cccc", 300_000)]


class _FakeNvml:
    NVMLError = RuntimeError

    @staticmethod
    def nvmlInit() -> None:
        return None

    @staticmethod
    def nvmlDeviceGetCount() -> int:
        return len(_DEVICES)

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> int:
        return index

    @staticmethod
    def nvmlDeviceGetName(handle: int) -> str:
        return "NVIDIA H100"

    @staticmethod
    def nvmlDeviceGetUUID(handle: int) -> str:
        return _DEVICES[handle][0]

    @staticmethod
    def nvmlDeviceGetPowerUsage(handle: int) -> int:
        return _DEVICES[handle][1]


@pytest.fixture(autouse=True)
def _fake_nvml(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr("worker.power.pynvml", _FakeNvml)
    monkeypatch.setattr("worker.hw.pynvml", _FakeNvml)
    visible_gpus.cache_clear()
    yield
    visible_gpus.cache_clear()


@pytest.mark.parametrize("value", ["GPU-cccc,GPU-bbbb", "2,1"])
def test_samples_only_visible_gpus_under_their_cuda_ordinals(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)

    sample = PowerMonitor().sample()

    assert sample["gpu_watts"]["per_gpu"] == [
        {"index": 0, "power_w": 300.0},
        {"index": 1, "power_w": 250.0},
    ]
    assert sample["gpu_watts"]["total"] == 550.0


def test_no_visible_gpus_samples_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    sample = PowerMonitor().sample()

    assert sample["gpu_watts"] == {"total": None, "per_gpu": []}
