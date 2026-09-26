"""Tests for worker hardware detection."""

from unittest.mock import mock_open, patch

import pytest

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


def test_collect_hw_marks_gb10_as_unified_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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
    assert len(hardware.gpu.devices) == 1
    assert hardware.gpu.devices[0].name == "NVIDIA GB10"
    assert hardware.gpu.devices[0].uuid == "GPU-GB10"
    assert hardware.gpu.devices[0].memory_total_bytes is None


class _FallbackNamePynvml(_FakePynvml):
    @staticmethod
    def nvmlDeviceGetName(handle: int) -> bytes:
        return b"NVIDIA Tegra Thor"


def test_collect_hw_falls_back_to_name_heuristic_for_integrated_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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
    assert hardware.gpu.devices[0].name == "NVIDIA Tegra Thor"
    assert hardware.gpu.devices[0].memory_total_bytes is None


_HOST = [
    ("GPU-aaaa-1111", "NVIDIA H100"),
    ("GPU-bbbb-2222", "NVIDIA H100"),
    ("GPU-cccc-3333", "NVIDIA H100"),
    ("GPU-cccd-4444", "NVIDIA H100"),
]
_MIG = {2: ["MIG-aaaa-1", "MIG-aaaa-2"], 3: ["MIG-bbbb-1"]}


def _host_migs(index: int) -> list[str]:
    return _MIG.get(index, [])


class TestVisibleDeviceOrder:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, [0, 1, 2, 3]),
            ("", []),
            ("-1", []),
            ("2,0", [2, 0]),
            (" 1 , 3 ", [1, 3]),
            ("GPU-bbbb-2222,GPU-aaaa-1111", [1, 0]),
            ("GPU-BBBB", [1]),
            ("1,7,2", [1]),
            ("1,1,2", [1]),
            ("GPU-ccc,1", []),
            ("NoDevFiles", []),
            ("MIG-aaaa-1", [2]),
            ("MIG-aaaa-1,MIG-bbbb-1", [2]),
            ("1,MIG-bbbb-1", [1, 3]),
            ("2,MIG-aaaa-1", [2]),
            ("MIG-aaaa", []),
            ("MIG-zzzz,1", []),
        ],
    )
    def test_follows_cuda_rules(
        self, monkeypatch: pytest.MonkeyPatch, value: str | None, expected: list[int]
    ) -> None:
        if value is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        assert hw.visible_device_order(_HOST, _host_migs) == expected

    def test_warns_on_a_mig_entry_it_cannot_resolve(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-GPU-aaaa/1/0")
        assert hw.visible_device_order(_HOST, _host_migs) == []
        assert "MIG-GPU-aaaa/1/0" in caplog.text

    def test_mig_slices_are_not_listed_without_a_mig_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

        def unexpected(_index: int) -> list[str]:
            raise AssertionError("MIG devices listed with no MIG entry")

        assert hw.visible_device_order(_HOST, unexpected) == [0, 1]

    def test_warns_when_positions_are_ambiguous_on_mixed_gpus(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        mixed = [("GPU-a", "NVIDIA H100"), ("GPU-b", "NVIDIA L4")]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        assert hw.visible_device_order(mixed) == [1]
        assert "PCI_BUS_ID" in caplog.text

        caplog.clear()
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        hw.visible_device_order(mixed)
        assert caplog.text == ""

    @pytest.mark.parametrize("value", ["", "NoDevFiles", "GPU-b"])
    def test_no_warning_when_no_position_was_read(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        value: str,
    ) -> None:
        mixed = [("GPU-a", "NVIDIA H100"), ("GPU-b", "NVIDIA L4")]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        hw.visible_device_order(mixed)
        assert caplog.text == ""


class _FourGpuPynvml(_FakePynvml):
    @staticmethod
    def nvmlDeviceGetCount() -> int:
        return len(_HOST)

    @staticmethod
    def nvmlDeviceGetName(handle: int) -> bytes:
        return _HOST[handle][1].encode()

    @staticmethod
    def nvmlDeviceGetUUID(handle: int) -> bytes:
        return _HOST[handle][0].encode()

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle: int) -> object:
        return type("Mem", (), {"total": 80 << 30})()


def test_collect_hw_reports_only_visible_gpus_under_their_cuda_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    probed: list[int] = []

    def integrated(ordinal: int) -> bool:
        probed.append(ordinal)
        return False

    with (
        patch.object(hw, "pynvml", _FourGpuPynvml),
        patch("worker.hw._cuda_device_is_integrated", side_effect=integrated),
    ):
        hardware = hw.collect_hw()

    devices = hardware.gpu.devices
    assert [(d.index, d.uuid) for d in devices] == [
        (0, "GPU-cccd-4444"),
        (1, "GPU-bbbb-2222"),
    ]
    assert probed == [0, 1]


class _MigPynvml(_FourGpuPynvml):
    """GPU 2 is split into two MIG devices."""

    @staticmethod
    def nvmlDeviceGetMaxMigDeviceCount(handle: int) -> int:
        if handle != 2:
            raise _FakeNvmlError("MIG not enabled")
        return 3

    @staticmethod
    def nvmlDeviceGetMigDeviceHandleByIndex(handle: int, slot: int) -> tuple[int, int]:
        if slot == 2:
            raise _FakeNvmlError("empty slot")
        return (handle, slot)

    @staticmethod
    def nvmlDeviceGetUUID(handle: int | tuple[int, int]) -> bytes:
        if isinstance(handle, tuple):
            return f"MIG-slice-{handle[0]}-{handle[1]}".encode()
        return _HOST[handle][0].encode()


def test_collect_hw_reports_the_gpu_a_mig_slice_belongs_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-slice-2-1")

    with (
        patch.object(hw, "pynvml", _MigPynvml),
        patch("worker.hw._cuda_device_is_integrated", return_value=False),
    ):
        hardware = hw.collect_hw()

    assert [(d.index, d.uuid) for d in hardware.gpu.devices] == [(0, "GPU-cccc-3333")]
