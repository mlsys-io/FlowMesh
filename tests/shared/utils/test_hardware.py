"""Tests for the shared GPU requirement helpers."""

import pytest

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
    gpu_device_matches,
    gpu_type_pattern,
    normalize_gpu_type,
    parse_gpu_memory_bytes,
    select_matching_gpu_indices,
    unified_gpu_memory_satisfies,
)


class TestNormalizeGpuType:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("any", None),
            ("AUTO", None),
            ("*", None),
            ("a100", "a100"),
            ("A100", "a100"),
            ("  A100  ", "a100"),
        ],
    )
    def test_normalize(self, value: str | None, expected: str | None) -> None:
        assert normalize_gpu_type(value) == expected


class TestGpuTypePattern:
    def test_returns_none_for_wildcard(self) -> None:
        assert gpu_type_pattern(None) is None
        assert gpu_type_pattern("any") is None
        assert gpu_type_pattern("*") is None

    def test_returns_case_insensitive_substring_pattern(self) -> None:
        pattern = gpu_type_pattern("A100")
        assert pattern is not None
        assert pattern.search("NVIDIA A100-SXM4-80GB") is not None
        assert pattern.search("nvidia a100") is not None
        assert pattern.search("NVIDIA T4") is None

    def test_escapes_special_regex_metachars(self) -> None:
        # User-provided strings must be matched literally, not as regex.
        pattern = gpu_type_pattern("A100.foo")
        assert pattern is not None
        assert pattern.search("NVIDIA A100xfoo") is None
        assert pattern.search("NVIDIA A100.foo") is not None


class TestParseGpuMemoryBytes:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            ("40Gi", 40 * 1024**3),
            ("512Mi", 512 * 1024**2),
            ("80GB", 80 * 1024**3),
            (1048576, 1048576),
            (1024.0, 1024),
            (0, 0),
        ],
    )
    def test_supported_inputs(
        self, value: str | int | float | None, expected: int | None
    ) -> None:
        assert parse_gpu_memory_bytes(value) == expected

    def test_unparsable_string_returns_none(self) -> None:
        assert parse_gpu_memory_bytes("garbage") is None


class TestGpuDeviceMatches:
    def _device(
        self, name: str = "NVIDIA A100-SXM4-80GB", memory_bytes: int = 80 * 1024**3
    ) -> GpuInfo:
        return GpuInfo(index=0, name=name, uuid="x", memory_total_bytes=memory_bytes)

    def test_no_constraints_accepts(self) -> None:
        assert gpu_device_matches(self._device()) is True

    def test_type_match(self) -> None:
        pattern = gpu_type_pattern("A100")
        assert gpu_device_matches(self._device(), type_pattern=pattern) is True

    def test_type_mismatch(self) -> None:
        pattern = gpu_type_pattern("H100")
        assert gpu_device_matches(self._device(), type_pattern=pattern) is False

    def test_memory_meets_floor(self) -> None:
        assert (
            gpu_device_matches(
                self._device(memory_bytes=80 * 1024**3),
                min_memory_bytes=40 * 1024**3,
            )
            is True
        )

    def test_memory_below_floor(self) -> None:
        assert (
            gpu_device_matches(
                self._device(memory_bytes=16 * 1024**3),
                min_memory_bytes=40 * 1024**3,
            )
            is False
        )

    def test_missing_device_memory_treated_as_zero(self) -> None:
        device = GpuInfo(index=0, name="A100", uuid="x", memory_total_bytes=None)
        assert gpu_device_matches(device, min_memory_bytes=1) is False
        # No memory constraint still accepts even when memory_total_bytes is None.
        assert gpu_device_matches(device) is True

    def test_combined_predicates(self) -> None:
        pattern = gpu_type_pattern("A100")
        # Type matches but memory below floor → reject.
        assert (
            gpu_device_matches(
                self._device(memory_bytes=40 * 1024**3),
                type_pattern=pattern,
                min_memory_bytes=80 * 1024**3,
            )
            is False
        )
        # Both satisfied → accept.
        assert (
            gpu_device_matches(
                self._device(memory_bytes=80 * 1024**3),
                type_pattern=pattern,
                min_memory_bytes=40 * 1024**3,
            )
            is True
        )


class TestSelectMatchingGpuIndices:
    def _devices(self) -> list[GpuInfo]:
        return [
            GpuInfo(
                index=0, name="NVIDIA T4", uuid="t4-0", memory_total_bytes=16 * 1024**3
            ),
            GpuInfo(
                index=1,
                name="NVIDIA A100-SXM4-40GB",
                uuid="a100-0",
                memory_total_bytes=40 * 1024**3,
            ),
            GpuInfo(
                index=2,
                name="NVIDIA A100-SXM4-80GB",
                uuid="a100-1",
                memory_total_bytes=80 * 1024**3,
            ),
            GpuInfo(
                index=3,
                name="NVIDIA A100-SXM4-80GB",
                uuid="a100-2",
                memory_total_bytes=80 * 1024**3,
            ),
        ]

    def test_no_constraints_returns_all_indices(self) -> None:
        result = select_matching_gpu_indices(self._devices(), GPURequirements())
        assert result == [0, 1, 2, 3]

    def test_type_filter(self) -> None:
        result = select_matching_gpu_indices(
            self._devices(), GPURequirements(type="A100")
        )
        assert result == [1, 2, 3]

    def test_memory_filter(self) -> None:
        result = select_matching_gpu_indices(
            self._devices(), GPURequirements(memory="80Gi")
        )
        assert result == [2, 3]

    def test_per_device_and_semantics(self) -> None:
        # An A100-40GB device matches type but not the 80Gi memory floor; it
        # must be excluded. This is what makes the helper consistent across
        # the dispatcher and the SSH executor.
        result = select_matching_gpu_indices(
            self._devices(), GPURequirements(type="A100", memory="80Gi")
        )
        assert result == [2, 3]

    def test_limit_stops_early(self) -> None:
        result = select_matching_gpu_indices(
            self._devices(), GPURequirements(type="A100"), limit=2
        )
        assert result == [1, 2]

    def test_limit_zero_returns_empty(self) -> None:
        result = select_matching_gpu_indices(
            self._devices(), GPURequirements(), limit=0
        )
        assert result == []

    def test_empty_devices(self) -> None:
        assert select_matching_gpu_indices([], GPURequirements(type="A100")) == []


def _unified_hw(
    *,
    is_unified: bool,
    shared_bytes: int | None,
) -> WorkerHardware:
    return WorkerHardware(
        cpu=CPUInfo(logical_cores=8, model="x"),
        memory=MemoryInfo(total_bytes=128 * 1024**3),
        gpu=GpuPlatformInfo(
            driver_version=None,
            cuda_version=None,
            devices=[
                GpuInfo(
                    index=0, name="NVIDIA GB10", uuid="gb10", memory_total_bytes=None
                )
            ],
            memory_is_unified=is_unified,
            shared_memory_total_bytes=shared_bytes,
        ),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )


class TestUnifiedGpuMemorySatisfies:
    def test_non_unified_returns_false(self) -> None:
        hw = _unified_hw(is_unified=False, shared_bytes=128 * 1024**3)
        assert unified_gpu_memory_satisfies(hw, 40 * 1024**3, 1) is False

    def test_no_shared_pool_returns_false(self) -> None:
        hw = _unified_hw(is_unified=True, shared_bytes=None)
        assert unified_gpu_memory_satisfies(hw, 40 * 1024**3, 1) is False

    def test_pool_covers_single_gpu(self) -> None:
        hw = _unified_hw(is_unified=True, shared_bytes=128 * 1024**3)
        assert unified_gpu_memory_satisfies(hw, 40 * 1024**3, 1) is True

    def test_per_gpu_share_below_request(self) -> None:
        # 128 GiB pool / 4 requested = 32 GiB per slot, below the 40 GiB floor.
        hw = _unified_hw(is_unified=True, shared_bytes=128 * 1024**3)
        assert unified_gpu_memory_satisfies(hw, 40 * 1024**3, 4) is False

    def test_per_gpu_share_meets_request(self) -> None:
        hw = _unified_hw(is_unified=True, shared_bytes=128 * 1024**3)
        assert unified_gpu_memory_satisfies(hw, 32 * 1024**3, 4) is True
