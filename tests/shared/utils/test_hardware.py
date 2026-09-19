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
    gpu_device_is_claimed,
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


class TestClaimedGpuIsNotOffered:
    """Regression for the 2026-09-19 office incident.

    The NVIDIA device plugin gave luyao0's GPU 0 to the sandbox pod
    `sbx-official-dev`, which held 43.63 of 47.37 GiB. FlowMesh counts only its
    own tasks, so `wkr-141` still reported IDLE and the node still reported
    `current_gpu_count: 0`. Three Lumilake jobs in a row were placed there and
    died in vLLM init, while `wkr-140` -- the same box, same card, GPU 1 -- sat
    free with 4 MiB used.
    """

    def _device(
        self,
        *,
        total: int = 48 * 1024**3,
        free: int | None = None,
        name: str = "NVIDIA RTX 6000 Ada Generation",
    ) -> GpuInfo:
        return GpuInfo(
            index=0,
            name=name,
            uuid="GPU-da09fb0d",
            memory_total_bytes=total,
            memory_free_bytes=free,
        )

    def test_the_incident_card_is_refused_with_no_memory_constraint(self) -> None:
        # The jobs that failed carried NO gpu_memory requirement. A check that
        # only compares against `min_memory_bytes` short-circuits on None and
        # admits this card, which is exactly what happened.
        claimed = self._device(free=18 * 1024**2)  # 18.94 MiB free, measured
        assert gpu_device_is_claimed(claimed) is True
        assert gpu_device_matches(claimed) is False

    def test_its_free_sibling_is_still_offered(self) -> None:
        free_card = self._device(free=49_000 * 1024**2)  # GPU 1, 4 MiB used
        assert gpu_device_is_claimed(free_card) is False
        assert gpu_device_matches(free_card) is True

    def test_unreported_free_memory_changes_nothing(self) -> None:
        # Older workers and hosts without NVML report None. They must schedule
        # exactly as before, or this fix strands the fleet it was meant to help.
        legacy = self._device(free=None)
        assert gpu_device_is_claimed(legacy) is False
        assert gpu_device_matches(legacy) is True
        assert gpu_device_matches(legacy, min_memory_bytes=40 * 1024**3) is True

    def test_free_memory_beats_total_when_a_size_is_requested(self) -> None:
        # 48 GiB card with 8 GiB left: big enough on paper, not in reality.
        partly_used = self._device(free=8 * 1024**3)
        assert gpu_device_matches(partly_used, min_memory_bytes=40 * 1024**3) is False
        assert gpu_device_matches(partly_used, min_memory_bytes=4 * 1024**3) is True

    def test_claimed_beats_a_satisfiable_request(self) -> None:
        # Even a tiny request must not land on a card someone else holds.
        claimed = self._device(free=18 * 1024**2)
        assert gpu_device_matches(claimed, min_memory_bytes=1024) is False

    def test_scheduler_skips_the_claimed_card_and_picks_the_free_one(self) -> None:
        devices = [
            self._device(free=18 * 1024**2),  # GPU 0 -- sandbox holds it
            self._device(free=49_000 * 1024**2),  # GPU 1 -- free
        ]
        assert select_matching_gpu_indices(devices, GPURequirements(count=1)) == [1]


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
