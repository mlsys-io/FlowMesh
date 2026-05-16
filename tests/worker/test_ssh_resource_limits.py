"""Tests for SSH container resource-limit resolution and propagation."""

import logging
from typing import Any, cast

import pytest

from shared.schemas.worker import SSHLimits
from shared.tasks.specs import SSHSpecStrict
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)
from tests.worker.factories import make_worker_config, make_worker_hardware
from worker.config import WorkerConfig
from worker.executors.ssh_executor import SSHConfig


def _spec(resources: dict[str, object] | None = None) -> SSHSpecStrict:
    payload: dict[str, object] = {
        "taskType": "ssh",
        "interactive": False,
        "image": "python:3.12-slim",
        "command": ["true"],
    }
    if resources is not None:
        payload["resources"] = resources
    return cast(SSHSpecStrict, SSHSpecStrict.model_validate(payload))


class TestSSHConfigResolveLimits:
    def test_no_spec_no_cap_yields_unbounded(self) -> None:
        cfg = SSHConfig.from_spec(_spec(), make_worker_config())
        assert cfg.cpu_limit is None
        assert cfg.memory_limit_bytes is None
        assert cfg.pids_limit is None

    def test_spec_only(self) -> None:
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"cpu": 2, "memory": "4Gi"}}),
            make_worker_config(),
        )
        assert cfg.cpu_limit == 2.0
        assert cfg.memory_limit_bytes == 4 * 1024**3
        assert cfg.pids_limit is None

    def test_worker_cap_only(self) -> None:
        cfg = SSHConfig.from_spec(
            _spec(),
            make_worker_config(
                ssh_limits=SSHLimits(
                    max_cpu_cores=1.0, max_memory_bytes=2 * 1024**3, max_pids=128
                )
            ),
        )
        assert cfg.cpu_limit == 1.0
        assert cfg.memory_limit_bytes == 2 * 1024**3
        assert cfg.pids_limit == 128

    def test_spec_below_cap_uses_spec(self) -> None:
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"cpu": 1, "memory": "1Gi"}}),
            make_worker_config(
                ssh_limits=SSHLimits(max_cpu_cores=4.0, max_memory_bytes=8 * 1024**3)
            ),
        )
        assert cfg.cpu_limit == 1.0
        assert cfg.memory_limit_bytes == 1 * 1024**3

    def test_spec_above_cap_clamps_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="worker.executors.ssh_executor")
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"cpu": 8, "memory": "16Gi"}}),
            make_worker_config(
                ssh_limits=SSHLimits(max_cpu_cores=2.0, max_memory_bytes=4 * 1024**3)
            ),
        )
        assert cfg.cpu_limit == 2.0
        assert cfg.memory_limit_bytes == 4 * 1024**3
        messages = " ".join(rec.message for rec in caplog.records)
        assert "clamping to cap" in messages

    def test_numeric_memory_is_treated_as_bytes(self) -> None:
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"memory": 1048576}}),
            make_worker_config(),
        )
        assert cfg.memory_limit_bytes == 1048576

    def test_invalid_memory_string_raises(self) -> None:
        with pytest.raises(Exception, match="not a valid memory string"):
            SSHConfig.from_spec(
                _spec({"hardware": {"memory": "lots"}}),
                make_worker_config(),
            )


def _worker_config_gpu_limit(**overrides: Any) -> WorkerConfig:
    return make_worker_config(enable_ssh_gpu_limit=True, **overrides)


class TestSSHConfigResolveGpuDevices:
    def test_no_host_gpus_yields_empty_slice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WORKER_HOST_GPU_ID", raising=False)
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"count": 1}}}),
            make_worker_config(),
        )
        assert cfg.gpu_device_ids == []

    def test_disabled_flag_passes_all_host_gpus_despite_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "2,3,4,5")
        hardware = make_worker_hardware(
            [
                GpuInfo(
                    index=0, name="T4", uuid="t4-0", memory_total_bytes=16 * 1024**3
                ),
                GpuInfo(
                    index=1, name="A100", uuid="a100-0", memory_total_bytes=80 * 1024**3
                ),
                GpuInfo(
                    index=2, name="A100", uuid="a100-1", memory_total_bytes=80 * 1024**3
                ),
                GpuInfo(
                    index=3, name="A100", uuid="a100-2", memory_total_bytes=80 * 1024**3
                ),
            ]
        )
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"count": 1, "type": "A100", "memory": "40Gi"}}}),
            make_worker_config(),
            hardware=hardware,
        )
        assert cfg.gpu_device_ids == ["2", "3", "4", "5"]

    def test_disabled_flag_yields_empty_when_no_host_gpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WORKER_HOST_GPU_ID", raising=False)
        cfg = SSHConfig.from_spec(_spec(), make_worker_config())
        assert cfg.gpu_device_ids == []

    def test_no_gpu_spec_passes_all_worker_gpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "2,3")
        cfg = SSHConfig.from_spec(_spec(), _worker_config_gpu_limit())
        assert cfg.gpu_device_ids == ["2", "3"]

    def test_count_only_slices_first_n(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "2,3,4,5")
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"count": 2}}}),
            _worker_config_gpu_limit(),
        )
        assert cfg.gpu_device_ids == ["2", "3"]

    def test_type_filter_skips_non_matching_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0,1,2")
        hardware = make_worker_hardware(
            [
                GpuInfo(
                    index=0,
                    name="NVIDIA T4",
                    uuid="t4-0",
                    memory_total_bytes=16 * 1024**3,
                ),
                GpuInfo(
                    index=1,
                    name="NVIDIA A100-SXM4-80GB",
                    uuid="a100-0",
                    memory_total_bytes=80 * 1024**3,
                ),
                GpuInfo(
                    index=2,
                    name="NVIDIA A100-SXM4-80GB",
                    uuid="a100-1",
                    memory_total_bytes=80 * 1024**3,
                ),
            ]
        )
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"count": 2, "type": "A100"}}}),
            _worker_config_gpu_limit(),
            hardware=hardware,
        )
        assert cfg.gpu_device_ids == ["1", "2"]

    def test_memory_filter_skips_small_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0,1")
        hardware = make_worker_hardware(
            [
                GpuInfo(
                    index=0, name="T4", uuid="t4-0", memory_total_bytes=16 * 1024**3
                ),
                GpuInfo(
                    index=1, name="A100", uuid="a100-0", memory_total_bytes=80 * 1024**3
                ),
            ]
        )
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"count": 1, "memory": "40Gi"}}}),
            _worker_config_gpu_limit(),
            hardware=hardware,
        )
        assert cfg.gpu_device_ids == ["1"]

    def test_insufficient_matching_devices_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0,1")
        hardware = make_worker_hardware(
            [
                GpuInfo(
                    index=0, name="T4", uuid="t4-0", memory_total_bytes=16 * 1024**3
                ),
                GpuInfo(
                    index=1, name="T4", uuid="t4-1", memory_total_bytes=16 * 1024**3
                ),
            ]
        )
        with pytest.raises(Exception, match="GPU"):
            SSHConfig.from_spec(
                _spec({"hardware": {"gpu": {"count": 1, "type": "A100"}}}),
                _worker_config_gpu_limit(),
                hardware=hardware,
            )

    def test_count_zero_yields_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "2,3")
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"count": 0}}}),
            _worker_config_gpu_limit(),
        )
        assert cfg.gpu_device_ids == []

    def test_no_hardware_metadata_still_count_slices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without WorkerHardware, type / memory filters can't be evaluated;
        # count-only slicing still works as a graceful fallback.
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "2,3,4")
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"count": 2}}}),
            _worker_config_gpu_limit(),
        )
        assert cfg.gpu_device_ids == ["2", "3"]

    def test_unified_memory_satisfies_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # GB10 / GH200-style unified-memory worker: per-device memory is
        # unreported, but the shared pool covers the requested floor. The slice
        # should still go through.
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0")
        hardware = WorkerHardware(
            cpu=CPUInfo(logical_cores=8, model="x"),
            memory=MemoryInfo(total_bytes=128 * 1024**3),
            gpu=GpuPlatformInfo(
                driver_version=None,
                cuda_version=None,
                devices=[
                    GpuInfo(
                        index=0,
                        name="NVIDIA GB10",
                        uuid="gb10",
                        memory_total_bytes=None,
                    )
                ],
                memory_is_unified=True,
                shared_memory_total_bytes=128 * 1024**3,
            ),
            network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
        )
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"count": 1, "memory": "40Gi"}}}),
            _worker_config_gpu_limit(),
            hardware=hardware,
        )
        assert cfg.gpu_device_ids == ["0"]

    def test_type_filter_applies_when_count_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `count` omitted but `type` set: the dispatcher admits the worker on
        # one matching device; the slicer must restrict to that single device,
        # not pass through all worker GPUs.
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0,1")
        hardware = make_worker_hardware(
            [
                GpuInfo(
                    index=0,
                    name="NVIDIA T4",
                    uuid="t4-0",
                    memory_total_bytes=16 * 1024**3,
                ),
                GpuInfo(
                    index=1,
                    name="NVIDIA A100-SXM4-80GB",
                    uuid="a100-0",
                    memory_total_bytes=80 * 1024**3,
                ),
            ]
        )
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"type": "A100"}}}),
            _worker_config_gpu_limit(),
            hardware=hardware,
        )
        assert cfg.gpu_device_ids == ["1"]

    def test_memory_filter_applies_when_count_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0,1")
        hardware = make_worker_hardware(
            [
                GpuInfo(
                    index=0, name="T4", uuid="t4-0", memory_total_bytes=16 * 1024**3
                ),
                GpuInfo(
                    index=1, name="A100", uuid="a100-0", memory_total_bytes=80 * 1024**3
                ),
            ]
        )
        cfg = SSHConfig.from_spec(
            _spec({"hardware": {"gpu": {"memory": "40Gi"}}}),
            _worker_config_gpu_limit(),
            hardware=hardware,
        )
        assert cfg.gpu_device_ids == ["1"]

    def test_unified_memory_pool_too_small_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0")
        hardware = WorkerHardware(
            cpu=CPUInfo(logical_cores=8, model="x"),
            memory=MemoryInfo(total_bytes=16 * 1024**3),
            gpu=GpuPlatformInfo(
                driver_version=None,
                cuda_version=None,
                devices=[
                    GpuInfo(
                        index=0,
                        name="NVIDIA GB10",
                        uuid="gb10",
                        memory_total_bytes=None,
                    )
                ],
                memory_is_unified=True,
                shared_memory_total_bytes=16 * 1024**3,
            ),
            network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
        )
        with pytest.raises(Exception, match="GPU"):
            SSHConfig.from_spec(
                _spec({"hardware": {"gpu": {"count": 1, "memory": "40Gi"}}}),
                _worker_config_gpu_limit(),
                hardware=hardware,
            )
