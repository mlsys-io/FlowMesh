"""Tests for server GPU resource management and worker capacity reporting."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from server import env
from server.hooks import PrincipalContext
from server.supervisor.adapters.docker import (
    DockerWorkerAdapter,
    DockerWorkerConfig,
    WorkerType,
)
from server.supervisor.manager import WorkerInitConfig, WorkerManager
from server.supervisor.resource_manager import GpuArch, MachineEnv, ResourceManager

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _resource_manager(available: set[int]) -> ResourceManager:
    """Build a ResourceManager with a fixed set of available GPU indices."""
    rm = object.__new__(ResourceManager)
    rm._env = MachineEnv(
        cpu_count=16,
        gpu_families={i: GpuArch.UNKNOWN for i in available},
        available_gpus=set(available),
    )
    return rm


def _worker_manager() -> WorkerManager:
    """Construct a WorkerManager in started state without filesystem or Docker."""
    wm = object.__new__(WorkerManager)
    wm.config_path = "/dev/null"
    wm.logger = logging.getLogger("test-wm")
    wm._registry = MagicMock()
    wm._is_started = True
    wm._default_worker_config = {}
    wm._capacity_change_callback = None
    return wm


# ------------------------------------------------------------------ #
# ResourceManager.reserve_gpus
# ------------------------------------------------------------------ #


class TestReserveGpusByCount:
    def test_single_gpu_returns_lowest_index_and_reserves(self) -> None:
        rm = _resource_manager({0, 1, 2, 3})
        devices, _ = rm.reserve_gpus(n=1)
        assert devices == [0]
        assert rm.available_gpu_count == 3
        assert 0 not in rm._env.available_gpus

    def test_single_gpu_picks_minimum_when_indices_nonzero(self) -> None:
        rm = _resource_manager({2, 3})
        devices, _ = rm.reserve_gpus(n=1)
        assert devices == [2]

    def test_two_gpus_returns_two_lowest_sorted(self) -> None:
        rm = _resource_manager({0, 1, 2, 3})
        devices, _ = rm.reserve_gpus(n=2)
        assert devices == [0, 1]
        assert rm.available_gpu_count == 2

    def test_two_gpus_sparse_indices(self) -> None:
        rm = _resource_manager({1, 3})
        devices, _ = rm.reserve_gpus(n=2)
        assert devices == [1, 3]

    def test_four_gpus_returns_all_sorted(self) -> None:
        rm = _resource_manager({0, 1, 2, 3})
        devices, _ = rm.reserve_gpus(n=4)
        assert devices == [0, 1, 2, 3]
        assert rm.available_gpu_count == 0

    def test_requesting_more_than_available_raises(self) -> None:
        rm = _resource_manager({0, 1})
        with pytest.raises(ValueError, match="Not enough available GPUs"):
            rm.reserve_gpus(n=3)
        assert rm.available_gpu_count == 2

    def test_zero_raises(self) -> None:
        rm = _resource_manager({0, 1})
        with pytest.raises(ValueError, match="Invalid number of GPUs"):
            rm.reserve_gpus(n=0)

    def test_negative_raises(self) -> None:
        rm = _resource_manager({0, 1})
        with pytest.raises(ValueError, match="Invalid number of GPUs"):
            rm.reserve_gpus(n=-1)


class TestReserveGpusByDevices:
    def test_explicit_devices_reserve_and_remove(self) -> None:
        rm = _resource_manager({0, 1, 2, 3})
        devices, _ = rm.reserve_gpus(devices=[1, 2])
        assert devices == [1, 2]
        assert rm._env.available_gpus == {0, 3}

    def test_unavailable_device_raises(self) -> None:
        rm = _resource_manager({0, 1})
        with pytest.raises(ValueError, match="Requested GPUs are not available"):
            rm.reserve_gpus(devices=[5])
        assert rm.available_gpu_count == 2

    def test_partially_unavailable_raises_without_partial_reserve(self) -> None:
        rm = _resource_manager({0, 1})
        with pytest.raises(ValueError, match="Requested GPUs are not available"):
            rm.reserve_gpus(devices=[0, 5])
        assert rm._env.available_gpus == {0, 1}

    def test_empty_device_list_raises(self) -> None:
        rm = _resource_manager({0, 1})
        with pytest.raises(ValueError, match="Empty device list"):
            rm.reserve_gpus(devices=[])


class TestReserveGpusArgs:
    def test_neither_arg_raises(self) -> None:
        rm = _resource_manager({0, 1})
        with pytest.raises(ValueError, match="exactly one of n or devices"):
            rm.reserve_gpus()

    def test_both_args_raises(self) -> None:
        rm = _resource_manager({0, 1})
        with pytest.raises(ValueError, match="exactly one of n or devices"):
            rm.reserve_gpus(n=1, devices=[0])


class TestReserveGpusAtomicity:
    def test_repeated_calls_yield_disjoint_devices(self) -> None:
        rm = _resource_manager({0, 1, 2, 3})
        d1, _ = rm.reserve_gpus(n=2)
        d2, _ = rm.reserve_gpus(n=2)
        assert set(d1).isdisjoint(d2)
        assert sorted(d1 + d2) == [0, 1, 2, 3]
        assert rm.available_gpu_count == 0

    def test_mixed_arch_raises(self) -> None:
        rm = object.__new__(ResourceManager)
        rm._env = MachineEnv(
            cpu_count=16,
            gpu_families={0: GpuArch.HOPPER, 1: GpuArch.BLACKWELL},
            available_gpus={0, 1},
        )
        with pytest.raises(ValueError, match="different architectures"):
            rm.reserve_gpus(devices=[0, 1])
        # No partial reservation on failure.
        assert rm._env.available_gpus == {0, 1}


class TestAvailableGpuCount:
    def test_four_gpus(self) -> None:
        assert _resource_manager({0, 1, 2, 3}).available_gpu_count == 4

    def test_two_gpus(self) -> None:
        assert _resource_manager({0, 1}).available_gpu_count == 2

    def test_no_gpus(self) -> None:
        assert _resource_manager(set()).available_gpu_count == 0


class TestMachineEnvDetection:
    def test_gpu_probe_uses_configured_cuda_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rm = object.__new__(ResourceManager)
        containers = MagicMock()
        containers.run.return_value = b"0, NVIDIA H100\n"
        rm._docker_client = MagicMock()
        rm._docker_client.info.return_value = {"NCPU": 32}
        rm._docker_client.containers = containers

        monkeypatch.setattr(env, "SERVER_CUDA_PROBE_IMAGE", "example/probe:arm64")
        monkeypatch.setattr(env, "CUDA_VISIBLE_DEVICES", None)

        detected = rm._detect_machine_env()

        assert detected.cpu_count == 32
        assert detected.available_gpus == {0}
        assert detected.gpu_families == {0: GpuArch.HOPPER}
        containers.run.assert_called_once()
        assert containers.run.call_args.kwargs["image"] == "example/probe:arm64"
        assert "runtime" not in containers.run.call_args.kwargs

    def test_gpu_probe_uses_runtime_override_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rm = object.__new__(ResourceManager)
        containers = MagicMock()
        containers.run.return_value = b"0, NVIDIA H100\n"
        rm._docker_client = MagicMock()
        rm._docker_client.info.return_value = {"NCPU": 32}
        rm._docker_client.containers = containers

        monkeypatch.setattr(env, "SERVER_CUDA_PROBE_IMAGE", "example/probe:legacy")
        monkeypatch.setattr(env, "DOCKER_GPU_RUNTIME", "nvidia")
        monkeypatch.setattr(env, "CUDA_VISIBLE_DEVICES", None)

        detected = rm._detect_machine_env()

        assert detected.available_gpus == {0}
        assert containers.run.call_args.kwargs["runtime"] == "nvidia"


class TestDockerWorkerRuntimeSelection:
    def _worker(self) -> DockerWorkerAdapter:
        worker = object.__new__(DockerWorkerAdapter)
        worker.config = DockerWorkerConfig(worker_type=WorkerType.GPU, cuda_devices=[3])
        worker.token = "worker-token"  # type: ignore[assignment]
        worker.owner = PrincipalContext(
            principal_id="test-user",
            org_id="test-org",
            external_id="test-user",
            principal_type="user",
            scopes=[],
        )
        worker.container_name = "worker-gpu-3"
        worker.cuda_devices = [3]
        worker.gpu_arch = GpuArch.BLACKWELL
        return worker

    def test_gpu_worker_omits_runtime_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = self._worker()
        monkeypatch.setattr(env, "DOCKER_GPU_RUNTIME", None)

        environment: dict[str, str] = {}
        labels: dict[str, str] = {}

        device_requests, runtime = worker._apply_worker_type_settings(
            environment, labels
        )

        assert runtime is None
        assert device_requests is not None
        assert environment["CUDA_VISIBLE_DEVICES"] == "0"
        assert environment["WORKER_HOST_GPU_ID"] == "3"
        assert environment["WORKER_HOST_GPU_ARCH"] == GpuArch.BLACKWELL.value
        assert labels["flowmesh.worker.gpu_id"] == "3"

    def test_gpu_worker_uses_runtime_override_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = self._worker()
        monkeypatch.setattr(env, "DOCKER_GPU_RUNTIME", "nvidia")

        device_requests, runtime = worker._apply_worker_type_settings({}, {})

        assert runtime == "nvidia"
        assert device_requests is not None

    def test_worker_environment_passes_runtime_override_to_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = self._worker()
        monkeypatch.setattr(env, "DOCKER_GPU_RUNTIME", "nvidia")

        environment = worker._base_environment()

        assert environment["DOCKER_GPU_RUNTIME"] == "nvidia"


class TestCapacityChangeReporting:
    def _run(self, coro: object) -> object:  # type: ignore[return]
        return asyncio.run(coro)  # type: ignore[arg-type]

    def test_create_worker_reports_capacity_change(self) -> None:
        wm = _worker_manager()
        callback = MagicMock()
        wm._capacity_change_callback = callback

        worker = MagicMock()
        info = MagicMock()
        worker.name = "w-1"
        worker.get_info.return_value = info
        wm._create_worker = MagicMock(return_value=worker)  # type: ignore[method-assign]
        wm._start_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        result = self._run(
            wm.create_worker(
                WorkerInitConfig(
                    provider="docker", init_on_start=True, worker_config={}
                )
            )
        )

        assert result is info
        callback.assert_called_once_with()

    def test_destroy_worker_reports_capacity_change(self) -> None:
        wm = _worker_manager()
        callback = MagicMock()
        wm._capacity_change_callback = callback

        worker = MagicMock()
        worker.name = "w-1"
        wm._registry.try_get_by_name.return_value = worker  # type: ignore[attr-defined]
        wm._registry.try_pop_by_name = MagicMock()  # type: ignore[attr-defined, method-assign]
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        result = self._run(wm.destroy_worker("w-1"))

        assert result is True
        callback.assert_called_once_with()
