"""Tests for server worker configuration models."""

import pytest
from pydantic import ValidationError

from server.supervisor.adapters.ssh import SSHConfig
from server.supervisor.adapters.utils import get_worker_image_name
from server.supervisor.adapters.vastai import VastAIWorkerConfig, offer_gpu_arch
from server.supervisor.manager import ServerWorkerConfig, WorkerInitConfig
from server.supervisor.resource_manager import GpuArch
from shared.schemas.worker import SSHBackendName


class TestWorkerInitConfig:
    def test_defaults(self) -> None:
        cfg = WorkerInitConfig()
        assert cfg.provider == "docker"
        assert cfg.init_on_start is True
        assert cfg.worker_config == {}

    def test_custom_provider(self) -> None:
        cfg = WorkerInitConfig(provider="vastai", init_on_start=False)
        assert cfg.provider == "vastai"
        assert cfg.init_on_start is False

    def test_extra_fields_preserved(self) -> None:
        cfg = WorkerInitConfig(  # type: ignore[call-arg]
            provider="docker",
            worker_config={"image": "my-image:latest"},
            custom_field="custom_value",
        )
        extras = cfg.extra_kwargs
        assert "custom_field" in extras
        assert extras["custom_field"] == "custom_value"
        assert "provider" not in extras
        assert "worker_config" not in extras

    def test_worker_config_nested(self) -> None:
        cfg = WorkerInitConfig(
            worker_config={
                "image": "flowmesh_worker:gpu",
                "gpu_count": 4,
                "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            }
        )
        assert cfg.worker_config["gpu_count"] == 4


class TestServerWorkerConfig:
    def test_defaults(self) -> None:
        cfg = ServerWorkerConfig()
        assert cfg.default_worker_config == {}
        assert cfg.workers == []

    def test_with_workers(self) -> None:
        cfg = ServerWorkerConfig(
            default_worker_config={"tags": "gpu"},
            workers=[
                WorkerInitConfig(provider="docker"),
                WorkerInitConfig(provider="vastai", init_on_start=False),
            ],
        )
        assert len(cfg.workers) == 2
        assert cfg.workers[0].provider == "docker"
        assert cfg.workers[1].provider == "vastai"


class TestVastAISessionBackend:
    """A VastAI instance is the worker container and exposes no Docker socket."""

    def test_docker_backend_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be 'docker'"):
            VastAIWorkerConfig(ssh=SSHConfig(session_backend=SSHBackendName.DOCKER))

    def test_rejection_ignores_case_and_padding(self) -> None:
        with pytest.raises(ValidationError, match="cannot be 'docker'"):
            VastAIWorkerConfig(
                ssh=SSHConfig.model_validate({"session_backend": "  Docker  "})
            )

    def test_process_backend_is_accepted(self) -> None:
        cfg = VastAIWorkerConfig(ssh=SSHConfig(session_backend=SSHBackendName.PROCESS))
        assert cfg.ssh.session_backend is SSHBackendName.PROCESS

    def test_auto_is_accepted(self) -> None:
        cfg = VastAIWorkerConfig(ssh=SSHConfig(session_backend=SSHBackendName.AUTO))
        assert cfg.ssh.session_backend is SSHBackendName.AUTO

    def test_unset_backend_is_accepted(self) -> None:
        VastAIWorkerConfig(ssh=SSHConfig(session_backend=None))


class TestVastAIImageSelection:
    """VastAI names a GPU-less offer rather than omitting the field."""

    @pytest.mark.parametrize("gpu_name", ["N/A", "n/a", " ", "", None, "None"])
    def test_a_gpuless_offer_selects_the_cpu_image(self, gpu_name: str | None) -> None:
        arch = offer_gpu_arch(gpu_name)
        assert arch is None
        assert get_worker_image_name("reg", "v1", arch).endswith("-cpu")

    @pytest.mark.parametrize(
        ("gpu_name", "expected"),
        [
            ("RTX 4090", GpuArch.UNKNOWN),
            ("H100 SXM", GpuArch.HOPPER),
            ("B200", GpuArch.BLACKWELL),
        ],
    )
    def test_a_gpu_offer_selects_the_gpu_image(
        self, gpu_name: str, expected: GpuArch
    ) -> None:
        arch = offer_gpu_arch(gpu_name)
        assert arch is expected
        assert get_worker_image_name("reg", "v1", arch).endswith("-gpu")
