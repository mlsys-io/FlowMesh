"""GPU metadata passed to rented Vast.ai worker containers."""

import shlex
from typing import Any
from unittest.mock import MagicMock

import pytest

from server.hooks import PrincipalContext
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.vastai import (
    VastAIWorkerAdapter,
    VastAIWorkerConfig,
    _offer_gpu_environment,
    offer_gpu_arch,
)
from server.utils.helpers import ResourcePool


def test_single_gpu_offer_exposes_device_zero_and_arch() -> None:
    arch = offer_gpu_arch("RTX 5080")
    env = _offer_gpu_environment({"num_gpus": 1, "gpu_name": "RTX 5080"}, arch)
    assert env["WORKER_HOST_GPU_ID"] == "0"
    assert env["WORKER_HOST_GPU_ARCH"] == "blackwell"


def test_multi_gpu_offer_lists_every_rented_device() -> None:
    env = _offer_gpu_environment({"num_gpus": 4}, None)
    assert env == {"WORKER_HOST_GPU_ID": "0,1,2,3"}


def test_gpuless_or_unknown_offer_sets_nothing() -> None:
    assert _offer_gpu_environment({"num_gpus": 0}, None) == {}
    assert _offer_gpu_environment({}, None) == {}
    assert _offer_gpu_environment(None, None) == {}
    assert _offer_gpu_environment({"num_gpus": "n/a"}, None) == {}


@pytest.mark.parametrize(
    ("gpu_fields", "expected_gpu_env", "image_variant"),
    [
        (
            {"num_gpus": 1, "gpu_name": "RTX 5080"},
            {"WORKER_HOST_GPU_ID": "0", "WORKER_HOST_GPU_ARCH": "blackwell"},
            "gpu",
        ),
        (
            {"num_gpus": 4, "gpu_name": "H200 NVL"},
            {"WORKER_HOST_GPU_ID": "0,1,2,3", "WORKER_HOST_GPU_ARCH": "hopper"},
            "gpu",
        ),
        ({"num_gpus": 0, "gpu_name": "N/A"}, {}, "cpu"),
        ({"gpu_name": "RTX 5080"}, {}, "gpu"),
        ({"num_gpus": "n/a", "gpu_name": "RTX 5080"}, {}, "gpu"),
    ],
)
def test_instance_creation_passes_offer_gpu_metadata(
    monkeypatch: pytest.MonkeyPatch,
    gpu_fields: dict[str, Any],
    expected_gpu_env: dict[str, str],
    image_variant: str,
) -> None:
    offer = {"id": 123, **gpu_fields}
    client = MagicMock()
    client.search_offers.return_value = [offer]
    client.create_instance.return_value = {"success": True, "new_contract": 456}
    client.show_instance.return_value = {"id": 456, **gpu_fields}
    worker = VastAIWorkerAdapter(
        token=WorkerTokenType("worker-token"),
        alias="vast-worker",
        config=VastAIWorkerConfig(docker_registry="test-registry", version="test"),
        vastai_client=client,
        instance_pool=ResourcePool[int](),
        owner=PrincipalContext(
            principal_id="test-user",
            org_id="test-org",
            external_id="test-user",
            principal_type="user",
            scopes=[],
        ),
    )
    monkeypatch.setattr(worker, "_base_environment", lambda: {"WORKER_TOKEN": "test"})

    assert worker._create_and_start()

    client.create_instance.assert_called_once()
    kwargs = client.create_instance.call_args.kwargs
    assert kwargs["id"] == 123
    assert kwargs["image"] == f"test-registry/flowmesh_worker:test-{image_variant}"
    tokens = shlex.split(kwargs["env"])
    assert tokens[::2] == ["-e"] * (len(tokens) // 2)
    environment = dict(token.split("=", 1) for token in tokens[1::2])
    assert environment == {"WORKER_TOKEN": "test", **expected_gpu_env}
