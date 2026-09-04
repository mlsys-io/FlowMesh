"""build_omni_init_kwargs forwards deploy_config / stage_overrides to Omni.

``deploy_config`` is a deploy-YAML path or an inline mapping; ``stage_overrides``
is a per-stage mapping (keys coerced to str) or a JSON string. Any other type is
rejected rather than silently dropped.
"""

from pathlib import Path

import pytest

from worker.executors.base_executor import ExecutionError
from worker.executors.omni_text2general_executor import OmniText2GeneralExecutor

from .factories import DEFAULT_WORKER_CONFIG


def _executor() -> OmniText2GeneralExecutor:
    return OmniText2GeneralExecutor(DEFAULT_WORKER_CONFIG)


def test_init_kwargs_forward_stage_overrides_with_string_keys() -> None:
    cfg = {
        "deploy_config": "/opt/deploy.yaml",
        "stage_overrides": {0: {"devices": "0"}, "1": {"gpu_memory_utilization": 0.2}},
        "async_chunk": True,
        "stage_init_timeout": 1800,
        "init_timeout": 1800,
    }

    kwargs = _executor().build_omni_init_kwargs("Qwen/Qwen3-Omni", cfg)

    assert kwargs["model"] == "Qwen/Qwen3-Omni"
    assert kwargs["deploy_config"] == "/opt/deploy.yaml"
    assert kwargs["stage_overrides"] == {
        "0": {"devices": "0"},
        "1": {"gpu_memory_utilization": 0.2},
    }
    assert kwargs["async_chunk"] is True
    assert kwargs["stage_init_timeout"] == 1800
    assert kwargs["init_timeout"] == 1800


def test_init_kwargs_omit_absent_knobs() -> None:
    kwargs = _executor().build_omni_init_kwargs("Qwen/Qwen3-Omni", {})

    assert kwargs == {"model": "Qwen/Qwen3-Omni"}


def test_ill_typed_knobs_fail_fast_rather_than_silently_drop() -> None:
    executor = _executor()

    with pytest.raises(ExecutionError):
        executor.build_omni_init_kwargs("m", {"deploy_config": ["not", "a", "path"]})
    with pytest.raises(ExecutionError):
        executor.build_omni_init_kwargs("m", {"stage_overrides": 3})


def test_inline_deploy_config_is_materialized_to_yaml() -> None:
    executor = _executor()
    cfg = {"deploy_config": {"async_chunk": True, "stages": [{"stage_id": 0}]}}

    kwargs = executor.build_omni_init_kwargs("Qwen/Qwen3-Omni", cfg)
    path = Path(kwargs["deploy_config"])

    assert path.is_file() and path.suffix == ".yaml"
    executor.teardown()
    assert not path.exists()
