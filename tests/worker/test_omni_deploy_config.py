"""Omni deploy-config / stage-override passthrough to ``vllm_omni.Omni``.

vllm-omni 0.28 dropped ``stage_configs_path`` in favor of ``deploy_config`` and
per-stage ``stage_overrides``; these guard that the executor forwards the new
knobs and keys the reuse spec on them.
"""

from pathlib import Path
from typing import Any

import pytest

from worker.executors.base_executor import ExecutionError
from worker.executors.omni_executor_base import OmniExecutorBase
from worker.executors.omni_text2general_executor import OmniText2GeneralExecutor


def _executor() -> OmniText2GeneralExecutor:
    executor = OmniText2GeneralExecutor.__new__(OmniText2GeneralExecutor)
    executor._deploy_config_tmp = None
    return executor


def test_init_kwargs_forward_stage_overrides_with_string_keys() -> None:
    cfg: dict[str, Any] = {
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
    executor._cleanup_deploy_config_tmp()
    assert not path.exists()


def test_reuse_spec_distinguishes_stage_overrides() -> None:
    base = {"stage_overrides": {"1": {"devices": "0"}}}
    other = {"stage_overrides": {"1": {"devices": "1"}}}

    assert OmniExecutorBase._build_omni_spec(
        "m", base
    ) != OmniExecutorBase._build_omni_spec("m", other)
