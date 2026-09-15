"""Tests for the retryable classification of vLLM engine-init failures.
A memory-constrained load failure is transient and must be retryable;
deterministic failures must fail fast. The engine is mocked, so no GPU or
model download is needed.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from worker.executors.base_executor import ExecutionError
from worker.executors.vllm_executor import VLLMExecutor

MEMORY_CONSTRAINED_FREE_RATIO = 0.04
AMPLE_FREE_RATIO = 0.99


def _executor() -> VLLMExecutor:
    return VLLMExecutor(cast(Any, SimpleNamespace()), lifecycle=None)


def _call_init(executor: VLLMExecutor, *, free_ratio: float | None) -> None:
    """Drive ``_init_vllm_engine`` with a mocked memory signal and a failing LLM."""
    requested = 0.9

    def _safe_util(requested_util: float) -> tuple[float, float | None]:
        if free_ratio is None:
            return requested_util, None
        if free_ratio - 0.05 >= requested_util:
            return requested_util, free_ratio
        return max(0.02, free_ratio * 0.8), free_ratio

    with (
        patch.object(VLLMExecutor, "_compute_safe_utilization", side_effect=_safe_util),
        patch("worker.executors.vllm_executor.LLM", side_effect=RuntimeError("boom")),
    ):
        executor._init_vllm_engine(
            ident="org/model",
            vllm_cfg={"gpu_memory_utilization": requested},
            checkpoint_cfg={},
            new_inference_spec={},
            requested_gpu_count=1,
            revision=None,
            extra_llm_kwargs={},
            adjust_tp=lambda size: size,
            task_ids=None,
        )


def test_memory_constrained_failure_is_retryable() -> None:
    executor = _executor()
    with pytest.raises(ExecutionError) as excinfo:
        _call_init(executor, free_ratio=MEMORY_CONSTRAINED_FREE_RATIO)
    assert excinfo.value.retryable is True


def test_unconstrained_failure_is_not_retryable() -> None:
    executor = _executor()
    with pytest.raises(ExecutionError) as excinfo:
        _call_init(executor, free_ratio=None)
    assert excinfo.value.retryable is False


def test_plenty_free_but_failure_is_not_retryable() -> None:
    executor = _executor()
    with pytest.raises(ExecutionError) as excinfo:
        _call_init(executor, free_ratio=AMPLE_FREE_RATIO)
    assert excinfo.value.retryable is False
