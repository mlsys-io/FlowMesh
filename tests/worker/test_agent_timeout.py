"""Unit tests for AgentExecutor's spec.agent.timeout resolution."""

import pytest

from worker.executors.agent_executor import (
    DEFAULT_AGENT_TASK_TIMEOUT_SEC,
    _resolve_task_timeout,
)
from worker.executors.base_executor import ExecutionError


def test_default_timeout_when_agent_dict_is_none() -> None:
    assert _resolve_task_timeout(None) == DEFAULT_AGENT_TASK_TIMEOUT_SEC


def test_default_timeout_when_agent_dict_omits_timeout() -> None:
    assert _resolve_task_timeout({}) == DEFAULT_AGENT_TASK_TIMEOUT_SEC


def test_default_timeout_when_value_is_explicit_none() -> None:
    assert _resolve_task_timeout({"timeout": None}) == DEFAULT_AGENT_TASK_TIMEOUT_SEC


def test_int_timeout_passes_through() -> None:
    assert _resolve_task_timeout({"timeout": 42}) == 42


@pytest.mark.parametrize("bad", [0, -1, "10", True, False, [600], 12.9, 600.0])
def test_invalid_timeout_raises(bad: object) -> None:
    with pytest.raises(ExecutionError):
        _resolve_task_timeout({"timeout": bad})
