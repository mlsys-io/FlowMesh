"""The echo executor's optional in-flight delay knob (spec.data.delay_sec)."""

import time

import pytest

from worker.executors.base_executor import ExecutionError
from worker.executors.echo_executor import EchoExecutor


def test_delay_none_is_instant() -> None:
    start = time.monotonic()
    EchoExecutor._maybe_delay(None)
    assert time.monotonic() - start < 0.5


def test_delay_zero_or_negative_is_instant() -> None:
    start = time.monotonic()
    EchoExecutor._maybe_delay(0)
    EchoExecutor._maybe_delay(-5)
    assert time.monotonic() - start < 0.5


def test_delay_sleeps_for_positive_value() -> None:
    start = time.monotonic()
    EchoExecutor._maybe_delay(0.2)
    assert time.monotonic() - start >= 0.2


def test_delay_rejects_non_numeric() -> None:
    with pytest.raises(ExecutionError):
        EchoExecutor._maybe_delay("slow")
    with pytest.raises(ExecutionError):
        EchoExecutor._maybe_delay(True)
