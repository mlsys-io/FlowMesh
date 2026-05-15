"""Tests for SSH container resource-limit resolution and propagation."""

import logging
from typing import cast

import pytest

from shared.schemas.worker import SSHLimits
from shared.tasks.specs import SSHSpecStrict
from tests.worker.factories import make_worker_config
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
