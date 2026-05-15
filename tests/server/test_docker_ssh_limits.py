"""Tests for the supervisor Docker adapter's SSH resource cap plumbing."""

import pytest

from server.supervisor.adapters.docker import SSHConfig


class TestSSHConfigToEnv:
    def test_omits_unset_limits(self) -> None:
        env = SSHConfig().to_env()
        assert "SSH_MAX_CPU" not in env
        assert "SSH_MAX_MEMORY" not in env
        assert "SSH_MAX_PIDS" not in env

    def test_emits_set_limits(self) -> None:
        env = SSHConfig(max_cpu=4.0, max_memory="8Gi", max_pids=512).to_env()
        assert env["SSH_MAX_CPU"] == "4.0"
        assert env["SSH_MAX_MEMORY"] == "8Gi"
        assert env["SSH_MAX_PIDS"] == "512"


class TestSSHConfigToLimits:
    def test_returns_none_when_unset(self) -> None:
        assert SSHConfig().to_limits() is None

    def test_parses_memory_string(self) -> None:
        limits = SSHConfig(max_cpu=2.0, max_memory="4Gi", max_pids=128).to_limits()
        assert limits is not None
        assert limits.max_cpu_cores == 2.0
        assert limits.max_memory_bytes == 4 * 1024**3
        assert limits.max_pids == 128

    def test_invalid_memory_raises(self) -> None:
        with pytest.raises(ValueError, match="SSH_MAX_MEMORY"):
            SSHConfig(max_memory="garbage").to_limits()

    def test_partial_caps(self) -> None:
        limits = SSHConfig(max_cpu=1.5).to_limits()
        assert limits is not None
        assert limits.max_cpu_cores == 1.5
        assert limits.max_memory_bytes is None
        assert limits.max_pids is None
