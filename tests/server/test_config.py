"""Tests for server environment configuration."""

import pytest

from server.config import PortForwardConfig


def test_port_forward_config_enables_capabilities_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ENABLE_PERSISTENT_PORT_FORWARD", raising=False)
    monkeypatch.delenv("ENABLE_SERVER_SSH_PROXY", raising=False)
    monkeypatch.delenv("ENABLE_SERVER_SERVE_PROXY", raising=False)

    config = PortForwardConfig.from_env()

    assert config.persistent_listeners is True
    assert config.ssh_proxy_enabled is True
    assert config.serve_proxy_enabled is True


def test_port_forward_config_reads_persistent_listener_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_PERSISTENT_PORT_FORWARD", "false")

    config = PortForwardConfig.from_env()

    assert config.persistent_listeners is False


@pytest.mark.parametrize(
    ("ssh_proxy", "serve_proxy"),
    [("false", "true"), ("true", "false")],
)
def test_port_forward_config_reads_proxy_capabilities_independently(
    monkeypatch: pytest.MonkeyPatch,
    ssh_proxy: str,
    serve_proxy: str,
) -> None:
    monkeypatch.setenv("ENABLE_SERVER_SSH_PROXY", ssh_proxy)
    monkeypatch.setenv("ENABLE_SERVER_SERVE_PROXY", serve_proxy)

    config = PortForwardConfig.from_env()

    assert config.ssh_proxy_enabled is (ssh_proxy == "true")
    assert config.serve_proxy_enabled is (serve_proxy == "true")
