"""Choosing a mode the session can actually be reached in.

The worker binds sshd before the server decides how the session will be served.
A relayed session binds loopback, so a mode that hands the user an address is
not available for it.
"""

import logging
from typing import Any
from unittest.mock import MagicMock

from server.services.monitoring import EventMonitor


def _make_monitor(
    port_forward: MagicMock | None = None,
    ssh_proxy_enabled: bool = False,
) -> EventMonitor:
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test.monitoring.ssh_bind"),
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
        ssh_proxy_enabled=ssh_proxy_enabled,
        serve_proxy_enabled=False,
        port_forward=port_forward,
        server_base_url="http://server.example.com:8000",
    )


def _ssh_payload(mode: str, bind_host: str) -> dict[str, Any]:
    return {
        "ssh": {
            "session_id": "ssn-1",
            "mode": mode,
            "username": "flowmesh",
            "host": "worker.example.com",
            "port": 2222,
            "_bind_host": bind_host,
            "directHost": "worker.example.com",
            "directPort": 2222,
        }
    }


class TestLoopbackBoundSession:
    def test_a_proxy_session_is_failed_rather_than_demoted(self) -> None:
        """Demoting would advertise a port that refuses every connection."""
        monitor = _make_monitor(ssh_proxy_enabled=False)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("proxy", "127.0.0.1")
        )

        assert "ssh" not in result

    def test_a_forward_session_without_a_service_is_failed(self) -> None:
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=False)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "127.0.0.1")
        )

        assert "ssh" not in result

    def test_proxy_is_preferred_over_failing_when_available(self) -> None:
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=True)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "127.0.0.1")
        )

        assert result["ssh"]["mode"] == "proxy"

    def test_a_dead_direct_route_is_not_advertised(self) -> None:
        """`forward` publishes directHost/directPort as a second route."""
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "session_id": "ssn-1",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_bind_host": "127.0.0.1",
            "directHost": "worker.example.com",
            "directPort": 2222,
        }
        monitor = _make_monitor(port_forward=port_forward, ssh_proxy_enabled=True)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "127.0.0.1")
        )

        assert "directHost" not in result["ssh"]
        assert "directPort" not in result["ssh"]


class TestRoutableSession:
    def test_a_routable_session_still_demotes_to_direct(self) -> None:
        """A docker session is reachable, so the old fallback is unchanged."""
        monitor = _make_monitor(ssh_proxy_enabled=False)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("proxy", "0.0.0.0")
        )

        assert result["ssh"]["mode"] == "direct"

    def test_a_routable_forward_session_keeps_its_direct_route(self) -> None:
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "session_id": "ssn-1",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_bind_host": "0.0.0.0",
            "directHost": "worker.example.com",
            "directPort": 2222,
        }
        monitor = _make_monitor(port_forward=port_forward, ssh_proxy_enabled=True)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "0.0.0.0")
        )

        assert result["ssh"]["directHost"] == "worker.example.com"
