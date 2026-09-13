"""Choosing a mode the session can actually be reached in.

The worker advertises the address it listens on, so a relayed session reports
loopback. `direct` therefore stays available as the last fallback: on the
worker's own machine that address is reachable, and reporting it beats
discarding a session the client may still be able to use.
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


def _ssh_payload(mode: str, host: str) -> dict[str, Any]:
    return {
        "ssh": {
            "session_id": "ssn-1",
            "mode": mode,
            "username": "flowmesh",
            "host": host,
            "port": 2222,
            "directHost": host,
            "directPort": 2222,
        }
    }


class TestLoopbackBoundSession:
    def test_a_proxy_session_degrades_to_its_loopback_address(self) -> None:
        monitor = _make_monitor(ssh_proxy_enabled=False)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("proxy", "127.0.0.1")
        )

        assert result["ssh"]["mode"] == "direct"
        assert result["ssh"]["host"] == "127.0.0.1"
        assert result["ssh"]["port"] == 2222
        monitor._dispatcher.fail_task.assert_not_called()  # type: ignore[attr-defined]

    def test_a_forward_session_without_a_service_degrades(self) -> None:
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=False)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "127.0.0.1")
        )

        assert result["ssh"]["mode"] == "direct"
        assert result["ssh"]["host"] == "127.0.0.1"
        monitor._dispatcher.fail_task.assert_not_called()  # type: ignore[attr-defined]

    def test_proxy_is_preferred_over_degrading_when_available(self) -> None:
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=True)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "127.0.0.1")
        )

        assert result["ssh"]["mode"] == "proxy"

    def test_a_loopback_direct_route_is_still_advertised(self) -> None:
        """`forward` publishes directHost/directPort as a second route."""
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "session_id": "ssn-1",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "directHost": "127.0.0.1",
            "directPort": 2222,
        }
        monitor = _make_monitor(port_forward=port_forward, ssh_proxy_enabled=True)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "127.0.0.1")
        )

        assert result["ssh"]["host"] == "server.example.com"
        assert result["ssh"]["directHost"] == "127.0.0.1"
        assert result["ssh"]["directPort"] == 2222

    def test_a_failed_registration_degrades_instead_of_failing_the_task(self) -> None:
        port_forward = MagicMock()
        port_forward.register_port_forward.side_effect = RuntimeError("no ports left")
        monitor = _make_monitor(port_forward=port_forward, ssh_proxy_enabled=False)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "127.0.0.1")
        )

        assert result["ssh"]["mode"] == "direct"
        assert result["ssh"]["host"] == "127.0.0.1"
        monitor._dispatcher.fail_task.assert_not_called()  # type: ignore[attr-defined]


class TestRoutableSession:
    def test_a_routable_session_degrades_to_its_routable_address(self) -> None:
        monitor = _make_monitor(ssh_proxy_enabled=False)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("proxy", "worker.example.com")
        )

        assert result["ssh"]["mode"] == "direct"
        assert result["ssh"]["host"] == "worker.example.com"

    def test_a_routable_forward_session_keeps_its_direct_route(self) -> None:
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "session_id": "ssn-1",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "directHost": "worker.example.com",
            "directPort": 2222,
        }
        monitor = _make_monitor(port_forward=port_forward, ssh_proxy_enabled=True)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_ssh_task_update(
            "tsk-abc", "wrk-1", _ssh_payload("forward", "worker.example.com")
        )

        assert result["ssh"]["directHost"] == "worker.example.com"
