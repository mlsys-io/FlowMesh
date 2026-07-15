"""Serve TASK_UPDATE forward-registration tests for EventMonitor.

Regression coverage for the gap where a serve task with accessMode=forward
would never get a server-allocated client-facing port because
PortForwardService.register_port_forward requires a session_id that serve
payloads don't have.
"""

import logging
from unittest.mock import MagicMock

from server.services.monitoring import EventMonitor


def _make_monitor(port_forward: MagicMock | None = None) -> EventMonitor:
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test.monitoring.serve_forward"),
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
        ssh_proxy_enabled=False,
        port_forward=port_forward,
    )


def _serve_forward_payload(host: str = "127.0.0.1", port: int = 8000) -> dict:
    return {
        "serve": {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": host,
            "port": port,
            "_relay_target": {"host": host, "port": port},
        }
    }


class TestServeForwardRegistration:
    def test_forward_mode_calls_register_port_forward(self) -> None:
        """A serve TASK_UPDATE with mode=forward triggers register_port_forward."""
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        port_forward.register_port_forward.assert_called_once()

    def test_forward_mode_injects_task_id_as_session_id(self) -> None:
        """task_id is passed as session_id so the relay service's logging works."""
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        call_args = port_forward.register_port_forward.call_args
        endpoint_arg = call_args.args[3]
        assert endpoint_arg.get("session_id") == "tsk-abc"

    def test_forward_mode_surfaces_server_allocated_port(self) -> None:
        """The returned payload carries the server's public host/port, not the
        worker-internal endpoint."""
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert result["serve"]["host"] == "server.example.com"
        assert result["serve"]["port"] == 32001

    def test_forward_mode_strips_injected_session_id_from_result(self) -> None:
        """The synthetic session_id must not appear in latest_update.serve."""
        port_forward = MagicMock()
        # Simulate register_port_forward echoing back the injected session_id
        port_forward.register_port_forward.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "session_id": "tsk-abc",
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert "session_id" not in result["serve"]

    def test_forward_mode_strips_relay_target_from_result(self) -> None:
        """_relay_target must not appear in latest_update.serve after registration."""
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert "_relay_target" not in result["serve"]

    def test_direct_mode_does_not_call_register(self) -> None:
        """A serve update with mode=direct leaves the payload unchanged."""
        port_forward = MagicMock()
        monitor = _make_monitor(port_forward=port_forward)

        payload = {
            "serve": {
                "model": "m",
                "mode": "direct",
                "host": "127.0.0.1",
                "port": 8000,
            }
        }
        result = monitor._handle_serve_task_update("tsk-abc", "wrk-1", payload)

        port_forward.register_port_forward.assert_not_called()
        assert result["serve"]["host"] == "127.0.0.1"
        assert result["serve"]["port"] == 8000

    def test_direct_mode_preserves_worker_resolvable_host(self) -> None:
        """The worker-advertised resolvable host is surfaced to clients as-is;
        direct mode is not relayed through the server."""
        port_forward = MagicMock()
        monitor = _make_monitor(port_forward=port_forward)

        payload = {
            "serve": {
                "model": "m",
                "mode": "direct",
                "host": "worker-1.cluster.local",
                "port": 8000,
            }
        }
        result = monitor._handle_serve_task_update("tsk-abc", "wrk-1", payload)

        port_forward.register_port_forward.assert_not_called()
        assert result["serve"]["host"] == "worker-1.cluster.local"
        assert result["serve"]["port"] == 8000

    def test_no_forward_service_drops_serve_endpoint(self) -> None:
        """When forward is None, forward mode is rejected and the serve endpoint
        is dropped rather than degraded to direct (serve has no proxy fallback)."""
        monitor = _make_monitor(port_forward=None)

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert "serve" not in result

    def test_no_forward_service_fails_task(self) -> None:
        """With no way to serve the endpoint, the task must be failed instead of
        left DISPATCHED with an executor running to no purpose until its TTL."""
        monitor = _make_monitor(port_forward=None)

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        monitor._dispatcher.fail_task.assert_called_once()  # type: ignore[attr-defined]
        call_args = monitor._dispatcher.fail_task.call_args  # type: ignore[attr-defined]
        assert call_args.args[0] == "tsk-abc"
        assert call_args.kwargs["worker_id"] == "wrk-1"

    def test_register_port_forward_failure_drops_serve_endpoint(self) -> None:
        """If register_port_forward raises, the serve endpoint is dropped so the
        worker-internal address is never stored as a client-facing endpoint."""
        port_forward = MagicMock()
        port_forward.register_port_forward.side_effect = RuntimeError("port exhausted")

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert "serve" not in result

    def test_register_port_forward_failure_fails_task(self) -> None:
        """A forward-registration failure must fail the task (no fallback for
        serve) so the worker executor is torn down rather than left running."""
        port_forward = MagicMock()
        port_forward.register_port_forward.side_effect = RuntimeError("port exhausted")

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        monitor._dispatcher.fail_task.assert_called_once()  # type: ignore[attr-defined]
        call_args = monitor._dispatcher.fail_task.call_args  # type: ignore[attr-defined]
        assert call_args.args[0] == "tsk-abc"
        assert call_args.kwargs["worker_id"] == "wrk-1"

    def test_direct_mode_does_not_fail_task(self) -> None:
        """A healthy direct-mode update must never fail the task."""
        monitor = _make_monitor(port_forward=MagicMock())

        payload = {
            "serve": {
                "model": "m",
                "mode": "direct",
                "host": "127.0.0.1",
                "port": 8000,
            }
        }
        monitor._handle_serve_task_update("tsk-abc", "wrk-1", payload)

        monitor._dispatcher.fail_task.assert_not_called()  # type: ignore[attr-defined]

    def test_forward_mode_success_does_not_fail_task(self) -> None:
        """A successful forward registration must never fail the task."""
        port_forward = MagicMock()
        port_forward.register_port_forward.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        monitor._dispatcher.fail_task.assert_not_called()  # type: ignore[attr-defined]

    def test_ssh_forward_failure_does_not_fail_task(self) -> None:
        """SSH keeps its fall_back_to_direct behavior; forward-registration
        failure must never fail the SSH task."""
        port_forward = MagicMock()
        port_forward.register_port_forward.side_effect = RuntimeError("port exhausted")

        monitor = _make_monitor(port_forward=port_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        payload = {
            "ssh": {
                "mode": "forward",
                "host": "127.0.0.1",
                "port": 8000,
                "_relay_target": {"host": "127.0.0.1", "port": 8000},
            }
        }
        result = monitor._handle_ssh_task_update("tsk-abc", "wrk-1", payload)

        assert result["ssh"]["mode"] == "direct"
        monitor._dispatcher.fail_task.assert_not_called()  # type: ignore[attr-defined]
