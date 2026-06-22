"""Serve TASK_UPDATE forward-registration tests for EventMonitor.

Regression coverage for the gap where a serve task with accessMode=forward
would never get a server-allocated client-facing port because
ForwardService.register_forward_task requires a session_id that serve
payloads don't have.
"""

import logging
from unittest.mock import MagicMock

from server.services.monitoring import EventMonitor


def _make_monitor(forward: MagicMock | None = None) -> EventMonitor:
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
        forward=forward,
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
    def test_forward_mode_calls_register_forward_task(self) -> None:
        """A serve TASK_UPDATE with mode=forward triggers register_forward_task."""
        ssh_forward = MagicMock()
        ssh_forward.register_forward_task.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(forward=ssh_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        ssh_forward.register_forward_task.assert_called_once()

    def test_forward_mode_injects_task_id_as_session_id(self) -> None:
        """task_id is passed as session_id so the relay service's logging works."""
        ssh_forward = MagicMock()
        ssh_forward.register_forward_task.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(forward=ssh_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        call_args = ssh_forward.register_forward_task.call_args
        ssh_info_arg = call_args.args[3]
        assert ssh_info_arg.get("session_id") == "tsk-abc"

    def test_forward_mode_surfaces_server_allocated_port(self) -> None:
        """The returned payload carries the server's public host/port, not the
        worker-internal endpoint."""
        ssh_forward = MagicMock()
        ssh_forward.register_forward_task.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(forward=ssh_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert result["serve"]["host"] == "server.example.com"
        assert result["serve"]["port"] == 32001

    def test_forward_mode_strips_injected_session_id_from_result(self) -> None:
        """The synthetic session_id must not appear in latest_update.serve."""
        ssh_forward = MagicMock()
        # Simulate register_forward_task echoing back the injected session_id
        ssh_forward.register_forward_task.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "session_id": "tsk-abc",
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(forward=ssh_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert "session_id" not in result["serve"]

    def test_forward_mode_strips_relay_target_from_result(self) -> None:
        """_relay_target must not appear in latest_update.serve after registration."""
        ssh_forward = MagicMock()
        ssh_forward.register_forward_task.return_value = {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "forward",
            "host": "server.example.com",
            "port": 32001,
            "_relay_target": {"host": "127.0.0.1", "port": 8000},
        }

        monitor = _make_monitor(forward=ssh_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert "_relay_target" not in result["serve"]

    def test_direct_mode_does_not_call_register(self) -> None:
        """A serve update with mode=direct leaves the payload unchanged."""
        ssh_forward = MagicMock()
        monitor = _make_monitor(forward=ssh_forward)

        payload = {
            "serve": {
                "model": "m",
                "mode": "direct",
                "host": "127.0.0.1",
                "port": 8000,
            }
        }
        result = monitor._handle_serve_task_update("tsk-abc", "wrk-1", payload)

        ssh_forward.register_forward_task.assert_not_called()
        assert result["serve"]["host"] == "127.0.0.1"
        assert result["serve"]["port"] == 8000

    def test_no_forward_service_drops_serve_endpoint(self) -> None:
        """When forward is None, forward mode is rejected and the serve endpoint
        is dropped rather than degraded to direct (serve has no proxy fallback)."""
        monitor = _make_monitor(forward=None)

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert "serve" not in result

    def test_register_forward_task_failure_drops_serve_endpoint(self) -> None:
        """If register_forward_task raises, the serve endpoint is dropped so the
        worker-internal address is never stored as a client-facing endpoint."""
        ssh_forward = MagicMock()
        ssh_forward.register_forward_task.side_effect = RuntimeError("port exhausted")

        monitor = _make_monitor(forward=ssh_forward)
        monitor._runtime.get_record.return_value = None  # type: ignore[attr-defined]

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_forward_payload()
        )

        assert "serve" not in result
