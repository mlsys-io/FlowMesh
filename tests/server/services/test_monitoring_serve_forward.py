"""Serve TASK_UPDATE forward-registration tests for EventMonitor.

Regression coverage for the gap where a serve task with accessMode=forward
would never get a server-allocated client-facing port because
PortForwardService.register_port_forward requires a session_id that serve
payloads don't have.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from server.services.monitoring import EventMonitor


def _make_monitor(
    port_forward: MagicMock | None = None,
    ssh_proxy_enabled: bool = False,
    server_base_url: str = "http://server.example.com:8000",
) -> EventMonitor:
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test.monitoring.serve_forward"),
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
        ssh_proxy_enabled=ssh_proxy_enabled,
        port_forward=port_forward,
        server_base_url=server_base_url,
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

    def test_no_forward_service_also_stops_the_worker(self) -> None:
        """Marking the task failed only updates scheduling state; it does not
        stop the worker process. Without an explicit stop, the vLLM server
        the executor started keeps running (and holding its GPU) until the
        task's TTL expires, so a stop must be published too."""
        monitor = _make_monitor(port_forward=None)
        worker = SimpleNamespace(id="wrk-1")
        monitor._worker_registry.get_worker.return_value = worker  # type: ignore[attr-defined]

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        monitor._worker_registry.get_worker.assert_called_once_with(  # type: ignore[attr-defined]
            "wrk-1"
        )
        monitor._worker_registry.publish_stop.assert_called_once()  # type: ignore[attr-defined]
        call_args = monitor._worker_registry.publish_stop.call_args  # type: ignore[attr-defined]
        assert call_args.args[0] is worker
        stop_message = call_args.args[1]
        assert stop_message.task_id == "tsk-abc"
        assert stop_message.worker_id == "wrk-1"

    def test_fail_forward_task_without_worker_id_does_not_call_worker_registry(
        self,
    ) -> None:
        """No worker_id means there's nothing to stop; `get_worker`/
        `publish_stop` must not be invoked (e.g. with `None`)."""
        monitor = _make_monitor(port_forward=None)

        monitor._handle_serve_task_update("tsk-abc", None, _serve_forward_payload())

        monitor._worker_registry.get_worker.assert_not_called()  # type: ignore[attr-defined]
        monitor._worker_registry.publish_stop.assert_not_called()  # type: ignore[attr-defined]

    def test_fail_forward_task_worker_not_found_does_not_crash(self) -> None:
        """If the worker has already unregistered, there's nothing to stop;
        this must not raise."""
        monitor = _make_monitor(port_forward=None)
        monitor._worker_registry.get_worker.return_value = None  # type: ignore[attr-defined]

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        monitor._worker_registry.publish_stop.assert_not_called()  # type: ignore[attr-defined]

    def test_fail_forward_task_publish_stop_failure_does_not_crash(self) -> None:
        """A transient failure publishing the stop signal must be logged, not
        raised — the task is already marked failed regardless."""
        monitor = _make_monitor(port_forward=None)
        worker = SimpleNamespace(id="wrk-1")
        monitor._worker_registry.get_worker.return_value = worker  # type: ignore[attr-defined]
        monitor._worker_registry.publish_stop.side_effect = RuntimeError(  # type: ignore[attr-defined]
            "redis unavailable"
        )

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_forward_payload())

        monitor._dispatcher.fail_task.assert_called_once()  # type: ignore[attr-defined]

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


def _serve_proxy_payload(host: str = "127.0.0.1", port: int = 8000) -> dict:
    return {
        "serve": {
            "model": "Qwen/Qwen3-7B",
            "api_key": "key123",
            "mode": "proxy",
            "host": host,
            "port": port,
            "_relay_target": {"host": host, "port": port},
        }
    }


class TestServeProxyRegistration:
    def test_proxy_mode_does_not_call_register_port_forward(self) -> None:
        """Proxy mode never touches the port_forward relay-allocation path."""
        port_forward = MagicMock()
        monitor = _make_monitor(port_forward=port_forward, ssh_proxy_enabled=True)

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_proxy_payload())

        port_forward.register_port_forward.assert_not_called()

    def test_proxy_mode_advertises_server_public_proxy_url(self) -> None:
        """The client-visible endpoint is the server's own proxy route, not the
        worker-internal host/port."""
        monitor = _make_monitor(
            port_forward=None,
            ssh_proxy_enabled=True,
            server_base_url="http://server.example.com:8000",
        )

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_proxy_payload()
        )

        serve_info = result["serve"]
        assert serve_info["mode"] == "proxy"
        assert serve_info["host"] == "server.example.com"
        assert serve_info["port"] == 8000
        assert (
            serve_info["url"]
            == "http://server.example.com:8000/api/v1/serve/tasks/tsk-abc"
        )

    def test_proxy_mode_keeps_relay_target_for_server_side_uplink(self) -> None:
        """`_relay_target` must survive so the proxy router can start the
        uplink; it never reaches the client because tasks.py strips private
        (underscore-prefixed) fields before returning latest_update."""
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=True)

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_proxy_payload(host="127.0.0.1", port=9001)
        )

        assert result["serve"]["_relay_target"] == {"host": "127.0.0.1", "port": 9001}

    def test_proxy_mode_keeps_api_key_and_model(self) -> None:
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=True)

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_proxy_payload()
        )

        assert result["serve"]["api_key"] == "key123"
        assert result["serve"]["model"] == "Qwen/Qwen3-7B"

    def test_proxy_mode_disabled_drops_endpoint(self) -> None:
        """When the relay proxy is disabled, proxy mode is rejected like an
        unservable access mode (dropped, no fallback)."""
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=False)

        result = monitor._handle_serve_task_update(
            "tsk-abc", "wrk-1", _serve_proxy_payload()
        )

        assert "serve" not in result

    def test_proxy_mode_disabled_fails_task(self) -> None:
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=False)

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_proxy_payload())

        monitor._dispatcher.fail_task.assert_called_once()  # type: ignore[attr-defined]

    def test_proxy_mode_success_does_not_fail_task(self) -> None:
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=True)

        monitor._handle_serve_task_update("tsk-abc", "wrk-1", _serve_proxy_payload())

        monitor._dispatcher.fail_task.assert_not_called()  # type: ignore[attr-defined]

    def test_proxy_mode_no_worker_id_drops_endpoint(self) -> None:
        monitor = _make_monitor(port_forward=None, ssh_proxy_enabled=True)

        result = monitor._handle_serve_task_update(
            "tsk-abc", None, _serve_proxy_payload()
        )

        assert "serve" not in result


class TestServerBaseUrlValidation:
    """`server_base_url` is validated at construction so a malformed value
    can't silently produce a broken advertised proxy URL."""

    def test_valid_https_url_passes_through(self) -> None:
        monitor = _make_monitor(server_base_url="https://serve.example.com:9443")

        assert (
            monitor._server_base_url  # type: ignore[attr-defined]
            == "https://serve.example.com:9443"
        )

    def test_missing_scheme_falls_back_to_default(self) -> None:
        monitor = _make_monitor(server_base_url="serve.example.com:9443")

        assert monitor._server_base_url == "http://localhost:8000"  # type: ignore[attr-defined]

    def test_missing_host_falls_back_to_default(self) -> None:
        monitor = _make_monitor(server_base_url="http://")

        assert monitor._server_base_url == "http://localhost:8000"  # type: ignore[attr-defined]

    def test_empty_string_falls_back_to_default(self) -> None:
        monitor = _make_monitor(server_base_url="")

        assert monitor._server_base_url == "http://localhost:8000"  # type: ignore[attr-defined]

    def test_unsupported_scheme_falls_back_to_default(self) -> None:
        monitor = _make_monitor(server_base_url="ftp://serve.example.com")

        assert monitor._server_base_url == "http://localhost:8000"  # type: ignore[attr-defined]

    def test_unparsable_url_falls_back_instead_of_raising(self) -> None:
        """A malformed bracketed IPv6 host makes `urlparse` itself raise
        `ValueError`; construction must still fall back, not propagate."""
        monitor = _make_monitor(server_base_url="http://[::1")

        assert monitor._server_base_url == "http://localhost:8000"  # type: ignore[attr-defined]
