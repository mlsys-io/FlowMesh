"""Tests for the relay target an SSH session publishes.

``_relay_target`` is dialled by the supervisor that owns the worker's node, so
the address has to be routable *from the supervisor*. Loopback is right only
when the two share a host.
"""

from pathlib import Path
from typing import Any, cast

import pytest

from shared.tasks.specs import SSHSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.ssh_session import SSHConfig
from worker.executors.ssh_session.backends import docker as docker_backend_module
from worker.executors.ssh_session.backends import process as process_backend_module
from worker.executors.ssh_session.backends.docker import DockerSessionBackend
from worker.executors.ssh_session.backends.process import ProcessSessionBackend

_TAILNET_ADDRESS = "100.89.73.50"


@pytest.fixture(autouse=True)
def _docker_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_backend_module, "docker_available", lambda: True)


def _task_message() -> WorkerTaskMessage:
    return WorkerTaskMessage.model_validate(
        {
            "task_id": "task-ssh",
            "workflow_id": "wf-1",
            "owner_id": "owner",
            "assigned_worker": "worker-1",
            "dispatched_at": "2026-03-22T00:00:00Z",
            "task": {
                "apiVersion": "mloc/v1",
                "kind": "Task",
                "metadata": {"name": "wf:shell"},
                "spec": {
                    "taskType": "ssh",
                    "accessMode": "proxy",
                    "authorizedKeys": ["ssh-ed25519 AAAA..."],
                },
            },
        }
    )


class _ReadySession:
    """Minimal session stand-in: ready on a fixed port, nothing else."""

    def __init__(self, port: int = 2222, user: str = "flowmesh") -> None:
        self.port = port
        self.user = user

    def wait_ready(self, timeout_sec: float) -> int:
        return self.port

    def login_user(self) -> str:
        return self.user


class TestBackendRelayHost:
    def test_docker_backend_publishes_loopback(self, tmp_path: Path) -> None:
        backend = DockerSessionBackend(make_live_worker_config(tmp_path))
        assert backend.relay_host() == "127.0.0.1"

    def test_docker_backend_honours_explicit_override(self, tmp_path: Path) -> None:
        backend = DockerSessionBackend(
            make_live_worker_config(tmp_path, ssh_relay_host="10.0.0.9")
        )
        assert backend.relay_host() == "10.0.0.9"

    def test_process_backend_publishes_tailnet_address(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            process_backend_module, "resolve_tailnet_address", lambda: _TAILNET_ADDRESS
        )
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        assert backend.relay_host() == _TAILNET_ADDRESS

    def test_process_backend_override_wins_over_tailnet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            process_backend_module, "resolve_tailnet_address", lambda: _TAILNET_ADDRESS
        )
        backend = ProcessSessionBackend(
            make_live_worker_config(tmp_path, ssh_relay_host="10.0.0.9")
        )
        assert backend.relay_host() == "10.0.0.9"

    def test_process_backend_refuses_unroutable_relay_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failing loudly beats publishing a loopback the supervisor can't dial."""
        monkeypatch.setattr(
            process_backend_module, "resolve_tailnet_address", lambda: None
        )
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        with pytest.raises(ExecutionError, match="no tailnet address"):
            backend.relay_host()


class TestPublishedSessionInfo:
    def _emit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **worker_overrides: Any
    ) -> dict[str, Any]:
        executor = SSHExecutor(make_live_worker_config(tmp_path, **worker_overrides))
        emitted: dict[str, Any] = {}
        monkeypatch.setattr(
            executor,
            "emit_update",
            lambda task_id, payload: emitted.update(payload),
        )
        task = _task_message()
        cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
        executor._wait_session_ready(cast(Any, _ReadySession()), "ssn-1234", task, cfg)
        return cast(dict[str, Any], emitted["ssh"])

    def test_docker_session_keeps_loopback_relay_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ssh_info = self._emit(tmp_path, monkeypatch)
        assert ssh_info["_relay_target"] == {"host": "127.0.0.1", "port": 2222}
        assert ssh_info["mode"] == "proxy"

    def test_relay_target_uses_configured_host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ssh_info = self._emit(tmp_path, monkeypatch, ssh_relay_host=_TAILNET_ADDRESS)
        assert ssh_info["_relay_target"] == {
            "host": _TAILNET_ADDRESS,
            "port": 2222,
        }

    def test_direct_mode_publishes_no_relay_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = SSHExecutor(make_live_worker_config(tmp_path))
        emitted: dict[str, Any] = {}
        monkeypatch.setattr(
            executor,
            "emit_update",
            lambda task_id, payload: emitted.update(payload),
        )
        task = _task_message()
        cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
        cfg.access_mode = "direct"
        executor._wait_session_ready(cast(Any, _ReadySession()), "ssn-1234", task, cfg)
        assert "_relay_target" not in emitted["ssh"]
