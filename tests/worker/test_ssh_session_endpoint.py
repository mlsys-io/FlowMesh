"""What an SSH session publishes: its endpoint, and where it says it lives.

The supervisor no longer dials the session, so nothing here needs to be
routable from it. ``session_host`` still does, because a ``direct`` session is
dialled by the client.
"""

import socket
from pathlib import Path
from typing import Any, cast

import pytest

from shared.tasks.specs import SSHSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.config import WorkerConfig
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
                "apiVersion": "flowmesh/v1",
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


def _emit_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_backend: bool = False,
) -> dict[str, Any]:
    """Run a session to ready and return the update it published."""
    executor = SSHExecutor(make_live_worker_config(tmp_path))
    if process_backend:
        executor._backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
    emitted: dict[str, Any] = {}
    monkeypatch.setattr(
        executor, "emit_update", lambda task_id, payload: emitted.update(payload)
    )
    task = _task_message()
    cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
    executor._wait_session_ready(cast(Any, _ReadySession()), "ssn-1234", task, cfg)
    return emitted


class TestSessionHost:
    def test_docker_backend_reports_its_fqdn(self, tmp_path: Path) -> None:
        backend = DockerSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_host() == socket.getfqdn()

    def test_docker_backend_honours_explicit_override(self, tmp_path: Path) -> None:
        backend = DockerSessionBackend(
            make_live_worker_config(tmp_path, ssh_direct_host="10.0.0.9")
        )
        assert backend.session_host() == "10.0.0.9"

    def test_process_backend_prefers_its_tailnet_address(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            process_backend_module, "resolve_tailnet_address", lambda: _TAILNET_ADDRESS
        )
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_host() == _TAILNET_ADDRESS

    def test_process_backend_override_wins_over_tailnet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            process_backend_module, "resolve_tailnet_address", lambda: _TAILNET_ADDRESS
        )
        backend = ProcessSessionBackend(
            make_live_worker_config(tmp_path, ssh_direct_host="10.0.0.9")
        )
        assert backend.session_host() == "10.0.0.9"

    def test_process_backend_falls_back_to_fqdn_without_a_tailnet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No tailnet is no longer fatal: only a direct session needs this."""
        monkeypatch.setattr(
            process_backend_module, "resolve_tailnet_address", lambda: None
        )
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_host() == socket.getfqdn()


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

    def test_a_proxy_session_publishes_its_port_and_no_relay_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing dials the worker now, so no address is published for it."""
        ssh_info = self._emit(tmp_path, monkeypatch)
        assert ssh_info["mode"] == "proxy"
        assert ssh_info["port"] == 2222
        assert "_relay_target" not in ssh_info

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
        assert emitted["ssh"]["port"] == 2222


class TestEndpointPublication:
    def test_a_ready_session_is_published_for_relaying(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The registry entry is what a relay request resolves against."""
        executor = SSHExecutor(make_live_worker_config(tmp_path))
        published: dict[str, int] = {}
        monkeypatch.setattr(executor, "emit_update", lambda *_: None)
        monkeypatch.setattr(
            executor,
            "publish_endpoint",
            lambda eid, port: published.update({eid: port}),
        )
        task = _task_message()
        cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
        executor._wait_session_ready(cast(Any, _ReadySession()), "ssn-1234", task, cfg)

        assert published == {"ssn-1234": 2222}


class TestSessionBindHost:
    def test_a_relayed_process_session_binds_loopback(self, tmp_path: Path) -> None:
        """Nothing outside the worker dials it, so nothing outside should reach it."""
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_bind_host("proxy") == "127.0.0.1"
        assert backend.session_bind_host("forward") == "127.0.0.1"

    def test_a_direct_process_session_binds_every_interface(
        self, tmp_path: Path
    ) -> None:
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_bind_host("direct") == "0.0.0.0"

    def test_a_docker_session_is_reachable_in_every_mode(self, tmp_path: Path) -> None:
        """Docker publishes the container's port on the host either way."""
        backend = DockerSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_bind_host("proxy") == "0.0.0.0"
        assert backend.session_bind_host("direct") == "0.0.0.0"


class TestSessionAddress:
    """What the worker tells the server the session can be reached at."""

    def test_a_relayed_process_session_reports_loopback(self, tmp_path: Path) -> None:
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_address("proxy") == "127.0.0.1"
        assert backend.session_address("forward") == "127.0.0.1"

    def test_a_direct_process_session_reports_its_host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            process_backend_module, "resolve_tailnet_address", lambda: _TAILNET_ADDRESS
        )
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_address("direct") == _TAILNET_ADDRESS

    def test_a_docker_session_reports_its_host_in_every_mode(
        self, tmp_path: Path
    ) -> None:
        """Docker publishes the port on the host, so it is never loopback-only."""
        backend = DockerSessionBackend(make_live_worker_config(tmp_path))
        assert backend.session_address("proxy") == socket.getfqdn()
        assert backend.session_address("direct") == socket.getfqdn()

    def test_the_advertised_address_is_what_the_session_is_published_at(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        emitted = _emit_ready(tmp_path, monkeypatch)

        assert emitted["ssh"]["host"] == socket.getfqdn()
        assert emitted["ssh"]["port"] == 2222
        assert "_bind_host" not in emitted["ssh"]

    def test_a_relayed_session_reports_its_own_route_and_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`host` may be rewritten to the server's route; this one is not."""
        emitted = _emit_ready(tmp_path, monkeypatch)["ssh"]

        assert emitted["directHost"] == socket.getfqdn()
        assert emitted["directPort"] == 2222
        assert emitted["directScope"] == "network"
        assert emitted["workerId"] == "worker-1"

    def test_a_loopback_bound_session_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        emitted = _emit_ready(tmp_path, monkeypatch, process_backend=True)["ssh"]

        assert emitted["directHost"] == "127.0.0.1"
        assert emitted["directScope"] == "loopback"


class TestDirectHostFromEnv:
    @pytest.fixture(autouse=True)
    def _required_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("WORKER_TOKEN", "tok")
        monkeypatch.setenv("SUPERVISOR_GRPC_TARGET", "supervisor:50051")
        monkeypatch.setenv("WORKER_HB_FILE", (tmp_path / "worker.hb").as_posix())
        monkeypatch.delenv("SSH_DIRECT_HOST", raising=False)
        monkeypatch.delenv("SSH_RELAY_HOST", raising=False)

    def test_unset_leaves_the_worker_to_discover_its_address(self) -> None:
        assert WorkerConfig.from_env().ssh_direct_host is None

    def test_reads_the_current_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SSH_DIRECT_HOST", "10.0.0.9")
        assert WorkerConfig.from_env().ssh_direct_host == "10.0.0.9"

    def test_falls_back_to_the_former_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SSH_RELAY_HOST", "10.0.0.8")
        assert WorkerConfig.from_env().ssh_direct_host == "10.0.0.8"

    def test_the_current_name_wins_over_the_former(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SSH_DIRECT_HOST", "10.0.0.9")
        monkeypatch.setenv("SSH_RELAY_HOST", "10.0.0.8")
        assert WorkerConfig.from_env().ssh_direct_host == "10.0.0.9"
