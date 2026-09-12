"""Tests for SSH session TTL and idle reaping."""

from pathlib import Path
from typing import cast

import pytest

from shared.tasks.specs import SSHSpecStrict
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.executors.base_executor import ExecutionError, TaskCancelledError
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.ssh_session import SSHConfig, SSHSession
from worker.executors.ssh_session import docker_backend as docker_backend_module


@pytest.fixture(autouse=True)
def _docker_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_backend_module, "docker_available", lambda: True)


class _FakeSession(SSHSession):
    """Session whose observable state the test drives directly."""

    def __init__(
        self,
        connections: int | None = 0,
        exit_code: int | None = None,
        output_size: int | None = None,
    ) -> None:
        self.connections = connections
        self.exit_code = exit_code
        self.output_size = output_size
        self.stopped_with: float | None = None
        self.cleaned = False

    def login_user(self) -> str:
        return "flowmesh"

    def wait_ready(self, timeout_sec: float) -> int:
        return 2222

    def poll(self) -> int | None:
        return self.exit_code

    def finish_requested(self) -> bool:
        return False

    def established_connections(self) -> int | None:
        return self.connections

    def output_size_bytes(self) -> int | None:
        return self.output_size

    def collect_output(self, destination: Path) -> None:
        return None

    def stop(self, timeout_sec: float) -> None:
        self.stopped_with = timeout_sec

    def cleanup(self) -> None:
        self.cleaned = True


def _cfg(**spec_updates: object) -> SSHConfig:
    payload: dict[str, object] = {
        "taskType": "ssh",
        "authorizedKeys": ["ssh-ed25519 AAAA..."],
        **spec_updates,
    }
    return SSHConfig.from_spec(
        cast(SSHSpecStrict, SSHSpecStrict.model_validate(payload)),
        DEFAULT_WORKER_CONFIG,
    )


def _executor(tmp_path: Path) -> SSHExecutor:
    return SSHExecutor(make_live_worker_config(tmp_path))


def _fast_poll(cfg: SSHConfig, idle_sec: float) -> SSHConfig:
    cfg.poll_interval_sec = 0.01
    cfg.idle_sec = idle_sec
    return cfg


class TestIdleReaping:
    def test_idle_session_is_reaped_before_ttl(self, tmp_path: Path) -> None:
        executor = _executor(tmp_path)
        cfg = _fast_poll(_cfg(ttlSeconds=600), idle_sec=0.05)
        session = _FakeSession(connections=0)

        assert executor._wait_for_session(session, cfg) == 0  # noqa: SLF001
        assert session.stopped_with == 1

    def test_connected_session_is_not_reaped(self, tmp_path: Path) -> None:
        executor = _executor(tmp_path)
        cfg = _fast_poll(_cfg(ttlSeconds=600), idle_sec=0.05)
        cfg.ttl_sec = 0.3
        session = _FakeSession(connections=1)

        assert executor._wait_for_session(session, cfg) == 0  # noqa: SLF001
        assert session.stopped_with is None

    def test_unobservable_connections_never_reap(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``None`` means no evidence, which must not be read as idle."""
        executor = _executor(tmp_path)
        cfg = _fast_poll(_cfg(ttlSeconds=600), idle_sec=0.05)
        cfg.ttl_sec = 0.3
        session = _FakeSession(connections=None)

        with caplog.at_level("WARNING"):
            assert executor._wait_for_session(session, cfg) == 0  # noqa: SLF001

        assert session.stopped_with is None
        assert any(
            "idle timeout cannot be enforced" in record.message
            for record in caplog.records
        )

    def test_idle_reaping_is_off_for_noninteractive_sessions(
        self, tmp_path: Path
    ) -> None:
        executor = _executor(tmp_path)
        cfg = _fast_poll(
            _cfg(interactive=False, image="x", command=["true"], ttlSeconds=600),
            idle_sec=0.05,
        )
        cfg.ttl_sec = 0.3
        session = _FakeSession(connections=0)

        assert executor._wait_for_session(session, cfg) == 0  # noqa: SLF001
        assert session.stopped_with is None

    def test_zero_idle_timeout_disables_reaping(self, tmp_path: Path) -> None:
        executor = _executor(tmp_path)
        cfg = _fast_poll(_cfg(ttlSeconds=600), idle_sec=0)
        cfg.ttl_sec = 0.3
        session = _FakeSession(connections=0)

        assert executor._wait_for_session(session, cfg) == 0  # noqa: SLF001
        assert session.stopped_with is None


class TestSessionLoop:
    def test_exit_code_is_returned(self, tmp_path: Path) -> None:
        executor = _executor(tmp_path)
        cfg = _fast_poll(_cfg(ttlSeconds=600), idle_sec=600)
        session = _FakeSession(connections=1, exit_code=3)

        assert executor._wait_for_session(session, cfg) == 3  # noqa: SLF001

    def test_cancellation_propagates(self, tmp_path: Path) -> None:
        executor = _executor(tmp_path)
        cfg = _fast_poll(_cfg(ttlSeconds=600), idle_sec=600)
        executor._cancel_event.set()  # noqa: SLF001

        with pytest.raises(TaskCancelledError):
            executor._wait_for_session(_FakeSession(), cfg)  # noqa: SLF001

    def test_output_limit_breach_fails_the_task(self, tmp_path: Path) -> None:
        """A maxBytes breach must surface, not be swallowed into a clean exit."""
        executor = _executor(tmp_path)
        cfg = _fast_poll(_cfg(ttlSeconds=600, sshOutput={"maxBytes": 10}), idle_sec=600)
        session = _FakeSession(connections=1, output_size=11)

        with pytest.raises(ExecutionError, match="exceeded maxBytes"):
            executor._wait_for_session(session, cfg)  # noqa: SLF001
        assert session.stopped_with == 1


class TestIdleClamping:
    def test_idle_timeout_never_exceeds_ttl(self) -> None:
        cfg = _cfg(ttlSeconds=60, idleTimeoutSeconds=3600)
        assert cfg.idle_sec == 60

    def test_idle_timeout_default_is_kept_below_ttl(self) -> None:
        cfg = _cfg(ttlSeconds=120)
        assert cfg.idle_sec == 120

    def test_spec_idle_timeout_is_honoured(self) -> None:
        cfg = _cfg(ttlSeconds=3600, idleTimeoutSeconds=120)
        assert cfg.idle_sec == 120
