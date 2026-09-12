"""Regression tests for executor bootstrap in ``worker.main.initialize_executors``.

Pins the contract that ``initialize_executors`` constructs every non-MP executor
with ``cls(config, hardware, lifecycle)``. Subclasses are expected to accept this
via ``(*args, **kwargs)`` passthrough so future ``Executor.__init__`` extensions
don't break the chain.
"""

import logging
import os
from pathlib import Path
from typing import Any

import pytest

from shared.schemas.result import BaseExecutorResult
from tests.worker.factories import make_live_worker_config, make_worker_hardware
from worker.executors.base_executor import Executor, ExecutorTask
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.ssh_session.backends import docker as docker_backend_mod
from worker.executors.ssh_session.backends import process as process_backend_mod
from worker.main import initialize_executors


class _PassthroughExecutor(Executor):
    """Executor that forwards constructor args via the recommended pattern."""

    name = "passthrough"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def run(self, task: ExecutorTask, out_dir: Path) -> BaseExecutorResult:
        return BaseExecutorResult.model_validate({"ok": True})


class _UnavailableExecutor(_PassthroughExecutor):
    """Executor that declares itself unavailable on this worker."""

    name = "unavailable"

    @classmethod
    def is_available(cls, config: Any) -> bool:
        return False


class TestInitializeExecutorsHardware:
    def test_executor_receives_hardware_via_passthrough(self, tmp_path: Path) -> None:
        cfg = make_live_worker_config(tmp_path)
        hw = make_worker_hardware()
        executors, default = initialize_executors(
            config=cfg,
            hardware=hw,
            logger=logging.getLogger("test"),
            lifecycle=None,  # type: ignore[arg-type]
            registry={"echo": _PassthroughExecutor, "default": _PassthroughExecutor},
            import_errors={},
            cuda_available=False,
            enable_mp_executors=False,
        )
        # Pre-fix this would silently drop the executor because the subclass
        # constructor didn't accept the new positional ``hardware`` arg.
        assert isinstance(executors["echo"], _PassthroughExecutor)
        assert isinstance(default, _PassthroughExecutor)
        assert executors["echo"]._hardware is hw
        assert default._hardware is hw


class TestInitializeExecutorsAvailability:
    def test_unavailable_executor_is_skipped(self, tmp_path: Path) -> None:
        cfg = make_live_worker_config(tmp_path)
        executors, _ = initialize_executors(
            config=cfg,
            hardware=make_worker_hardware(),
            logger=logging.getLogger("test"),
            lifecycle=None,  # type: ignore[arg-type]
            registry={
                "default": _PassthroughExecutor,
                "echo": _UnavailableExecutor,
            },
            import_errors={},
            cuda_available=False,
            enable_mp_executors=False,
        )
        assert "echo" not in executors
        assert "default" in executors

    def test_ssh_availability_tracks_docker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = make_live_worker_config(tmp_path)
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(docker_backend_mod, "docker_available", lambda: False)
        assert SSHExecutor.is_available(cfg) is False
        monkeypatch.setattr(docker_backend_mod, "docker_available", lambda: True)
        assert SSHExecutor.is_available(cfg) is True

    def test_auto_refuses_process_backend_on_a_non_root_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker image that ships sshd must not silently downgrade.

        Without root there is no second identity to give the session, so
        ``auto`` declines rather than running it as the worker.
        """
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(docker_backend_mod, "docker_available", lambda: False)
        monkeypatch.setattr(process_backend_mod, "find_sshd", lambda: "/usr/sbin/sshd")
        monkeypatch.setattr(
            process_backend_mod, "find_ssh_keygen", lambda: "/usr/bin/ssh-keygen"
        )
        assert SSHExecutor.is_available(make_live_worker_config(tmp_path)) is False

    def test_auto_falls_back_to_process_on_a_root_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Root can mint a per-session account, so the fallback is isolated."""
        monkeypatch.setattr(os, "getuid", lambda: 0)
        monkeypatch.setattr(docker_backend_mod, "docker_available", lambda: False)
        monkeypatch.setattr(process_backend_mod, "find_sshd", lambda: "/usr/sbin/sshd")
        monkeypatch.setattr(
            process_backend_mod, "find_ssh_keygen", lambda: "/usr/bin/ssh-keygen"
        )
        assert SSHExecutor.is_available(make_live_worker_config(tmp_path)) is True

    def test_auto_needs_sshd_even_as_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "getuid", lambda: 0)
        monkeypatch.setattr(docker_backend_mod, "docker_available", lambda: False)
        monkeypatch.setattr(process_backend_mod, "find_sshd", lambda: None)
        assert SSHExecutor.is_available(make_live_worker_config(tmp_path)) is False

    def test_explicit_process_backend_still_needs_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming the backend is not enough to waive the isolation it needs."""
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(docker_backend_mod, "docker_available", lambda: False)
        monkeypatch.setattr(process_backend_mod, "find_sshd", lambda: "/usr/sbin/sshd")
        monkeypatch.setattr(
            process_backend_mod, "find_ssh_keygen", lambda: "/usr/bin/ssh-keygen"
        )
        cfg = make_live_worker_config(tmp_path, ssh_session_backend="process")
        assert SSHExecutor.is_available(cfg) is False

    def test_unisolated_flag_serves_sessions_without_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The operator accepting the trade is what unlocks it."""
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(docker_backend_mod, "docker_available", lambda: False)
        monkeypatch.setattr(process_backend_mod, "find_sshd", lambda: "/usr/sbin/sshd")
        monkeypatch.setattr(
            process_backend_mod, "find_ssh_keygen", lambda: "/usr/bin/ssh-keygen"
        )
        cfg = make_live_worker_config(
            tmp_path,
            ssh_session_backend="process",
            enable_unisolated_ssh_session=True,
        )
        assert SSHExecutor.is_available(cfg) is True

    def test_unisolated_flag_lets_auto_reach_the_process_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(docker_backend_mod, "docker_available", lambda: False)
        monkeypatch.setattr(process_backend_mod, "find_sshd", lambda: "/usr/sbin/sshd")
        monkeypatch.setattr(
            process_backend_mod, "find_ssh_keygen", lambda: "/usr/bin/ssh-keygen"
        )
        cfg = make_live_worker_config(tmp_path, enable_unisolated_ssh_session=True)
        assert SSHExecutor.is_available(cfg) is True

    def test_unknown_ssh_backend_is_unavailable(self, tmp_path: Path) -> None:
        cfg = make_live_worker_config(tmp_path, ssh_session_backend="nonsense")
        assert SSHExecutor.is_available(cfg) is False
