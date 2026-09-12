"""Tests for the SSH session backend seam and the process backend."""

from pathlib import Path
from typing import cast

import pytest

from shared.tasks.specs import SSHSpecStrict
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_session import (
    SessionRequest,
    SSHConfig,
    count_established_connections,
)
from worker.executors.ssh_session import docker_backend as docker_backend_module
from worker.executors.ssh_session import process_backend as process_backend_module
from worker.executors.ssh_session import (
    select_backend_cls,
)
from worker.executors.ssh_session.docker_backend import DockerSessionBackend
from worker.executors.ssh_session.process_backend import (
    ProcessSessionBackend,
    _render_authorized_keys,
    _render_sshd_config,
)

_PROC_NET_TCP = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid
   0: 00000000:08AE 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000
   1: 0100007F:08AE 0100007F:C1B6 01 00000000:00000000 00:00000000 00000000  1000
   2: 0100007F:08AE 0100007F:C1B7 01 00000000:00000000 00:00000000 00000000  1000
   3: 0100007F:1F90 0100007F:C1B8 01 00000000:00000000 00:00000000 00000000  1000
   4: 0100007F:08AE 0100007F:C1B9 06 00000000:00000000 00:00000000 00000000  1000
"""


def _interactive_cfg(**spec_updates: object) -> SSHConfig:
    payload: dict[str, object] = {
        "taskType": "ssh",
        "authorizedKeys": ["ssh-ed25519 AAAA... user@host"],
        **spec_updates,
    }
    return SSHConfig.from_spec(
        cast(SSHSpecStrict, SSHSpecStrict.model_validate(payload)),
        DEFAULT_WORKER_CONFIG,
    )


def _request(cfg: SSHConfig, tmp_path: Path) -> SessionRequest:
    return SessionRequest(
        task_id="task-ssh",
        session_id="ssn-1234",
        worker_name="worker-1",
        cfg=cfg,
        out_dir=tmp_path / "out",
        resolved_inputs=[],
    )


class TestBackendSelection:
    def test_auto_resolves_to_docker_when_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_backend_module, "docker_available", lambda: True)
        assert (
            select_backend_cls(make_live_worker_config(tmp_path))
            is DockerSessionBackend
        )

    def test_auto_never_falls_back_to_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_backend_module, "docker_available", lambda: False)
        monkeypatch.setattr(
            ProcessSessionBackend, "is_available", classmethod(lambda cls, config: True)
        )
        assert select_backend_cls(make_live_worker_config(tmp_path)) is None

    def test_explicit_process_selects_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_backend_module, "docker_available", lambda: False)
        monkeypatch.setattr(
            ProcessSessionBackend, "is_available", classmethod(lambda cls, config: True)
        )
        cfg = make_live_worker_config(tmp_path, ssh_session_backend="process")
        assert select_backend_cls(cfg) is ProcessSessionBackend

    def test_explicit_process_unavailable_without_sshd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(process_backend_module, "find_sshd", lambda: None)
        cfg = make_live_worker_config(tmp_path, ssh_session_backend="process")
        assert select_backend_cls(cfg) is None

    def test_unknown_backend_name_selects_nothing(self, tmp_path: Path) -> None:
        cfg = make_live_worker_config(tmp_path, ssh_session_backend="containerd")
        assert select_backend_cls(cfg) is None


class TestProcessBackendGuards:
    def test_refuses_noninteractive_tasks(self, tmp_path: Path) -> None:
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        cfg = SSHConfig.from_spec(
            cast(
                SSHSpecStrict,
                SSHSpecStrict.model_validate(
                    {
                        "taskType": "ssh",
                        "interactive": False,
                        "image": "python:3.12-slim",
                        "command": ["true"],
                    }
                ),
            ),
            DEFAULT_WORKER_CONFIG,
        )
        with pytest.raises(ExecutionError, match="container runtime"):
            backend.start_session(_request(cfg, tmp_path))

    def test_refuses_a_second_concurrent_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a container there is no boundary between two sessions."""
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        monkeypatch.setattr(
            backend, "_create_session", lambda request: cast(object, "session")
        )
        backend.start_session(_request(_interactive_cfg(), tmp_path))
        with pytest.raises(ExecutionError, match="only one session"):
            backend.start_session(_request(_interactive_cfg(), tmp_path))

    def test_backend_does_not_support_noninteractive(self) -> None:
        assert ProcessSessionBackend.supports_noninteractive is False
        assert DockerSessionBackend.supports_noninteractive is True


class TestSshdConfigRendering:
    def test_config_pins_key_only_login_for_the_worker_user(
        self, tmp_path: Path
    ) -> None:
        rendered = _render_sshd_config(
            port=2222,
            session_dir=tmp_path,
            host_key=tmp_path / "hostkey",
            authorized_keys=tmp_path / "authorized_keys",
            login_user="appuser",
        )
        assert "Port 2222" in rendered
        assert "AllowUsers appuser" in rendered
        assert "PasswordAuthentication no" in rendered
        assert f"AuthorizedKeysFile {(tmp_path / 'authorized_keys').as_posix()}" in (
            rendered
        )

    def test_authorized_keys_carry_session_environment(self) -> None:
        rendered = _render_authorized_keys(
            ["ssh-ed25519 AAAA... user@host"],
            {"CUDA_VISIBLE_DEVICES": "2,3", "FLOWMESH_FINISH_SENTINEL": "/x/finish"},
        )
        assert 'environment="CUDA_VISIBLE_DEVICES=2,3"' in rendered
        assert rendered.rstrip().endswith("ssh-ed25519 AAAA... user@host")

    def test_unsafe_environment_values_are_dropped(self) -> None:
        rendered = _render_authorized_keys(
            ["ssh-ed25519 AAAA..."],
            {"OK": "fine", "BAD": 'has"quote', "also bad": "x"},
        )
        assert 'environment="OK=fine"' in rendered
        assert "BAD" not in rendered
        assert "also bad" not in rendered

    def test_no_keys_renders_empty_file(self) -> None:
        assert _render_authorized_keys([], {"OK": "fine"}) == ""


class TestConnectionCounting:
    def test_counts_only_established_connections_on_the_session_port(self) -> None:
        assert count_established_connections(_PROC_NET_TCP, 0x08AE) == 2

    def test_other_ports_are_ignored(self) -> None:
        assert count_established_connections(_PROC_NET_TCP, 0x1F90) == 1

    def test_unlistened_port_counts_zero(self) -> None:
        assert count_established_connections(_PROC_NET_TCP, 22) == 0

    def test_garbage_input_counts_zero(self) -> None:
        assert count_established_connections("not a table", 22) == 0
