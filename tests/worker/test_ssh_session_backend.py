"""Tests for the SSH session backend seam and the process backend."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from shared.tasks.specs import SSHSpecStrict
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_session import (
    SessionRequest,
    SSHConfig,
    count_established_connections,
    select_backend_cls,
)
from worker.executors.ssh_session import session_identity as session_identity_module
from worker.executors.ssh_session.backends import docker as docker_backend_module
from worker.executors.ssh_session.backends import process as process_backend_module
from worker.executors.ssh_session.backends.docker import (
    DockerSession,
    DockerSessionBackend,
)
from worker.executors.ssh_session.backends.process import (
    ProcessSession,
    ProcessSessionBackend,
    _install_finish_helper,
    _render_authorized_keys,
    _render_sshd_config,
)
from worker.executors.ssh_session.session_identity import (
    ACCOUNT_NAME_RE,
    ACCOUNT_PREFIX,
    CurrentUser,
    DedicatedAccount,
    account_name_for,
    resolve_identity,
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
        owner="worker-1",
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

    def test_auto_falls_back_to_an_available_process_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whether that backend is safe to use is its own call to make."""
        monkeypatch.setattr(docker_backend_module, "docker_available", lambda: False)
        monkeypatch.setattr(
            ProcessSessionBackend, "is_available", classmethod(lambda cls, config: True)
        )
        assert (
            select_backend_cls(make_live_worker_config(tmp_path))
            is ProcessSessionBackend
        )

    def test_auto_yields_nothing_when_no_backend_is_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_backend_module, "docker_available", lambda: False)
        monkeypatch.setattr(
            ProcessSessionBackend,
            "is_available",
            classmethod(lambda cls, config: False),
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
            bind_host="127.0.0.1",
        )
        assert "Port 2222" in rendered
        assert "AllowUsers appuser" in rendered
        assert "PasswordAuthentication no" in rendered
        assert f"AuthorizedKeysFile {(tmp_path / 'authorized_keys').as_posix()}" in (
            rendered
        )

    def test_authorized_keys_carry_session_environment(self) -> None:
        rendered, exported = _render_authorized_keys(
            ["ssh-ed25519 AAAA... user@host"],
            {"CUDA_VISIBLE_DEVICES": "2,3", "FLOWMESH_FINISH_SENTINEL": "/x/finish"},
        )
        assert 'environment="CUDA_VISIBLE_DEVICES=2,3"' in rendered
        assert rendered.rstrip().endswith("ssh-ed25519 AAAA... user@host")
        assert "CUDA_VISIBLE_DEVICES" in exported

    def test_unsafe_environment_values_are_dropped(self) -> None:
        rendered, exported = _render_authorized_keys(
            ["ssh-ed25519 AAAA..."],
            {"OK": "fine", "BAD": 'has"quote', "also bad": "x"},
        )
        assert 'environment="OK=fine"' in rendered
        assert "BAD" not in rendered
        assert "also bad" not in rendered
        assert exported == ["OK"]

    def test_permit_user_environment_lists_only_exported_names(self) -> None:
        """A fixed pattern list would silently drop task-spec env vars."""
        _, exported = _render_authorized_keys(
            ["ssh-ed25519 AAAA..."], {"MY_TASK_VAR": "v", "CUDA_VISIBLE_DEVICES": "0"}
        )
        rendered = _render_sshd_config(
            port=2222,
            session_dir=Path("/tmp/s"),
            host_key=Path("/tmp/s/hk"),
            authorized_keys=Path("/tmp/s/ak"),
            login_user="fmssn1",
            bind_host="127.0.0.1",
            exported_env=exported,
        )
        assert "PermitUserEnvironment CUDA_VISIBLE_DEVICES,MY_TASK_VAR" in rendered

    def test_no_keys_renders_empty_file(self) -> None:
        assert _render_authorized_keys([], {"OK": "fine"}) == ("", [])


class TestConnectionCounting:
    def test_counts_only_established_connections_on_the_session_port(self) -> None:
        assert count_established_connections(_PROC_NET_TCP, 0x08AE) == 2

    def test_other_ports_are_ignored(self) -> None:
        assert count_established_connections(_PROC_NET_TCP, 0x1F90) == 1

    def test_unlistened_port_counts_zero(self) -> None:
        assert count_established_connections(_PROC_NET_TCP, 22) == 0

    def test_garbage_input_counts_zero(self) -> None:
        assert count_established_connections("not a table", 22) == 0


class TestSessionAccountNaming:
    def test_name_is_derived_from_the_session_id(self) -> None:
        name = account_name_for("ssn-a1b2c3d4e5")
        assert name.startswith(ACCOUNT_PREFIX)
        assert ACCOUNT_NAME_RE.match(name)

    def test_name_stays_within_the_linux_limit(self) -> None:
        name = account_name_for("ssn-" + "f" * 200)
        assert len(name) <= 31
        assert ACCOUNT_NAME_RE.match(name)

    def test_hostile_session_id_still_yields_a_safe_name(self) -> None:
        """Nothing but [a-z0-9-] may reach useradd, whatever the id contains."""
        name = account_name_for("ssn-;rm -rf /;$(id)")
        assert ACCOUNT_NAME_RE.match(name)
        assert not set(name) - set("abcdefghijklmnopqrstuvwxyz0123456789-")


class TestReportedLoginUser:
    """The reported username must be the one sshd will actually accept."""

    def test_process_session_reports_its_own_account(self, tmp_path: Path) -> None:
        identity = CurrentUser()
        session = ProcessSession.__new__(ProcessSession)
        session.identity = identity
        assert session.login_user() == identity.name

    def test_reported_user_matches_allowusers(self, tmp_path: Path) -> None:
        identity = CurrentUser()
        session = ProcessSession.__new__(ProcessSession)
        session.identity = identity
        rendered = _render_sshd_config(
            port=2222,
            session_dir=tmp_path,
            host_key=tmp_path / "hk",
            authorized_keys=tmp_path / "ak",
            login_user=identity.name,
            bind_host="127.0.0.1",
        )
        assert f"AllowUsers {session.login_user()}" in rendered

    def test_docker_session_reports_the_spec_user(self) -> None:
        session = DockerSession.__new__(DockerSession)
        session._cfg = _interactive_cfg()
        assert session.login_user() == session._cfg.user


class TestIdentitySelection:
    def test_non_root_worker_still_gets_a_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Process mode must never refuse just because the worker is not root."""
        monkeypatch.setattr(session_identity_module.os, "getuid", lambda: 10001)
        identity = resolve_identity("ssn-abcd1234", tmp_path)
        assert isinstance(identity, CurrentUser)
        assert identity.isolates_from_worker is False

    def test_current_user_identity_reports_no_isolation(self) -> None:
        assert CurrentUser().isolates_from_worker is False


class TestFinishSentinelPlacement:
    """The sentinel must never outlive its session, or the next one ends at once."""

    def test_sentinel_lives_under_the_session_dir_without_isolation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(session_identity_module.os, "getuid", lambda: 10001)
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        identity = resolve_identity("ssn-abcd1234", session_dir)
        plan = backend._build_paths(
            _request(_interactive_cfg(), tmp_path), session_dir, identity
        )
        assert plan.finish_sentinel.is_relative_to(session_dir)
        assert not plan.finish_sentinel.is_relative_to(Path.home())


class TestAccountRelease:
    def test_process_scan_uses_a_valid_psutil_attr(self) -> None:
        """A bad attr name only raises when release() actually runs."""
        account = DedicatedAccount(
            "fmssn-none", uid=4294967, gid=4294967, home=Path("/nonexistent")
        )
        account._kill_processes()

    def test_current_user_release_never_deletes_the_worker_account(self) -> None:
        before = CurrentUser().name
        CurrentUser().release()
        assert CurrentUser().name == before


class TestFinishHelperParity:
    """A process-mode session gets the same flowmesh-finish command Docker ships."""

    def test_helper_is_installed_and_executable(self, tmp_path: Path) -> None:
        sentinel = tmp_path / "home" / ".flowmesh_finish"
        bin_dir = _install_finish_helper(tmp_path, sentinel)
        helper = bin_dir / "flowmesh-finish"
        assert helper.exists()
        assert helper.stat().st_mode & 0o111
        assert sentinel.as_posix() in helper.read_text()

    def test_bin_dir_is_traversable_for_path_lookup(self, tmp_path: Path) -> None:
        """0711 is enough: a PATH search stats candidates, it does not list."""
        bin_dir = _install_finish_helper(tmp_path, tmp_path / "finish")
        assert bin_dir.stat().st_mode & 0o111 == 0o111

    def test_helper_creates_the_sentinel_the_session_loop_polls(
        self, tmp_path: Path
    ) -> None:
        sentinel = tmp_path / "finish"
        bin_dir = _install_finish_helper(tmp_path, sentinel)
        subprocess.run(
            [str(bin_dir / "flowmesh-finish")], check=True, capture_output=True
        )
        assert sentinel.exists()


class TestUnusablePasswordHash:
    """A fresh account is locked until it carries a real hash, so this runs on
    every process-mode session and must not depend on the value it generates."""

    def test_the_generated_secret_is_never_read_as_an_option(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """openssl parses a leading "-" as a flag, and token_urlsafe emits them."""
        seen: list[list[str]] = []

        def _capture(argv: list[str], _what: str) -> Any:
            seen.append(argv)
            return SimpleNamespace(stdout=b"$6$abc$def\n")

        monkeypatch.setattr(session_identity_module, "_run", _capture)
        monkeypatch.setattr(
            session_identity_module.shutil, "which", lambda _name: "/usr/bin/openssl"
        )

        for _ in range(200):
            session_identity_module._unusable_password_hash()

        assert seen, "the hash was never generated"
        assert all(not argv[-1].startswith("-") for argv in seen)
