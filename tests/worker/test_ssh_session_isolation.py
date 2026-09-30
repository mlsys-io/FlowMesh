"""Tests for keeping process-mode SSH sessions away from worker state.

Everything here runs unprivileged: accounts are stood in for by the test's own
uid, and ACL and account tooling is stubbed where a real call would need root.
"""

import dataclasses
import os
import stat
import tempfile
import types
import typing
from pathlib import Path
from typing import Any, cast

import pytest

from shared.tasks.specs import SSHSpecStrict
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker import config as worker_config_module
from worker.config import (
    SESSION_ALLOWED_PATH_FIELDS,
    SESSION_DENIED_PATH_FIELDS,
    WorkerConfig,
)
from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_session import acl
from worker.executors.ssh_session import session_identity as identity_module
from worker.executors.ssh_session.backends import process as process_module
from worker.executors.ssh_session.backends.process import (
    ProcessSession,
    ProcessSessionBackend,
    ProcessSessionPaths,
    _claim_root,
    _link_mount_path,
    _required_path_under,
    _reset_mount_root,
)
from worker.executors.ssh_session.base import iter_tree, path_size_bytes
from worker.executors.ssh_session.config import SSHConfig, normalize_mount_path
from worker.executors.ssh_session.session_identity import (
    CurrentUser,
    DedicatedAccount,
    purge_uid_files,
)


def _own_account(home: Path, uid: int | None = None) -> DedicatedAccount:
    return DedicatedAccount(
        "fmssn-test",
        uid=os.getuid() if uid is None else uid,
        gid=os.getgid(),
        home=home,
    )


def _cfg(**spec: object) -> SSHConfig:
    payload: dict[str, object] = {
        "taskType": "ssh",
        "authorizedKeys": ["ssh-ed25519 AAAA... user@host"],
        **spec,
    }
    return SSHConfig.from_spec(
        cast(SSHSpecStrict, SSHSpecStrict.model_validate(payload)),
        DEFAULT_WORKER_CONFIG,
    )


def _path_typed(annotation: Any) -> bool:
    if annotation is Path:
        return True
    return any(_path_typed(arg) for arg in typing.get_args(annotation))


class TestMountPathNormalization:
    @pytest.mark.parametrize(
        "raw",
        [
            "/mnt/flowmesh/../../etc",
            "/mnt/flowmesh/a/../../../root",
            "/mnt/flowmesh/..",
            "/mnt/flowmesh/a/..",
        ],
    )
    def test_parent_components_are_refused(self, raw: str) -> None:
        with pytest.raises(ExecutionError, match=r"\.\."):
            normalize_mount_path(raw, field_name="sshOutput.mountPath")

    def test_dot_and_repeated_separators_are_normalized(self) -> None:
        assert (
            normalize_mount_path("/mnt//flowmesh/./out//data/", field_name="f")
            == "/mnt/flowmesh/out/data"
        )

    @pytest.mark.parametrize("raw", ["mnt/flowmesh/out", "/mnt/other", "/", "/mnt"])
    def test_paths_outside_the_mount_root_are_refused(self, raw: str) -> None:
        with pytest.raises(ExecutionError):
            normalize_mount_path(raw, field_name="f")


class TestDeniedStateRoots:
    def test_every_path_field_is_classified_exactly_once(self) -> None:
        hints = typing.get_type_hints(WorkerConfig)
        path_fields = {
            field.name
            for field in dataclasses.fields(WorkerConfig)
            if _path_typed(hints[field.name])
        }
        denied = set(SESSION_DENIED_PATH_FIELDS)
        allowed = set(SESSION_ALLOWED_PATH_FIELDS)
        assert not denied & allowed
        unclassified = path_fields - denied - allowed
        assert not unclassified, (
            f"Classify {sorted(unclassified)} in SESSION_DENIED_PATH_FIELDS or "
            "SESSION_ALLOWED_PATH_FIELDS"
        )
        assert denied | allowed <= path_fields

    def test_denied_paths_cover_results_heartbeat_and_state(
        self, tmp_path: Path
    ) -> None:
        cache = tmp_path / "hf"
        cfg = make_live_worker_config(tmp_path, session_state_dirs=(cache,))
        denied = cfg.session_denied_paths()
        assert cfg.results_dir in denied
        assert cfg.hb_file in denied
        assert cache in denied

    def test_state_dirs_come_from_home_caches_and_temp_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HOME", (tmp_path / "hf").as_posix())
        monkeypatch.delenv("TORCH_HOME", raising=False)
        dirs = worker_config_module._session_state_dirs_from_env()
        assert Path.home() in dirs
        assert tmp_path / "hf" in dirs
        assert Path(tempfile.gettempdir()) / "utu" in dirs

    def test_a_root_above_what_sessions_need_is_detected(self) -> None:
        assert _required_path_under(Path("/")) is not None
        assert _required_path_under(Path("/mnt")) is not None
        assert _required_path_under(Path("/app/worker/results")) is None


class TestNoFollowWalks:
    def test_size_ignores_links(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.write_bytes(b"x" * 1000)
        tree = tmp_path / "tree"
        (tree / "sub").mkdir(parents=True)
        (tree / "sub" / "a").write_bytes(b"12345")
        (tree / "link").symlink_to(outside)
        (tree / "dirlink").symlink_to(tmp_path)
        assert path_size_bytes(tree) == 5

    def test_a_linked_root_is_not_walked(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        (real / "f").write_text("data")
        (tmp_path / "link").symlink_to(real)
        assert list(iter_tree(tmp_path / "link")) == []

    def test_a_tree_too_deep_to_walk_fails_sizing(self, tmp_path: Path) -> None:
        deepest = tmp_path.joinpath(*(["d"] * 70))
        deepest.mkdir(parents=True)
        (deepest / "f").write_bytes(b"x")
        with pytest.raises(ExecutionError, match="deeper"):
            path_size_bytes(tmp_path)


class TestHandingPathsToTheSession:
    def test_a_path_the_session_did_not_create_is_refused(self, tmp_path: Path) -> None:
        existing = tmp_path / "existing"
        existing.mkdir()
        with pytest.raises(ExecutionError, match="did not create"):
            _own_account(tmp_path).own(existing)

    def test_an_existing_path_cannot_be_created(self, tmp_path: Path) -> None:
        (tmp_path / "taken").mkdir()
        with pytest.raises(ExecutionError):
            _own_account(tmp_path).make_dir(tmp_path / "taken", 0o700)

    def test_a_link_at_the_created_path_is_refused(self, tmp_path: Path) -> None:
        account = _own_account(tmp_path)
        created = tmp_path / "out"
        account.make_dir(created, 0o700)
        created.rmdir()
        created.symlink_to(tmp_path)
        with pytest.raises(ExecutionError, match="link"):
            account.own(created)

    def test_recursive_handover_does_not_follow_links(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.write_text("secret")
        outside.chmod(0o644)
        account = _own_account(tmp_path)
        staged = tmp_path / "staged"
        account.make_dir(staged, 0o700)
        (staged / "shared").mkdir(mode=0o777)
        (staged / "shared" / "file").write_text("input")
        (staged / "shared" / "file").chmod(0o666)
        (staged / "link").symlink_to(outside)
        account.own(staged, recursive=True)
        assert stat.S_IMODE(outside.stat().st_mode) == 0o644
        assert stat.S_IMODE((staged / "shared").stat().st_mode) == 0o700
        assert stat.S_IMODE((staged / "shared" / "file").stat().st_mode) == 0o600


class TestAccountDenies:
    def test_a_partial_deny_is_rolled_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        applied: list[Path] = []
        revoked: list[Path] = []

        def fake_deny(uid: int, path: Path) -> None:
            if path.name == "second":
                raise ExecutionError("no ACL support")
            applied.append(path)

        monkeypatch.setattr(acl, "record", lambda uid, path: None)
        monkeypatch.setattr(acl, "forget", lambda uid, path: None)
        monkeypatch.setattr(acl, "deny", fake_deny)
        monkeypatch.setattr(acl, "revoke", lambda uid, path: revoked.append(path))
        with pytest.raises(ExecutionError, match="Could not isolate"):
            _own_account(tmp_path).deny([tmp_path / "first", tmp_path / "second"])
        assert applied == [tmp_path / "first"]
        assert set(revoked) == {tmp_path / "first", tmp_path / "second"}

    def test_denies_outlive_an_account_that_cannot_be_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        revoked: list[Path] = []
        monkeypatch.setattr(acl, "revoke", lambda uid, path: revoked.append(path))
        monkeypatch.setattr(acl, "forget", lambda uid, path: None)
        monkeypatch.setattr(identity_module, "_delete_account", lambda name: False)
        account = _own_account(tmp_path, uid=4294967)
        account._denied = [tmp_path / "results"]
        account.release()
        assert revoked == []

    def test_denies_outlive_processes_that_survive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deleted: list[str] = []

        def delete_account(name: str) -> bool:
            deleted.append(name)
            return True

        monkeypatch.setattr(identity_module, "_terminate_uid", lambda uid: False)
        monkeypatch.setattr(identity_module, "_delete_account", delete_account)
        account = _own_account(tmp_path, uid=4294967)
        account._denied = [tmp_path / "results"]
        account.release()
        assert deleted == []
        assert account._denied == [tmp_path / "results"]

    def test_orphaned_records_are_revoked_and_live_ones_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        revoked: list[tuple[int, str]] = []
        forgotten: list[tuple[int, str]] = []
        monkeypatch.setattr(
            acl, "recorded", lambda: {(61001, "/nonexistent/a"), (61002, "/b")}
        )
        monkeypatch.setattr(identity_module, "_uid_exists", lambda uid: uid == 61002)
        monkeypatch.setattr(
            acl, "revoke", lambda uid, path: revoked.append((uid, path.as_posix()))
        )
        monkeypatch.setattr(
            acl, "forget", lambda uid, path: forgotten.append((uid, path.as_posix()))
        )
        identity_module._revoke_orphaned_denies()
        assert revoked == []
        assert forgotten == [(61001, "/nonexistent/a")]

    def test_getfacl_output_parses_to_denied_uids(self) -> None:
        output = (
            "user::rwx\nuser:61001:---\nuser:1000:r-x\ngroup::r-x\nmask::r-x\n"
            "other::r-x\ndefault:user:61002:---\n"
        )
        assert acl.parse_denied_uids(output) == {61001}


class TestUidFilePurge:
    def test_files_the_uid_left_are_removed_without_following_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scratch = tmp_path / "scratch"
        (scratch / "nested").mkdir(parents=True)
        (scratch / "nested" / "left").write_text("old tenant")
        keep = tmp_path / "keep"
        keep.mkdir()
        (keep / "file").write_text("not in scratch")
        (scratch / "link").symlink_to(keep)
        monkeypatch.setattr(tempfile, "tempdir", scratch.as_posix())
        monkeypatch.setattr(identity_module, "_WORLD_WRITABLE_DIRS", ())
        purge_uid_files(os.getuid())
        assert list(scratch.iterdir()) == []
        assert (keep / "file").read_text() == "not in scratch"


class TestMountRoot:
    def test_reset_removes_links_without_following_them(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "precious").write_text("keep")
        root = tmp_path / "mnt"
        (root / "old" / "deep").mkdir(parents=True)
        (root / "link").symlink_to(outside)
        (root / "old" / "link").symlink_to(outside)
        _reset_mount_root(root, create=True)
        assert list(root.iterdir()) == []
        assert (outside / "precious").read_text() == "keep"
        assert stat.S_IMODE(root.stat().st_mode) == 0o755

    def test_reset_replaces_a_linked_root(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "precious").write_text("keep")
        root = tmp_path / "mnt"
        root.symlink_to(outside)
        _reset_mount_root(root, create=True)
        assert root.is_dir() and not root.is_symlink()
        assert (outside / "precious").read_text() == "keep"

    def test_nested_mount_paths_are_linked(self, tmp_path: Path) -> None:
        root = tmp_path / "mnt"
        root.mkdir()
        target = tmp_path / "session" / "output"
        target.mkdir(parents=True)
        _link_mount_path(root, (root / "a" / "b" / "out").as_posix(), target)
        link = root / "a" / "b" / "out"
        assert link.is_symlink()
        assert link.readlink() == target

    def test_a_linked_component_is_never_followed(self, tmp_path: Path) -> None:
        root = tmp_path / "mnt"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "a").symlink_to(outside)
        with pytest.raises(ExecutionError, match="conflicts"):
            _link_mount_path(root, (root / "a" / "out").as_posix(), tmp_path)
        assert list(outside.iterdir()) == []

    def test_an_existing_mount_path_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "mnt"
        (root / "out").mkdir(parents=True)
        with pytest.raises(ExecutionError, match="conflicts"):
            _link_mount_path(root, (root / "out").as_posix(), tmp_path)

    def test_the_mount_root_itself_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ExecutionError, match="below"):
            _link_mount_path(tmp_path, tmp_path.as_posix(), tmp_path)


class TestClaimingStateRoots:
    def test_a_missing_directory_root_is_created_private(self, tmp_path: Path) -> None:
        root = tmp_path / "cache" / "hf"
        assert _claim_root(root, is_dir=True) is True
        assert stat.S_IMODE(root.stat().st_mode) == 0o700

    def test_a_missing_file_root_needs_no_deny(self, tmp_path: Path) -> None:
        assert _claim_root(tmp_path / "worker.hb", is_dir=False) is False

    def test_a_linked_root_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "real").mkdir()
        (tmp_path / "linked").symlink_to(tmp_path / "real")
        with pytest.raises(ExecutionError, match="link"):
            _claim_root(tmp_path / "linked", is_dir=True)

    def test_a_planted_root_in_a_shared_dir_is_recreated(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o1777)
        victim = tmp_path / "victim"
        victim.mkdir()
        (shared / "utu").symlink_to(victim)
        assert _claim_root(shared / "utu", is_dir=True) is True
        assert (shared / "utu").is_dir() and not (shared / "utu").is_symlink()
        assert victim.is_dir()


class TestProcessBackendIsolation:
    def test_root_worker_without_acl_support_offers_no_process_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "getuid", lambda: 0)
        monkeypatch.setattr(process_module, "_acquire_host_lock", lambda: True)
        monkeypatch.setattr(acl, "tools_available", lambda: True)

        def unsupported(directory: Path) -> None:
            raise ExecutionError("ACL entries do not persist")

        monkeypatch.setattr(acl, "probe", unsupported)
        cfg = make_live_worker_config(tmp_path)
        assert ProcessSessionBackend._isolation_ready(cfg) is False

    def test_a_root_covering_the_mount_root_offers_no_process_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "getuid", lambda: 0)
        monkeypatch.setattr(process_module, "_acquire_host_lock", lambda: True)
        monkeypatch.setattr(acl, "tools_available", lambda: True)
        cfg = make_live_worker_config(tmp_path, session_state_dirs=(Path("/"),))
        assert ProcessSessionBackend._isolation_ready(cfg) is False

    def test_another_worker_holding_the_host_lock_offers_no_process_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(process_module, "_acquire_host_lock", lambda: False)
        cfg = make_live_worker_config(tmp_path)
        assert ProcessSessionBackend._isolation_ready(cfg) is False

    def test_lingering_session_processes_refuse_a_new_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            process_module, "live_session_accounts", lambda: ["fmssnold"]
        )
        backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
        request = types.SimpleNamespace(cfg=_cfg())
        with pytest.raises(ExecutionError, match="earlier SSH session"):
            backend.start_session(cast(Any, request))


class TestOutputCollection:
    def _session(
        self, tmp_path: Path, output: Path, identity: Any, max_bytes: int | None = None
    ) -> ProcessSession:
        spec: dict[str, object] = {"sshOutput": {"maxBytes": max_bytes}}
        session = ProcessSession.__new__(ProcessSession)
        session._plan = ProcessSessionPaths(
            finish_sentinel=tmp_path / ".finish", output_path=output, mount_root=None
        )
        session._cfg = _cfg(**spec)
        session.identity = identity
        session.stop = lambda timeout_sec: None  # type: ignore[method-assign]
        return session

    def test_only_regular_files_are_collected(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret"
        secret.write_text("worker credentials")
        output = tmp_path / "output"
        (output / "sub").mkdir(parents=True)
        (output / "result.txt").write_text("result")
        (output / "sub" / "nested.txt").write_text("nested")
        (output / "leak").symlink_to(secret)
        (output / "dirleak").symlink_to(tmp_path)
        os.mkfifo(output / "fifo")
        destination = tmp_path / "artifacts"
        self._session(tmp_path, output, CurrentUser()).collect_output(destination)
        assert (destination / "result.txt").read_text() == "result"
        assert (destination / "sub" / "nested.txt").read_text() == "nested"
        assert not (destination / "leak").exists()
        assert not (destination / "leak").is_symlink()
        assert not (destination / "dirleak").exists()
        assert not (destination / "fifo").exists()

    def test_files_the_session_does_not_own_are_not_collected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "output"
        output.mkdir()
        (output / "hardlinked").write_text("someone else's file")
        identity = _own_account(tmp_path, uid=os.getuid() + 1)
        monkeypatch.setattr(identity, "terminate_processes", lambda: True)
        destination = tmp_path / "artifacts"
        self._session(tmp_path, output, identity).collect_output(destination)
        assert not (destination / "hardlinked").exists()

    def test_output_over_max_bytes_is_refused(self, tmp_path: Path) -> None:
        output = tmp_path / "output"
        output.mkdir()
        (output / "big").write_bytes(b"x" * 100)
        with pytest.raises(ExecutionError, match="maxBytes"):
            self._session(tmp_path, output, CurrentUser(), max_bytes=10).collect_output(
                tmp_path / "artifacts"
            )
        assert not (tmp_path / "artifacts" / "big").exists()
