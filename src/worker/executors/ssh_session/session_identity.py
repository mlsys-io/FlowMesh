"""The OS identity a process-mode SSH session logs in as.

Which identity a session gets decides what it can reach. A worker running as
root mints a throwaway account per session, so the session is a different uid
from the worker and cannot read the worker's environment — where the task
token and every third-party API key live. A worker that is not root can only
authenticate the account it already runs as, so the session shares the
worker's identity; that path stays available but isolates nothing.
"""

import logging
import os
import pwd
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path

import psutil

from ..base_executor import ExecutionError
from . import acl

logger = logging.getLogger(__name__)

ACCOUNT_PREFIX = "fmssn"
ACCOUNT_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
PRIVSEP_DIR = Path("/run/sshd")
SESSION_DIR_PREFIX = "flowmesh-ssh-"
# Debian's on-demand global range: above the uids distributions hand out to
# accounts, so a session never shares a uid with a principal of a shared volume.
SESSION_UID_MIN = 61000
SESSION_UID_MAX = 64999
_UID_ATTEMPTS = 16
_KILL_GRACE_SEC = 5.0
_KILL_ROUNDS = 10
_KILL_ROUND_SEC = 0.5
# Prepended to every helper script. It imports what the scripts use before
# switching to the uid and gid in argv[1:3], since that uid may be unable to
# read the interpreter's standard library.
_AS_UID_PREAMBLE = (
    "import ctypes, os, signal, sys\n"
    "uid = int(sys.argv[1])\n"
    "if os.getuid() != uid:\n"
    "    os.setgroups([])\n"
    "    os.setgid(int(sys.argv[2]))\n"
    "    os.setuid(uid)\n"
)
_KILL_ALL_SCRIPT = (
    "try:\n"
    "    os.kill(-1, signal.SIGKILL)\n"
    "except ProcessLookupError:\n"
    "    pass\n"
)
# argv[3:] is "<kind>:<id>" per object; 0 is IPC_RMID.
_REMOVE_IPC_SCRIPT = (
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "failed = 0\n"
    "for spec in sys.argv[3:]:\n"
    "    kind, ipc_id = spec.split(':')\n"
    "    if kind == 'shm':\n"
    "        rc = libc.shmctl(int(ipc_id), 0, None)\n"
    "    elif kind == 'msg':\n"
    "        rc = libc.msgctl(int(ipc_id), 0, None)\n"
    "    else:\n"
    "        rc = libc.semctl(int(ipc_id), 0, 0)\n"
    "    if rc != 0:\n"
    "        print(spec, os.strerror(ctypes.get_errno()), file=sys.stderr)\n"
    "        failed = 1\n"
    "sys.exit(failed)\n"
)
_AS_UID_TIMEOUT_SEC = 10.0
_NOGROUP_GID = 65534
_USERADD_TIMEOUT_SEC = 30.0
_WORLD_WRITABLE_DIRS = (
    Path("/", "var", "tmp"),
    Path("/", "dev", "shm"),
    Path("/", "dev", "mqueue"),
)
_SYSV_IPC_DIR = Path("/", "proc", "sysvipc")
# Each /proc/sysvipc table and the column holding its object ids.
_SYSV_IPC_ID_COLUMNS = {"shm": "shmid", "msg": "msqid", "sem": "semid"}


def account_name_for(session_id: str) -> str:
    """Derive a valid Linux account name from a session id."""
    tail = re.sub(r"[^a-z0-9]", "", session_id.lower())[-16:]
    name = f"{ACCOUNT_PREFIX}{tail or secrets.token_hex(4)}"[:31]
    if not ACCOUNT_NAME_RE.match(name):
        raise ExecutionError(f"Cannot derive a valid account name from {session_id!r}")
    return name


class SessionIdentity(ABC):
    """Who the session runs as, and what that costs in isolation."""

    name: str
    uid: int
    gid: int
    home: Path

    def __init__(self) -> None:
        self._created: set[Path] = set()

    @property
    @abstractmethod
    def isolates_from_worker(self) -> bool:
        """Whether the session is a different principal from the worker."""

    def make_dir(self, path: Path, mode: int) -> None:
        """Create a new directory that :meth:`own` may later hand to the session.

        Fails if anything, a link included, already exists at ``path``.
        """
        try:
            os.mkdir(path, mode)
            os.chmod(path, mode)
        except OSError as exc:
            raise ExecutionError(
                f"Cannot create SSH session directory {path.as_posix()}: {exc}"
            ) from exc
        self._created.add(path)

    def own(self, path: Path, mode: int | None = None, recursive: bool = False) -> None:
        """Hand ``path``, made by :meth:`make_dir`, to the session."""
        return None

    def deny(self, paths: Iterable[Path]) -> None:
        """Deny the session any access to ``paths``."""
        return None

    def terminate_processes(self) -> bool:
        """End the session's processes; whether none remain."""
        return True

    def release(self) -> None:
        return None


class CurrentUser(SessionIdentity):
    """The worker's own account: no separation, used when the worker is not root."""

    def __init__(self) -> None:
        super().__init__()
        self.uid = os.getuid()
        self.gid = os.getgid()
        try:
            self.name = pwd.getpwuid(self.uid).pw_name
        except KeyError:
            self.name = os.getenv("USER", "root")
        self.home = Path.home()

    @property
    def isolates_from_worker(self) -> bool:
        return False


class DedicatedAccount(SessionIdentity):
    """A throwaway account created for one session, deleted with it."""

    def __init__(self, name: str, uid: int, gid: int, home: Path) -> None:
        super().__init__()
        self.name = name
        self.uid = uid
        self.gid = gid
        self.home = home
        self._denied: list[Path] = []

    @property
    def isolates_from_worker(self) -> bool:
        return True

    @classmethod
    def create(
        cls, name: str, home: Path, avoid_uids: Iterable[int] = ()
    ) -> "DedicatedAccount":
        """Create the account under a fresh uid outside ``avoid_uids``."""
        useradd = _require_binary("useradd")
        _add_account(useradd, name, home, frozenset(avoid_uids))
        try:
            entry = pwd.getpwnam(name)
        except KeyError as exc:
            raise ExecutionError(f"SSH session account {name} was not created") from exc
        account = cls(name, entry.pw_uid, entry.pw_gid, home)
        try:
            # A fresh account's shadow entry is "!", which sshd reads as locked
            # and refuses even for public-key auth once UsePAM is off. An
            # unguessable hash leaves it unlocked without granting a password.
            _unlock(name)
            account.make_dir(home, 0o700)
            account.own(home)
        except Exception:
            account.release()
            raise
        return account

    def own(self, path: Path, mode: int | None = None, recursive: bool = False) -> None:
        """Hand ``path`` to the session without following a link anywhere.

        Only a directory this identity created is accepted, so an author-chosen
        path can never make the session the owner of something that was
        already there. ``recursive`` also makes everything below owner-only,
        since staged content keeps the shared modes it was copied with.
        """
        if path not in self._created:
            raise ExecutionError(
                f"Refusing to hand {path.as_posix()} to {self.name}: this session "
                "did not create it"
            )
        try:
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise ExecutionError(
                    f"Refusing to hand {path.as_posix()} to {self.name}: it is a link"
                )
            os.chown(path, self.uid, self.gid, follow_symlinks=False)
            if mode is not None:
                os.chmod(path, mode)
            if recursive:
                for parent, dirs, files in os.walk(path, followlinks=False):
                    for name in (*dirs, *files):
                        self._own_entry(os.path.join(parent, name))
        except OSError as exc:
            raise ExecutionError(
                f"Failed to hand {path.as_posix()} to {self.name}: {exc}"
            ) from exc

    def _own_entry(self, target: str) -> None:
        os.chown(target, self.uid, self.gid, follow_symlinks=False)
        mode = os.lstat(target).st_mode
        if stat.S_ISDIR(mode):
            os.chmod(target, 0o700)
        elif stat.S_ISREG(mode):
            os.chmod(target, (mode & 0o700) | 0o600)

    def deny(self, paths: Iterable[Path]) -> None:
        """Deny the account ``paths`` through ACLs, all or nothing."""
        try:
            for path in paths:
                acl.record(self.uid, path)
                self._denied.append(path)
                acl.deny(self.uid, path)
        except ExecutionError as exc:
            self._revoke_denies()
            raise ExecutionError(
                f"Could not isolate SSH session account {self.name} from this "
                f"worker's state: {exc}"
            ) from exc

    def terminate_processes(self) -> bool:
        return _terminate_uid(self.uid)

    def release(self) -> None:
        """Delete the account, then lift what was applied on its behalf.

        The denies stay until the account is gone: lifting them while one of
        its processes lives would hand that process the worker's state. An
        account that cannot be deleted keeps them and is left to the sweep.
        """
        if not self.terminate_processes():
            logger.warning(
                "Processes of SSH session account %s survived SIGKILL; leaving the "
                "account to the stale-account sweep",
                self.name,
            )
            return
        if not _delete_account(self.name):
            return
        self._revoke_denies()
        purge_uid(self.uid)

    def _revoke_denies(self) -> None:
        remaining: list[Path] = []
        for path in self._denied:
            try:
                acl.revoke(self.uid, path)
                acl.forget(self.uid, path)
            except ExecutionError:
                logger.warning(
                    "Failed to lift the ACL entry of %s on %s; the next sweep retries",
                    self.name,
                    path,
                )
                remaining.append(path)
        self._denied = remaining


def resolve_identity(
    session_id: str, session_dir: Path, denied_roots: Sequence[Path] = ()
) -> SessionIdentity:
    """Pick the strongest identity this worker can give a session.

    A dedicated account is denied ``denied_roots`` before it is returned, and
    gets a uid no entry on those roots names, so lifting its entries later can
    never lift another worker's.
    """
    if os.getuid() != 0:
        logger.warning(
            "This worker is not root, so the SSH session runs as %s — the worker's "
            "own account. The session can read the worker's environment and files, "
            "including its task token and any API keys. Run the worker as root to "
            "give each session its own account.",
            CurrentUser().name,
        )
        return CurrentUser()
    _ensure_privsep_dir()
    taken: set[int] = set()
    for root in denied_roots:
        taken |= acl.named_uids(root)
    account = DedicatedAccount.create(
        account_name_for(session_id), session_dir / "home", avoid_uids=taken
    )
    try:
        account.deny(denied_roots)
    except Exception:
        account.release()
        raise
    return account


def live_session_accounts() -> list[str]:
    """Session accounts that still have processes running."""
    if os.getuid() != 0:
        return []
    running = {
        uid
        for proc in psutil.process_iter(["uids", "status"])
        if (uid := _live_uid(proc)) is not None
    }
    return [
        entry.pw_name
        for entry in pwd.getpwall()
        if entry.pw_name.startswith(ACCOUNT_PREFIX) and entry.pw_uid in running
    ]


def reap_stale_accounts(keep: str | None = None) -> None:
    """Remove what sessions left behind on an unclean worker exit.

    Kills their processes, then deletes their accounts, their files and session
    directories, and the ACL entries recorded for them. An account whose
    processes survive is left for the next sweep, as is every entry recorded
    for an account that still exists.
    """
    if os.getuid() != 0:
        return
    for entry in pwd.getpwall():
        name = entry.pw_name
        if not name.startswith(ACCOUNT_PREFIX) or name == keep:
            continue
        if not _terminate_uid(entry.pw_uid):
            logger.warning(
                "Processes of stale SSH session account %s survived SIGKILL; the "
                "next sweep retries",
                name,
            )
            continue
        if not _delete_account(name):
            continue
        logger.info("Reaped stale SSH session account %s", name)
        purge_uid(entry.pw_uid)
        _remove_session_dir(Path(entry.pw_dir))
    _revoke_orphaned_denies()


def purge_uid(uid: int) -> None:
    """Delete what ``uid`` left behind that outlives its processes.

    Session uids are drawn at random and may come round again, so a later
    session must not inherit what an earlier one owned.
    """
    purge_uid_files(uid)
    purge_uid_ipc(uid)


def purge_uid_files(uid: int) -> None:
    """Delete what ``uid`` left in the shared scratch directories."""
    for base in (Path(tempfile.gettempdir()), *_WORLD_WRITABLE_DIRS):
        for parent, dirs, files in os.walk(base, followlinks=False):
            for name in (*dirs, *files):
                target = os.path.join(parent, name)
                try:
                    info = os.lstat(target)
                except OSError:
                    continue
                if info.st_uid != uid:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    _unlink_quietly(target)
            dirs[:] = [
                name for name in dirs if os.path.lexists(os.path.join(parent, name))
            ]


def purge_uid_ipc(uid: int) -> None:
    """Remove the System V IPC objects ``uid`` owns or created.

    Sessions share the worker's IPC namespace, and these objects persist after
    their creator exits. Only an owner, a creator or a holder of
    ``CAP_SYS_ADMIN`` may remove one, and a container's root lacks that
    capability, so the removal runs as ``uid``.
    """
    if uid == 0:
        return
    specs = [
        f"{kind}:{ipc_id}"
        for kind, id_column in _SYSV_IPC_ID_COLUMNS.items()
        for ipc_id in owned_ipc_ids(_read_ipc_table(kind), id_column, uid)
    ]
    if not specs:
        return
    result = _run_as(uid, _REMOVE_IPC_SCRIPT, specs)
    if result is None or result.returncode != 0:
        detail = "" if result is None else _stderr_of(result)
        logger.warning(
            "Failed to remove System V IPC objects of uid %d: %s", uid, detail
        )


def owned_ipc_ids(table: str, id_column: str, uid: int) -> list[int]:
    """Ids in a ``/proc/sysvipc`` table whose owner or creator is ``uid``.

    The creator is matched too because the owner can hand an object to any uid.
    """
    lines = table.splitlines()
    if not lines:
        return []
    header = lines[0].split()
    try:
        id_at = header.index(id_column)
        uid_at = header.index("uid")
        cuid_at = header.index("cuid")
    except ValueError:
        return []
    ids: list[int] = []
    for line in lines[1:]:
        fields = line.split()
        try:
            if uid in (int(fields[uid_at]), int(fields[cuid_at])):
                ids.append(int(fields[id_at]))
        except (IndexError, ValueError):
            continue
    return ids


def _read_ipc_table(kind: str) -> str:
    try:
        return (_SYSV_IPC_DIR / kind).read_text(encoding="utf-8")
    except OSError:
        return ""


def _revoke_orphaned_denies() -> None:
    try:
        records = acl.recorded()
    except ExecutionError:
        logger.warning("Cannot read recorded SSH session ACL entries", exc_info=True)
        return
    for uid, raw_path in records:
        if _uid_exists(uid):
            continue
        path = Path(raw_path)
        try:
            if path.exists():
                acl.revoke(uid, path)
            acl.forget(uid, path)
        except ExecutionError:
            logger.warning("Failed to lift a stale ACL entry on %s", path)


def _remove_session_dir(home: Path) -> None:
    session_dir = home.parent
    if (
        session_dir.name.startswith(SESSION_DIR_PREFIX)
        and session_dir.parent == Path(tempfile.gettempdir())
        and not session_dir.is_symlink()
        and session_dir.is_dir()
    ):
        shutil.rmtree(session_dir, ignore_errors=True)


def _add_account(
    useradd: str, name: str, home: Path, avoid_uids: frozenset[int]
) -> None:
    """Create ``name`` under a uid drawn at random from the session range."""
    detail = ""
    for _ in range(_UID_ATTEMPTS):
        uid = SESSION_UID_MIN + secrets.randbelow(SESSION_UID_MAX - SESSION_UID_MIN + 1)
        if uid in avoid_uids or _uid_exists(uid):
            continue
        try:
            _run(
                [
                    useradd,
                    "--no-create-home",
                    "--no-user-group",
                    "--uid",
                    str(uid),
                    "--home-dir",
                    home.as_posix(),
                    "--shell",
                    _login_shell(),
                    name,
                ],
                f"create SSH session account {name}",
            )
            return
        except ExecutionError as exc:
            detail = str(exc)
            if _account_exists(name):
                raise
    raise ExecutionError(
        f"Could not find a free uid for SSH session account {name}. {detail}".strip()
    )


def _delete_account(name: str) -> bool:
    userdel = shutil.which("userdel")
    if userdel is None:
        logger.warning("userdel is missing; leaving account %s behind", name)
        return False
    try:
        _run([userdel, name], f"delete SSH session account {name}")
    except ExecutionError:
        logger.warning("Failed to delete SSH session account %s", name)
        return False
    return True


def _terminate_uid(uid: int) -> bool:
    victims = _processes_of(uid)
    for proc in victims:
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.Error:
            continue
    if victims:
        psutil.wait_procs(victims, timeout=_KILL_GRACE_SEC)
    # A snapshot can miss a process that forks and exits in a loop, so it is
    # only trusted once kill(-1) has left the uid unable to start another.
    for _ in range(_KILL_ROUNDS):
        _kill_all_as(uid)
        survivors = _processes_of(uid)
        if not survivors:
            return True
        for proc in survivors:
            try:
                proc.send_signal(signal.SIGKILL)
            except psutil.Error:
                continue
        psutil.wait_procs(survivors, timeout=_KILL_ROUND_SEC)
    return not _processes_of(uid)


def _kill_all_as(uid: int) -> None:
    """Have the kernel SIGKILL every process of ``uid`` in a single pass.

    A snapshot of the process table cannot catch a process forked after it was
    taken, so a fork loop outruns one. ``kill(-1)`` sent as ``uid`` reaches all
    of that uid's processes at once, and a process that cannot be started as
    ``uid`` leaves the snapshot rounds to do what they can.
    """
    if os.geteuid() != 0 or uid in (0, os.getuid()):
        return
    result = _run_as(uid, _KILL_ALL_SCRIPT, [])
    if result is not None and result.returncode != 0:
        logger.debug(
            "Signalling every process of uid %d exited %d: %s",
            uid,
            result.returncode,
            _stderr_of(result),
        )


def _run_as(
    uid: int, script: str, args: list[str]
) -> "subprocess.CompletedProcess[bytes] | None":
    """Run ``_AS_UID_PREAMBLE`` then ``script`` in a new interpreter, with ``uid``,
    a gid and ``args`` on its argv; return ``None`` if it cannot start.

    The interpreter starts as the worker and the preamble switches to ``uid``,
    because ``subprocess``'s ``user=`` forces a plain ``fork()``, which gRPC's
    fork handlers crash.
    """
    try:
        return subprocess.run(  # nosec B603 - argv list, no shell=True, the worker's own interpreter
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                _AS_UID_PREAMBLE + script,
                str(uid),
                str(_NOGROUP_GID),
                *args,
            ],
            env={},
            cwd="/",
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_AS_UID_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("Failed to run a helper as uid %d", uid, exc_info=True)
        return None


def _stderr_of(result: "subprocess.CompletedProcess[bytes]") -> str:
    return result.stderr.decode("utf-8", errors="replace").strip()


def _processes_of(uid: int) -> list[psutil.Process]:
    return [p for p in psutil.process_iter(["uids", "status"]) if _owned_by(p, uid)]


def _uid_exists(uid: int) -> bool:
    try:
        pwd.getpwuid(uid)
    except KeyError:
        return False
    return True


def _account_exists(name: str) -> bool:
    try:
        pwd.getpwnam(name)
    except KeyError:
        return False
    return True


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        logger.debug("Failed to remove %s", path, exc_info=True)


def _owned_by(proc: psutil.Process, uid: int) -> bool:
    return _live_uid(proc) == uid


def _live_uid(proc: psutil.Process) -> int | None:
    """The real uid of ``proc``, or ``None`` once it has exited.

    A zombie is only an exit status waiting for its parent to reap it, and a
    PID 1 that never reaps would otherwise keep a session account alive
    forever.
    """
    try:
        if proc.status() == psutil.STATUS_ZOMBIE:
            return None
        return int(proc.uids().real)
    except (psutil.Error, AttributeError):
        return None


def _ensure_privsep_dir() -> None:
    """Create sshd's privilege-separation directory, which a root sshd needs."""
    try:
        PRIVSEP_DIR.mkdir(parents=True, exist_ok=True)
        os.chown(PRIVSEP_DIR, 0, 0)
        PRIVSEP_DIR.chmod(0o755)
    except OSError as exc:
        raise ExecutionError(
            f"Cannot prepare {PRIVSEP_DIR.as_posix()} for sshd: {exc}"
        ) from exc


def _login_shell() -> str:
    for candidate in ("/bin/bash", "/bin/sh"):
        if Path(candidate).exists():
            return candidate
    return "/bin/sh"


def _unlock(name: str) -> None:
    usermod = _require_binary("usermod")
    _run(
        [usermod, "--password", _unusable_password_hash(), name],
        f"unlock SSH session account {name}",
    )


def _unusable_password_hash() -> str:
    """A valid but unguessable hash, so sshd does not treat the account as locked."""
    openssl = shutil.which("openssl")
    if openssl is None:
        # "*" and a leading "!" both read as locked to sshd; a bare salted
        # marker does not, and no password hashes to it.
        return f"$6$nologin${secrets.token_hex(16)}"
    # token_hex, not token_urlsafe: the urlsafe alphabet includes "-", and a
    # value starting with one is parsed by openssl as an option.
    result = _run(
        [openssl, "passwd", "-6", secrets.token_hex(32)],
        "generate an unusable password hash",
    )
    return result.stdout.decode("utf-8", errors="replace").strip() or (
        f"$6$nologin${secrets.token_hex(16)}"
    )


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ExecutionError(
            f"{name} is required to give this SSH session its own account but is "
            "missing from the worker image"
        )
    return path


def _run(argv: list[str], what: str) -> "subprocess.CompletedProcess[bytes]":
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        argv, capture_output=True, timeout=_USERADD_TIMEOUT_SEC, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ExecutionError(f"Failed to {what}: {detail}")
    return result
