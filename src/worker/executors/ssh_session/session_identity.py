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
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import psutil

from ..base_executor import ExecutionError

logger = logging.getLogger(__name__)

ACCOUNT_PREFIX = "fmssn"
ACCOUNT_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
PRIVSEP_DIR = Path("/run/sshd")
_KILL_GRACE_SEC = 5.0
_USERADD_TIMEOUT_SEC = 30.0


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

    @property
    @abstractmethod
    def isolates_from_worker(self) -> bool:
        """Whether the session is a different principal from the worker."""

    def own(self, path: Path, mode: int | None = None, recursive: bool = False) -> None:
        """Hand ``path`` to the session so it can read or write it."""
        return None

    def release(self) -> None:
        return None


class CurrentUser(SessionIdentity):
    """The worker's own account: no separation, used when the worker is not root."""

    def __init__(self) -> None:
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
        self.name = name
        self.uid = uid
        self.gid = gid
        self.home = home

    @property
    def isolates_from_worker(self) -> bool:
        return True

    @classmethod
    def create(cls, name: str, home: Path) -> "DedicatedAccount":
        useradd = _require_binary("useradd")
        home.mkdir(parents=True, exist_ok=True)
        _run(
            [
                useradd,
                "--no-create-home",
                "--no-user-group",
                "--home-dir",
                home.as_posix(),
                "--shell",
                _login_shell(),
                name,
            ],
            f"create SSH session account {name}",
        )
        # A fresh account's shadow entry is "!", which sshd reads as locked and
        # refuses even for public-key auth once UsePAM is off. An unguessable
        # hash leaves it unlocked without granting a usable password.
        _unlock(name)
        try:
            entry = pwd.getpwnam(name)
        except KeyError as exc:
            raise ExecutionError(f"SSH session account {name} was not created") from exc
        account = cls(name, entry.pw_uid, entry.pw_gid, home)
        account.own(home, mode=0o700)
        return account

    def own(self, path: Path, mode: int | None = None, recursive: bool = False) -> None:
        targets = [path]
        if recursive and path.is_dir():
            targets.extend(path.rglob("*"))
        for target in targets:
            try:
                os.chown(target, self.uid, self.gid)
                if mode is not None and target == path:
                    target.chmod(mode)
            except OSError:
                logger.debug(
                    "Failed to hand %s to %s", target, self.name, exc_info=True
                )

    def release(self) -> None:
        self._kill_processes()
        userdel = shutil.which("userdel")
        if userdel is None:
            logger.warning("userdel is missing; leaving account %s behind", self.name)
            return
        try:
            _run([userdel, self.name], f"delete SSH session account {self.name}")
        except ExecutionError:
            logger.warning("Failed to delete SSH session account %s", self.name)

    def _kill_processes(self) -> None:
        victims = [p for p in psutil.process_iter(["uids"]) if _owned_by(p, self.uid)]
        for proc in victims:
            try:
                proc.send_signal(signal.SIGTERM)
            except psutil.Error:
                continue
        if not victims:
            return
        _, alive = psutil.wait_procs(victims, timeout=_KILL_GRACE_SEC)
        for proc in alive:
            try:
                proc.send_signal(signal.SIGKILL)
            except psutil.Error:
                continue


def resolve_identity(session_id: str, session_dir: Path) -> SessionIdentity:
    """Pick the strongest identity this worker can give a session."""
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
    return DedicatedAccount.create(account_name_for(session_id), session_dir / "home")


def reap_stale_accounts(keep: str | None = None) -> None:
    """Delete session accounts left behind by an unclean worker exit."""
    if os.getuid() != 0:
        return
    userdel = shutil.which("userdel")
    if userdel is None:
        return
    for entry in pwd.getpwall():
        name = entry.pw_name
        if not name.startswith(ACCOUNT_PREFIX) or name == keep:
            continue
        if any(_owned_by(p, entry.pw_uid) for p in psutil.process_iter(["uids"])):
            continue
        try:
            _run([userdel, name], f"reap stale SSH session account {name}")
            logger.info("Reaped stale SSH session account %s", name)
        except ExecutionError:
            logger.debug("Could not reap stale account %s", name, exc_info=True)


def _owned_by(proc: psutil.Process, uid: int) -> bool:
    try:
        return proc.uids().real == uid
    except (psutil.Error, AttributeError):
        return False


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
    result = _run(
        [openssl, "passwd", "-6", secrets.token_urlsafe(32)],
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
