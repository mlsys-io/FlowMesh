"""POSIX ACL entries that keep a session account out of the worker's state.

A named-user ACL entry is checked before a file's "other" bits, so
``u:<uid>:---`` on a directory stops that one account from traversing into it
however permissive the modes below it are, and changes nothing for any other
principal sharing the directory.

Every entry applied is recorded before it is written, in a root-only state
file, so a worker that crashed mid-session can revoke exactly its own entries
and never one a peer worker applied to a shared volume.
"""

import contextlib
import fcntl
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..base_executor import ExecutionError

logger = logging.getLogger(__name__)

STATE_DIR = Path("/var/lib/flowmesh")
DENY_RECORD = STATE_DIR / "ssh-session-denies"
PROBE_UID = 65534
_ACL_TIMEOUT_SEC = 30.0
_DENIED_USER_RE = re.compile(r"^user:(\d+):---$")
_NAMED_USER_RE = re.compile(r"^(?:default:)?user:(\d+):")
_NAMED_ACCESS_ENTRY_RE = re.compile(r"^(?:user|group):[^:]+:")
_PERMS = "rwx"
_LOCK_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_LOCK_TIMEOUT_SEC = 30.0
_LOCK_RETRY_SEC = 0.1
_record_lock = threading.Lock()


def find_setfacl() -> str | None:
    return shutil.which("setfacl")


def find_getfacl() -> str | None:
    return shutil.which("getfacl")


def tools_available() -> bool:
    return find_setfacl() is not None and find_getfacl() is not None


def deny(uid: int, path: Path) -> None:
    """Deny ``uid`` every access to ``path``, leaving the mask as it was."""
    _setfacl(path, "-n", "-m", f"u:{uid}:---")


def revoke(uid: int, path: Path) -> None:
    """Remove ``uid``'s entry from ``path``, and the mask when nothing needs it.

    A mask left behind would make a later ``chmod`` on ``path`` change the mask
    instead of the group bits, so it is dropped once no named entry remains and
    it grants the group everything the group entry does. A mask narrower than
    the group entry was set by someone else and is kept. Dropping it is best
    effort, since another worker may add an entry in between.
    """
    _setfacl(path, "-n", "-x", f"u:{uid}")
    if mask_is_redundant(_read_acl(path)):
        _setfacl(path, "-x", "m::", check=False)


def mask_is_redundant(getfacl_output: str) -> bool:
    """Whether dropping the access ACL's mask leaves every permission unchanged."""
    group = mask = None
    for line in getfacl_output.splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry.startswith("group::"):
            group = entry.removeprefix("group::")
        elif entry.startswith("mask::"):
            mask = entry.removeprefix("mask::")
        elif _NAMED_ACCESS_ENTRY_RE.match(entry):
            return False
    if group is None or mask is None:
        return False
    return all(perm in mask for perm in group if perm in _PERMS)


@contextlib.contextmanager
def locked(
    paths: Iterable[Path], timeout_sec: float = _LOCK_TIMEOUT_SEC
) -> Iterator[None]:
    """Hold an exclusive ``flock`` on each directory in ``paths``, waiting at
    most ``timeout_sec`` for all of them.

    Workers sharing a directory, in other containers on the same kernel too,
    serialize on it. Each inode is locked once, in inode order, so two holders
    cannot deadlock.
    """
    fds: list[int] = []
    try:
        try:
            by_inode: dict[tuple[int, int], int] = {}
            for path in paths:
                fd = os.open(path, _LOCK_FLAGS)
                fds.append(fd)
                info = os.fstat(fd)
                by_inode.setdefault((info.st_dev, info.st_ino), fd)
            deadline = time.monotonic() + timeout_sec
            for _, fd in sorted(by_inode.items()):
                _flock_until(fd, deadline)
        except OSError as exc:
            raise ExecutionError(
                f"Cannot lock worker state to add an ACL entry: {exc}",
                retryable=True,
            ) from exc
        yield
    finally:
        for fd in fds:
            os.close(fd)


def _flock_until(fd: int, deadline: float) -> None:
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_LOCK_RETRY_SEC)


def denied_uids(path: Path) -> set[int]:
    """Return the uids that ``path``'s access ACL denies all permissions."""
    return parse_denied_uids(_read_acl(path))


def named_uids(path: Path) -> set[int]:
    """Return the uids named by a user entry in ``path``'s access or default ACL."""
    return parse_named_uids(_read_acl(path))


def parse_denied_uids(getfacl_output: str) -> set[int]:
    return _parse_uids(_DENIED_USER_RE, getfacl_output)


def parse_named_uids(getfacl_output: str) -> set[int]:
    return _parse_uids(_NAMED_USER_RE, getfacl_output)


def _parse_uids(pattern: re.Pattern[str], getfacl_output: str) -> set[int]:
    return {
        int(match.group(1))
        for line in getfacl_output.splitlines()
        if (match := pattern.match(line.strip()))
    }


def _read_acl(path: Path) -> str:
    getfacl = _require(find_getfacl(), "getfacl")
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [getfacl, "-n", "-c", "-p", "--", path.as_posix()],
        capture_output=True,
        timeout=_ACL_TIMEOUT_SEC,
        check=False,
    )
    if result.returncode != 0:
        raise ExecutionError(
            f"Failed to read the ACL of {path.as_posix()}: {_stderr(result)}"
        )
    return result.stdout.decode("utf-8", errors="replace")


def probe(directory: Path) -> None:
    """Check that the filesystem backing ``directory`` stores a deny entry and
    takes the lock :func:`locked` holds."""
    fd, name = tempfile.mkstemp(prefix=".flowmesh-acl-probe-", dir=directory)
    os.close(fd)
    target = Path(name)
    try:
        deny(PROBE_UID, target)
        if PROBE_UID not in denied_uids(target):
            raise ExecutionError(
                f"ACL entries do not persist on the filesystem of {directory}"
            )
    finally:
        target.unlink(missing_ok=True)
    dir_fd = os.open(directory, _LOCK_FLAGS)
    try:
        fcntl.flock(dir_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    except OSError as exc:
        raise ExecutionError(
            f"Directories on the filesystem of {directory} cannot be locked: {exc}"
        ) from exc
    finally:
        os.close(dir_fd)


def record(uid: int, path: Path) -> None:
    """Note that ``uid`` is about to be denied ``path``."""
    with _record_lock:
        entries = _read_records()
        entries.add((uid, path.as_posix()))
        _write_records(entries)


def forget(uid: int, path: Path) -> None:
    with _record_lock:
        entries = _read_records()
        entries.discard((uid, path.as_posix()))
        _write_records(entries)


def recorded() -> set[tuple[int, str]]:
    with _record_lock:
        return _read_records()


def _read_records() -> set[tuple[int, str]]:
    try:
        text = DENY_RECORD.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise ExecutionError(f"Cannot read {DENY_RECORD.as_posix()}: {exc}") from exc
    entries: set[tuple[int, str]] = set()
    for line in text.splitlines():
        uid, sep, path = line.partition("\t")
        if sep and uid.isdigit() and path:
            entries.add((int(uid), path))
    return entries


def _write_records(entries: set[tuple[int, str]]) -> None:
    try:
        STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".ssh-session-denies-", dir=STATE_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(f"{uid}\t{path}\n" for uid, path in sorted(entries))
        os.replace(tmp_name, DENY_RECORD)
    except OSError as exc:
        raise ExecutionError(
            f"Cannot record SSH session ACL entries in {DENY_RECORD.as_posix()}: "
            f"{exc}"
        ) from exc


def _setfacl(path: Path, *options: str, check: bool = True) -> None:
    setfacl = _require(find_setfacl(), "setfacl")
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [setfacl, *options, "--", path.as_posix()],
        capture_output=True,
        timeout=_ACL_TIMEOUT_SEC,
        check=False,
    )
    if check and result.returncode != 0:
        raise ExecutionError(
            f"Failed to update the ACL of {path.as_posix()}: {_stderr(result)}"
        )


def _require(path: str | None, name: str) -> str:
    if path is None:
        raise ExecutionError(
            f"{name} is required to isolate SSH sessions but is missing from the "
            "worker image"
        )
    return path


def _stderr(result: "subprocess.CompletedProcess[bytes]") -> str:
    return result.stderr.decode("utf-8", errors="replace").strip()
