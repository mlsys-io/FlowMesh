"""POSIX ACL entries that keep a session account out of the worker's state.

A named-user ACL entry is checked before a file's "other" bits, so
``u:<uid>:---`` on a directory stops that one account from traversing into it
however permissive the modes below it are, and changes nothing for any other
principal sharing the directory.

Every entry applied is recorded before it is written, in a root-only state
file, so a worker that crashed mid-session can revoke exactly its own entries
and never one a peer worker applied to a shared volume.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from ..base_executor import ExecutionError

logger = logging.getLogger(__name__)

STATE_DIR = Path("/var/lib/flowmesh")
DENY_RECORD = STATE_DIR / "ssh-session-denies"
PROBE_UID = 65534
_ACL_TIMEOUT_SEC = 30.0
_DENIED_USER_RE = re.compile(r"^user:(\d+):---$")
_record_lock = threading.Lock()


def find_setfacl() -> str | None:
    return shutil.which("setfacl")


def find_getfacl() -> str | None:
    return shutil.which("getfacl")


def tools_available() -> bool:
    return find_setfacl() is not None and find_getfacl() is not None


def deny(uid: int, path: Path) -> None:
    """Deny ``uid`` every access to ``path``."""
    _setfacl("-m", f"u:{uid}:---", path)


def revoke(uid: int, path: Path) -> None:
    """Remove ``uid``'s entry from ``path``, and the mask when nothing needs it.

    A mask left behind would make a later ``chmod`` on ``path`` change the mask
    instead of the group bits, so it is dropped once no named entry remains;
    ``setfacl`` refuses that while one does, which is the case to leave alone.
    """
    _setfacl("-x", f"u:{uid}", path)
    setfacl = _require(find_setfacl(), "setfacl")
    subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [setfacl, "-x", "m::", "--", path.as_posix()],
        capture_output=True,
        timeout=_ACL_TIMEOUT_SEC,
        check=False,
    )


def denied_uids(path: Path) -> set[int]:
    """Uids that ``path``'s access ACL denies everything."""
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
    return parse_denied_uids(result.stdout.decode("utf-8", errors="replace"))


def parse_denied_uids(getfacl_output: str) -> set[int]:
    return {
        int(match.group(1))
        for line in getfacl_output.splitlines()
        if (match := _DENIED_USER_RE.match(line.strip()))
    }


def probe(directory: Path) -> None:
    """Check that the filesystem backing ``directory`` stores a deny entry."""
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


def _setfacl(action: str, entry: str, path: Path) -> None:
    setfacl = _require(find_setfacl(), "setfacl")
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [setfacl, action, entry, "--", path.as_posix()],
        capture_output=True,
        timeout=_ACL_TIMEOUT_SEC,
        check=False,
    )
    if result.returncode != 0:
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
