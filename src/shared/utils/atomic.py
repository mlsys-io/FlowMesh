"""Atomic file-write primitives for files written by multiple parties."""

import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, BinaryIO

_SHARED_FILE_MODE = 0o0666
_COPY_CHUNK_BYTES = 1 << 20
_TEMP_PREFIX = ".fm-tmp-"


def is_atomic_temp(name: str) -> bool:
    """Return whether ``name`` is an in-flight atomic write's temp file."""
    return name.startswith(_TEMP_PREFIX)


def atomic_write_text(target: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Replace ``target`` with ``content`` atomically via tempfile + os.replace.

    The writer only needs write permission on the parent directory, not on
    any pre-existing file (which may be owned by a different UID under a
    shared results volume). The new file is chmodded to 0o0666 so a peer
    UID can replace it on the next call.
    """
    _atomic_replace(target, lambda fh: fh.write(content.encode(encoding)))


def atomic_write_stream(
    target: Path,
    source: BinaryIO,
    *,
    commit: Callable[[], AbstractContextManager[Any]] | None = None,
) -> None:
    """Replace ``target`` with the rest of ``source`` atomically, copying it in
    bounded chunks; creates the parent directory when missing.

    ``commit``, when given, is a context manager factory entered around the final
    rename only, not while the data is copied.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace(
        target,
        lambda fh: shutil.copyfileobj(source, fh, _COPY_CHUNK_BYTES),
        commit,
    )


def _atomic_replace(
    target: Path,
    write: Callable[[BinaryIO], Any],
    commit: Callable[[], AbstractContextManager[Any]] | None = None,
) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=_TEMP_PREFIX, dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        tmp_path.chmod(_SHARED_FILE_MODE)
        with commit() if commit is not None else nullcontext():
            os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
