"""Atomic file-write primitives for files written by multiple parties."""

import os
import tempfile
from pathlib import Path

_SHARED_FILE_MODE = 0o0666


def atomic_write_text(target: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Replace ``target`` with ``content`` atomically via tempfile + os.replace.

    The writer only needs write permission on the parent directory, not on
    any pre-existing file (which may be owned by a different UID under a
    shared results volume). The new file is chmodded to 0o0666 so a peer
    UID can replace it on the next call.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        tmp_path.chmod(_SHARED_FILE_MODE)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
