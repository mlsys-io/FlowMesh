import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from worker.utils.logging import PrivateRotatingFileHandler


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def permissive_umask() -> Iterator[None]:
    previous = os.umask(0o000)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.usefixtures("permissive_umask")
class TestPrivateRotatingFileHandler:
    def test_a_new_log_is_owner_only(self, tmp_path: Path) -> None:
        log = tmp_path / "worker.log"
        handler = PrivateRotatingFileHandler(log.as_posix(), maxBytes=1024)
        handler.close()
        assert _mode(log) == 0o600

    def test_an_existing_log_and_its_backups_are_tightened(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "worker.log"
        backup = tmp_path / "worker.log.1"
        for path in (log, backup):
            path.write_text("earlier run")
            path.chmod(0o644)
        handler = PrivateRotatingFileHandler(
            log.as_posix(), maxBytes=1024, backupCount=2
        )
        handler.close()
        assert _mode(log) == 0o600
        assert _mode(backup) == 0o600
        assert log.read_text() == "earlier run"

    def test_rotation_keeps_every_file_owner_only(self, tmp_path: Path) -> None:
        log = tmp_path / "worker.log"
        handler = PrivateRotatingFileHandler(log.as_posix(), maxBytes=64, backupCount=2)
        logger = logging.getLogger("test_private_rotating_file_handler")
        logger.propagate = False
        logger.addHandler(handler)
        try:
            for index in range(20):
                logger.warning("line %d of the worker log", index)
        finally:
            logger.removeHandler(handler)
            handler.close()
        assert (tmp_path / "worker.log.2").exists()
        for path in tmp_path.iterdir():
            assert _mode(path) == 0o600, path
