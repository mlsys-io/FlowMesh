"""Tests for the shared manifest helpers."""

import json
import stat
import threading
from pathlib import Path

import pytest

from shared.utils.manifest import (
    ARTIFACTS_DIR,
    LOGS_DIR,
    MANIFEST_NAME,
    SCRATCH_DIR,
    prepare_output_dir,
    scratch_dir,
    sync_manifest,
)


def _race(fn, *, threads: int = 8) -> list[BaseException]:
    """Run ``fn`` on ``threads`` threads released simultaneously."""
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - recorded for assertion
            with lock:
                errors.append(exc)

    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    return errors


class TestPrepareOutputDir:
    def test_directories_are_world_writable(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        prepare_output_dir(out)
        for d in (out, out / LOGS_DIR, out / ARTIFACTS_DIR):
            assert stat.S_IMODE(d.stat().st_mode) == 0o0777

    def test_is_idempotent(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        prepare_output_dir(out)
        prepare_output_dir(out)
        for d in (out, out / LOGS_DIR, out / ARTIFACTS_DIR):
            assert stat.S_IMODE(d.stat().st_mode) == 0o0777

    def test_existing_directory_is_remoded(self, tmp_path: Path) -> None:
        """A directory materialized by another writer still ends world-writable.

        Merged-child mirroring creates the task directory with ``copytree``,
        which copies the source mode. Ingest must still be able to hand write
        access to peer worker UIDs.
        """
        out = tmp_path / "task-out"
        out.mkdir(mode=0o0755)
        prepare_output_dir(out)
        assert stat.S_IMODE(out.stat().st_mode) == 0o0777

    def test_concurrent_calls_do_not_raise(self, tmp_path: Path) -> None:
        """Regression: check-then-create raced between server threads.

        The results volume is written by the event loop (result ingest) and the
        ``tasks-events`` thread (merged-child mirroring) at once. The loser of
        the race used to surface as
        ``500 Failed to store result: [Errno 17] File exists: <results>/<task>``.
        """
        out = tmp_path / "tsk-concurrent"
        assert _race(lambda: prepare_output_dir(out)) == []
        for d in (out, out / LOGS_DIR, out / ARTIFACTS_DIR):
            assert stat.S_IMODE(d.stat().st_mode) == 0o0777

    def test_rejects_non_directory_at_path(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        out.write_text("not a directory")
        with pytest.raises(FileExistsError):
            prepare_output_dir(out)


class TestScratchDir:
    def test_creates_world_writable_dir(self, tmp_path: Path) -> None:
        path = scratch_dir(tmp_path)
        assert path == tmp_path / SCRATCH_DIR
        assert stat.S_IMODE(path.stat().st_mode) == 0o0777

    def test_concurrent_calls_do_not_raise(self, tmp_path: Path) -> None:
        assert _race(lambda: scratch_dir(tmp_path)) == []
        assert stat.S_IMODE((tmp_path / SCRATCH_DIR).stat().st_mode) == 0o0777


class TestSyncManifest:
    def test_manifest_is_group_and_other_writable(self, tmp_path: Path) -> None:
        sync_manifest(tmp_path, "t-1", expected=[])
        mode = stat.S_IMODE((tmp_path / MANIFEST_NAME).stat().st_mode)
        assert mode & 0o0066 == 0o0066

    def test_second_call_overwrites_first(self, tmp_path: Path) -> None:
        sync_manifest(tmp_path, "t-1", expected=[])
        sync_manifest(tmp_path, "t-2", expected=[])
        data = json.loads((tmp_path / MANIFEST_NAME).read_text())
        assert data["task_id"] == "t-2"
