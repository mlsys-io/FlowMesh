"""Tests for the shared manifest helpers."""

import hashlib
import io
import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from shared.utils import atomic, manifest
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

    def test_a_sync_never_overwrites_a_later_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write = manifest.atomic_write_text
        first_scanned = threading.Event()
        later_wrote = threading.Event()

        def _held_first_write(target: Path, content: str, **kwargs: Any) -> None:
            if not first_scanned.is_set():
                # The first sync has scanned; hold its write until the later sync
                # writes, or briefly when the later sync cannot run alongside it.
                first_scanned.set()
                later_wrote.wait(0.5)
            else:
                later_wrote.set()
            write(target, content, **kwargs)

        monkeypatch.setattr(manifest, "atomic_write_text", _held_first_write)
        (tmp_path / "a.txt").write_text("a")
        first = threading.Thread(target=sync_manifest, args=(tmp_path, "t", []))
        first.start()
        assert first_scanned.wait(5)
        (tmp_path / "b.txt").write_text("b")
        later = threading.Thread(target=sync_manifest, args=(tmp_path, "t", []))
        later.start()
        first.join(5)
        later.join(5)

        paths = {
            e["path"]
            for e in json.loads((tmp_path / MANIFEST_NAME).read_text())["entries"]
        }
        assert {"a.txt", "b.txt"} <= paths

    def test_a_directory_lock_lives_only_while_a_sync_holds_it(
        self, tmp_path: Path
    ) -> None:
        for index in range(50):
            sync_manifest(tmp_path / f"task-{index}", f"t{index}", [])

        assert len(manifest._MANIFEST_LOCKS) == 0


def test_a_scan_skips_in_flight_writes_and_files_removed_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "kept.bin").write_bytes(b"x" * 3)
    in_flight = Path(tempfile.mkstemp(prefix=".fm-tmp-", dir=artifacts)[1])
    in_flight.write_bytes(b"partial")
    assert atomic.is_atomic_temp(in_flight.name)
    (tmp_path / in_flight.name).write_bytes(b"partial")
    is_file = Path.is_file

    # A file the scan sees, then finds gone: renamed or removed between the two.
    def _removed_after_listing(self: Path) -> bool:
        return self.name == "gone.bin" or is_file(self)

    monkeypatch.setattr(Path, "is_file", _removed_after_listing)
    monkeypatch.setattr(Path, "rglob", _with_extra(artifacts / "gone.bin"))

    entries = {e["path"]: e for e in sync_manifest(tmp_path, "t", [])["entries"]}

    assert in_flight.name not in entries
    assert entries["artifacts"]["file_count"] == 1
    assert entries["artifacts"]["size"] == 3


def _with_extra(extra: Path) -> Any:
    rglob = Path.rglob

    def _rglob(self: Path, pattern: str) -> Any:
        yield from rglob(self, pattern)
        if self == extra.parent:
            yield extra

    return _rglob


def test_a_long_filename_writes_atomically(tmp_path: Path) -> None:
    target = tmp_path / ("a" * 250)

    atomic.atomic_write_stream(target, io.BytesIO(b"data"))

    assert target.read_bytes() == b"data"


def test_an_existing_entry_that_is_not_a_file_is_present(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")

    budget = manifest._HashBudget(manifest._HASH_BUDGET_BYTES)
    entry = manifest._describe_path(tmp_path, Path("pipe"), budget, required=False)
    gone = manifest._describe_path(tmp_path, Path("gone.bin"), budget, required=False)

    assert (entry["status"], entry["size"], entry["file_count"]) == ("present", 0, 0)
    assert gone["status"] == "missing"


_BUDGET = 256 << 20


class _CountingHashlib:
    """``hashlib`` for the manifest module, counting the bytes it digests."""

    def __init__(self) -> None:
        self.digested = 0

    def sha256(self) -> Any:
        counter = self
        real = hashlib.sha256()

        class _Counting:
            def update(self, data: bytes) -> None:
                counter.digested += len(data)
                real.update(data)

            def hexdigest(self) -> str:
                return real.hexdigest()

        return _Counting()


def _sparse(path: Path, size: int) -> None:
    with path.open("wb") as fh:
        fh.truncate(size)


def test_a_file_over_the_hash_budget_is_listed_by_size_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = _CountingHashlib()
    monkeypatch.setattr(manifest, "hashlib", counting)
    _sparse(tmp_path / "huge.bin", 8 << 30)

    entries = {e["path"]: e for e in sync_manifest(tmp_path, "t", [])["entries"]}

    assert entries["huge.bin"]["size"] == 8 << 30
    assert "sha256" not in entries["huge.bin"]
    assert counting.digested == 0


def test_a_sync_digests_at_most_its_budget_in_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = _CountingHashlib()
    monkeypatch.setattr(manifest, "hashlib", counting)
    for index in range(10):
        _sparse(tmp_path / f"f{index}.bin", 60 << 20)
    _sparse(tmp_path / "f9.bin", 1 << 20)

    entries = {e["path"]: e for e in sync_manifest(tmp_path, "t", [])["entries"]}

    digested = [f"f{i}.bin" for i in range(10) if "sha256" in entries[f"f{i}.bin"]]
    assert digested == ["f0.bin", "f1.bin", "f2.bin", "f3.bin", "f9.bin"]
    assert counting.digested <= _BUDGET
