import errno
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from shared.schemas.artifact import ArtifactContext
from shared.schemas.result import BaseExecutorResult, ResultEnvelope, write_result
from shared.utils.result_delivery import (
    RECEIPT_NAME,
    artifacts_ready,
    commit_delivery,
    create_delivery_bundle,
    delivery_lock,
    extract_delivery_bundle,
    make_receipt,
    read_receipt,
    write_receipt,
)


def populate(
    root: Path,
    content: bytes = b"model",
    task_id: str = "tsk-up",
    dispatch_id: str | None = None,
) -> Path:
    base = root / task_id
    write_result(
        root,
        ResultEnvelope(
            task_id=task_id,
            result=BaseExecutorResult.model_validate(
                {
                    "model": {"path": "model"},
                    "_artifacts": ArtifactContext(base_dir=base.resolve().as_posix()),
                }
            ),
            metadata={"independent_results": True, "result_dispatch": dispatch_id},
        ),
    )
    (base / "artifacts" / "model" / "empty").mkdir(parents=True)
    (base / "artifacts" / "model" / "weights").write_bytes(content)
    (base / "artifacts" / "unused-checkpoint").write_bytes(b"unused")
    write_receipt(base, make_receipt(base, task_id, None))
    return base


@pytest.mark.parametrize("paths", [None, [], ["model"], ["model/weights"]])
def test_selection_transfer_is_complete_and_preserves_empty_directories(
    tmp_path: Path, paths: list[str] | None
) -> None:
    producer = populate(tmp_path / "producer")
    bundle = create_delivery_bundle(producer, "tsk-up", paths)
    try:
        staging = extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")
        destination = tmp_path / "server" / "tsk-up"
        commit_delivery(staging, destination, "tsk-up")
        assert artifacts_ready(destination, "tsk-up", paths)
        assert artifacts_ready(destination, "tsk-up") is (paths is None)
        assert (destination / "artifacts" / "unused-checkpoint").exists() is (
            paths is None
        )
        if paths is None or paths == ["model"]:
            assert (destination / "artifacts" / "model" / "empty").is_dir()
    finally:
        bundle.unlink()


def test_partial_directory_and_corrupt_file_are_not_ready(tmp_path: Path) -> None:
    base = populate(tmp_path)
    (base / RECEIPT_NAME).unlink()
    assert not artifacts_ready(base, "tsk-up")
    write_receipt(base, make_receipt(base, "tsk-up", None))
    (base / "artifacts" / "model" / "weights").write_bytes(b"broken")
    assert not artifacts_ready(base, "tsk-up", ["model"])


def test_repeated_selective_delivery_preserves_full_shared_snapshot(
    tmp_path: Path,
) -> None:
    base = populate(tmp_path / "shared")
    bundle = create_delivery_bundle(base, "tsk-up", ["model"])
    try:
        staging = extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")
        commit_delivery(staging, base, "tsk-up")
        assert artifacts_ready(base, "tsk-up")
        assert (base / "artifacts" / "unused-checkpoint").read_bytes() == b"unused"
    finally:
        bundle.unlink()


def test_new_snapshot_does_not_reuse_stale_files(tmp_path: Path) -> None:
    destination = populate(tmp_path / "cache", b"old")
    old = read_receipt(destination)
    producer = populate(tmp_path / "producer", b"new")
    bundle = create_delivery_bundle(producer, "tsk-up", ["model"])
    try:
        staging = extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")
        commit_delivery(staging, destination, "tsk-up")
        assert old is not None
        assert not artifacts_ready(destination, "tsk-up", ["model"], old.generation)
        assert not (destination / "artifacts" / "unused-checkpoint").exists()
        assert (destination / "artifacts" / "model" / "weights").read_bytes() == b"new"
    finally:
        bundle.unlink()


def test_failed_incomplete_transfer_does_not_replace_ready_snapshot(
    tmp_path: Path,
) -> None:
    destination = populate(tmp_path / "server", b"old")
    staging = populate(tmp_path / "incoming", b"new")
    (staging / "artifacts" / "model" / "weights").unlink()
    with pytest.raises(ValueError, match="Incomplete artifact"):
        commit_delivery(staging, destination, "tsk-up")
    assert artifacts_ready(destination, "tsk-up")
    assert (destination / "artifacts" / "model" / "weights").read_bytes() == b"old"


def test_delivery_commit_handles_separate_staging_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = populate(tmp_path / "incoming")
    destination = tmp_path / "server" / "tsk-up"
    replace = Path.replace

    def require_same_filesystem(source: Path, target: Path) -> Path:
        if source.is_relative_to(staging):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        assert source.parent == target.parent
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", require_same_filesystem)
    commit_delivery(staging, destination, "tsk-up")
    assert artifacts_ready(destination, "tsk-up")
    assert (destination / "artifacts" / "model" / "weights").read_bytes() == b"model"


def test_commit_waits_for_snapshot_reader_lock(tmp_path: Path) -> None:
    destination = populate(tmp_path / "server", b"old")
    staging = populate(tmp_path / "incoming", b"new")
    started = Event()

    def replace() -> None:
        started.set()
        commit_delivery(staging, destination, "tsk-up")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with delivery_lock(destination):
            future = executor.submit(replace)
            assert started.wait(5)
            assert not future.done()
            assert (
                destination / "artifacts" / "model" / "weights"
            ).read_bytes() == b"old"
        future.result(timeout=5)
    assert artifacts_ready(destination, "tsk-up")
    assert (destination / "artifacts" / "model" / "weights").read_bytes() == b"new"


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", "file"),
        ("tsk-up/artifacts/link", "symlink"),
        ("tsk-up/.delivery.lock", "file"),
    ],
)
def test_ingest_rejects_unsafe_and_unexpected_members(
    tmp_path: Path, name: str, kind: str
) -> None:
    producer = populate(tmp_path / "producer")
    bundle = create_delivery_bundle(producer, "tsk-up", None)
    try:
        with tarfile.open(bundle, "a") as archive:
            info = tarfile.TarInfo(name)
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
            archive.addfile(info)
        with pytest.raises(ValueError):
            extract_delivery_bundle(bundle, tmp_path / "staging", "tsk-up")
    finally:
        bundle.unlink()


def test_symlinked_selection_parent_cannot_escape_artifacts(tmp_path: Path) -> None:
    base = populate(tmp_path / "producer")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"secret")
    (base / "artifacts" / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        create_delivery_bundle(base, "tsk-up", ["link/secret"])


def test_complete_receipt_does_not_cover_absent_selections(tmp_path: Path) -> None:
    base = populate(tmp_path)
    assert not artifacts_ready(base, "tsk-up", ["missing"])
    write_receipt(base, make_receipt(base, "tsk-up", ["model"]))
    assert not artifacts_ready(base, "tsk-up", ["model/missing"])


def test_delivery_directories_remain_writable_by_peer_uids(tmp_path: Path) -> None:
    source = populate(tmp_path / "producer")
    destination = tmp_path / "server" / "tsk-up"
    commit_delivery(source, destination, "tsk-up")
    for name in ("artifacts", "artifacts/model", "artifacts/model/empty"):
        assert (destination / name).stat().st_mode & 0o777 == 0o777
