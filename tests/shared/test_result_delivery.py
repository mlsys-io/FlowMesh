import errno
import io
import os
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from shared.schemas.artifact import ArtifactContext
from shared.schemas.result import BaseExecutorResult, ResultEnvelope, write_result
from shared.schemas.result_delivery import DeliveredFile
from shared.utils import result_delivery
from shared.utils.result_delivery import (
    RECEIPT_NAME,
    artifacts_ready,
    commit_delivery,
    create_delivery_bundle,
    delivery_lock,
    extract_delivery_bundle,
    make_receipt,
    read_receipt,
    validate_receipt,
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


@pytest.mark.parametrize("verify_content", [True, False])
def test_a_receipt_entry_under_an_outward_link_is_rejected_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verify_content: bool
) -> None:
    base = populate(tmp_path / "producer")
    receipt = make_receipt(base, "tsk-up", None)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"secret")
    (base / "artifacts" / "link").symlink_to(outside)
    receipt.files["artifacts/link/secret"] = DeliveredFile(size=6, sha256="0" * 64)
    describe_file = result_delivery.describe_file

    def describe_inside(path: Path) -> DeliveredFile:
        if outside in path.resolve().parents:
            pytest.fail(f"read {path} before confining it")
        return describe_file(path)

    monkeypatch.setattr(result_delivery, "describe_file", describe_inside)
    with pytest.raises(ValueError):
        validate_receipt(base, receipt, "tsk-up", verify_content=verify_content)


def test_a_listed_directory_replaced_by_an_outward_link_is_incomplete(
    tmp_path: Path,
) -> None:
    base = populate(tmp_path / "producer")
    outside = tmp_path / "outside"
    outside.mkdir()
    empty = base / "artifacts" / "model" / "empty"
    empty.rmdir()
    empty.symlink_to(outside)
    assert not artifacts_ready(base, "tsk-up", verify_content=False)


def test_a_listed_file_replaced_by_a_link_is_incomplete(tmp_path: Path) -> None:
    base = populate(tmp_path)
    weights = base / "artifacts" / "model" / "weights"
    weights.unlink()
    weights.symlink_to("empty")
    assert not artifacts_ready(base, "tsk-up", verify_content=False)


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


def test_symlinks_are_delivered_as_links_and_special_files_are_skipped(
    tmp_path: Path,
) -> None:
    producer = populate(tmp_path / "producer")
    venv = producer / "artifacts" / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to("/usr/bin/python3")
    (producer / "artifacts" / "latest").symlink_to("model")
    os.mkfifo(producer / "artifacts" / "pipe")
    write_receipt(producer, make_receipt(producer, "tsk-up", None))
    assert artifacts_ready(producer, "tsk-up")

    bundle = create_delivery_bundle(producer, "tsk-up", None)
    try:
        staging = extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")
        destination = tmp_path / "server" / "tsk-up"
        commit_delivery(staging, destination, "tsk-up")
    finally:
        bundle.unlink()

    assert artifacts_ready(destination, "tsk-up")
    python = destination / "artifacts" / "venv" / "bin" / "python"
    assert python.is_symlink() and os.readlink(python) == "/usr/bin/python3"
    assert os.readlink(destination / "artifacts" / "latest") == "model"
    assert not (destination / "artifacts" / "pipe").exists()


def test_a_changed_link_target_is_not_ready(tmp_path: Path) -> None:
    base = populate(tmp_path)
    (base / "artifacts" / "latest").symlink_to("model")
    write_receipt(base, make_receipt(base, "tsk-up", None))
    (base / "artifacts" / "latest").unlink()
    (base / "artifacts" / "latest").symlink_to("unused-checkpoint")
    assert not artifacts_ready(base, "tsk-up")


def test_bundle_member_beneath_a_link_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.tar"
    with tarfile.open(bundle, "w") as archive:
        link = tarfile.TarInfo("tsk-up/artifacts/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = str(tmp_path / "outside")
        archive.addfile(link)
        payload = tarfile.TarInfo("tsk-up/artifacts/escape/planted")
        payload.size = 1
        archive.addfile(payload, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe bundle member"):
        extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")
    assert not (tmp_path / "outside").exists()


def test_size_check_detects_incomplete_files_without_hashing(tmp_path: Path) -> None:
    base = populate(tmp_path)
    weights = base / "artifacts" / "model" / "weights"
    weights.write_bytes(b"MODEL")
    assert artifacts_ready(base, "tsk-up", verify_content=False)
    assert not artifacts_ready(base, "tsk-up")
    weights.write_bytes(b"mod")
    assert not artifacts_ready(base, "tsk-up", verify_content=False)


def test_recommit_of_the_same_snapshot_keeps_unchanged_files(tmp_path: Path) -> None:
    producer = populate(tmp_path / "producer")
    destination = tmp_path / "server" / "tsk-up"
    commit_delivery(producer, destination, "tsk-up")
    weights = destination / "artifacts" / "model" / "weights"
    inode = weights.stat().st_ino

    commit_delivery(producer, destination, "tsk-up")

    assert weights.stat().st_ino == inode
    assert artifacts_ready(destination, "tsk-up")


@pytest.mark.parametrize(
    "link_name", ["tsk-up/artifacts/./escape", "tsk-up/artifacts//escape"]
)
def test_unnormalized_link_names_cannot_hide_a_member_beneath_a_link(
    tmp_path: Path, link_name: str
) -> None:
    bundle = tmp_path / "bundle.tar"
    with tarfile.open(bundle, "w") as archive:
        for name, target in [
            (link_name, str(tmp_path / "outside")),
            ("tsk-up/artifacts/escape/planted", "/anywhere"),
        ]:
            link = tarfile.TarInfo(name)
            link.type = tarfile.SYMTYPE
            link.linkname = target
            archive.addfile(link)
    (tmp_path / "outside").mkdir()
    with pytest.raises(ValueError, match="Unsafe bundle member"):
        extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")
    assert not (tmp_path / "outside" / "planted").is_symlink()


def test_selected_link_travels_with_its_in_artifacts_target(tmp_path: Path) -> None:
    producer = populate(tmp_path / "producer")
    checkpoint = producer / "artifacts" / "checkpoint-500"
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(b"trained")
    (producer / "artifacts" / "final_model").symlink_to("checkpoint-500")
    (producer / "artifacts" / "latest").symlink_to("final_model")
    write_receipt(producer, make_receipt(producer, "tsk-up", None))

    bundle = create_delivery_bundle(producer, "tsk-up", ["latest"])
    try:
        staging = extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")
        destination = tmp_path / "consumer" / "tsk-up"
        commit_delivery(staging, destination, "tsk-up")
    finally:
        bundle.unlink()

    assert artifacts_ready(destination, "tsk-up", ["latest"])
    assert (destination / "artifacts" / "latest" / "weights").read_bytes() == b"trained"
    assert not (destination / "artifacts" / "unused-checkpoint").exists()


@pytest.mark.parametrize("name", ["tsk-up/logs", "tsk-up/results.json"])
def test_links_outside_artifacts_are_rejected(tmp_path: Path, name: str) -> None:
    bundle = tmp_path / "bundle.tar"
    with tarfile.open(bundle, "w") as archive:
        link = tarfile.TarInfo(name)
        link.type = tarfile.SYMTYPE
        link.linkname = str(tmp_path / "outside")
        archive.addfile(link)
    with pytest.raises(ValueError, match="Unsafe bundle member"):
        extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")


def test_commit_refuses_a_staged_link_outside_artifacts(tmp_path: Path) -> None:
    staging = populate(tmp_path / "staging")
    shutil.rmtree(staging / "logs", ignore_errors=True)
    (staging / "logs").symlink_to(tmp_path / "outside")
    destination = tmp_path / "server" / "tsk-up"
    with pytest.raises(ValueError, match="Unsupported delivery link"):
        commit_delivery(staging, destination, "tsk-up")
    assert not (destination / "logs").is_symlink()


def test_absolute_links_into_own_artifacts_become_portable(tmp_path: Path) -> None:
    producer = populate(tmp_path / "producer")
    # Written in the producer's mount namespace, which the reader does not share.
    (producer / "artifacts" / "latest").symlink_to(
        "/worker-results/tsk-up/artifacts/model"
    )
    write_receipt(producer, make_receipt(producer, "tsk-up", None))
    receipt = read_receipt(producer)
    assert receipt is not None and receipt.symlinks["artifacts/latest"] == "model"
    assert artifacts_ready(producer, "tsk-up")

    bundle = create_delivery_bundle(producer, "tsk-up", ["latest"])
    try:
        staging = extract_delivery_bundle(bundle, tmp_path / "stage", "tsk-up")
        destination = tmp_path / "consumer" / "tsk-up"
        commit_delivery(staging, destination, "tsk-up")
    finally:
        bundle.unlink()

    assert os.readlink(destination / "artifacts" / "latest") == "model"
    assert (destination / "artifacts" / "latest" / "weights").read_bytes() == b"model"
    assert artifacts_ready(destination, "tsk-up", ["latest"])
