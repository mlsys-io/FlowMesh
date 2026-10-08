from pathlib import Path
from unittest import mock

from server.dispatcher.snapshots import SnapshotCache
from shared.utils.result_delivery import read_receipt, validated_snapshot
from tests.shared.test_result_delivery import populate


def _names(base: Path) -> int:
    receipt = read_receipt(base)
    assert receipt is not None
    return len(receipt.directories) + len(receipt.files) + len(receipt.symlinks)


def test_least_recently_used_snapshots_are_evicted_by_name_count(
    tmp_path: Path,
) -> None:
    names = _names(populate(tmp_path, task_id="tsk-a"))
    populate(tmp_path, task_id="tsk-b")
    cache = SnapshotCache(tmp_path, name_limit=names + 1)

    with mock.patch(
        "server.dispatcher.snapshots.validated_snapshot", wraps=validated_snapshot
    ) as validated:
        assert cache.artifacts_ready("tsk-a", None)
        assert cache.artifacts_ready("tsk-b", None)
        assert cache.artifacts_ready("tsk-b", None)
        assert cache.artifacts_ready("tsk-a", None)

    assert [call.args[1] for call in validated.call_args_list] == [
        "tsk-a",
        "tsk-b",
        "tsk-a",
    ]


def test_a_snapshot_larger_than_the_limit_is_not_cached(tmp_path: Path) -> None:
    names = _names(populate(tmp_path, task_id="tsk-a"))
    cache = SnapshotCache(tmp_path, name_limit=names - 1)

    with mock.patch(
        "server.dispatcher.snapshots.validated_snapshot", wraps=validated_snapshot
    ) as validated:
        assert cache.artifacts_ready("tsk-a", None)
        assert cache.artifacts_ready("tsk-a", None)

    assert validated.call_count == 2


def test_generation_is_read_without_a_cached_snapshot(tmp_path: Path) -> None:
    base = populate(tmp_path, task_id="tsk-a")
    receipt = read_receipt(base)
    assert receipt is not None
    cache = SnapshotCache(tmp_path)

    assert cache.generation("tsk-a") == receipt.generation
    assert cache.artifacts_ready("tsk-a", ["model"])
    assert cache.generation("tsk-a") == receipt.generation
    assert cache.generation("tsk-missing") is None
