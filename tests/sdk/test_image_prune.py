"""Tests for prune selection in ``flowmesh_stack.image_prune``."""

from datetime import UTC, datetime, timedelta

import pytest
from flowmesh_stack.docker import ManagedImage
from flowmesh_stack.image_prune import parse_duration, select_prune_targets

NOW = datetime(2026, 7, 10, tzinfo=UTC)


def _img(
    version: str | None,
    *,
    target: str | None = "flowmesh_server",
    age_days: int | None = 0,
    image_id: str | None = None,
    dangling: bool = False,
    in_use: bool = False,
) -> ManagedImage:
    tag = None if dangling else f"ghcr.io/mlsys-io/flowmesh_server:{version}"
    return ManagedImage(
        repo="ghcr.io/mlsys-io/flowmesh_server",
        tag=tag,
        target=None if dangling else target,
        version=None if dangling else version,
        image_id=image_id or f"sha256:{version}-{target}",
        size_bytes=1,
        created=None if age_days is None else NOW - timedelta(days=age_days),
        dangling=dangling,
        in_use=in_use,
    )


def _deleted_versions(images: list[ManagedImage], **kwargs: object) -> set[str | None]:
    plan = select_prune_targets(images, now=NOW, **kwargs)  # type: ignore[arg-type]
    return {i.version for i in plan.deleted}


def test_parse_duration_units() -> None:
    assert parse_duration("45s") == timedelta(seconds=45)
    assert parse_duration("90m") == timedelta(minutes=90)
    assert parse_duration("12h") == timedelta(hours=12)
    assert parse_duration("30d") == timedelta(days=30)
    assert parse_duration("2w") == timedelta(weeks=2)


@pytest.mark.parametrize("bad", ["", "10", "5x", "d", "-1d", "1.5d"])
def test_parse_duration_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(bad)


def test_keep_last_keeps_newest_versions() -> None:
    images = [
        _img("v3", age_days=1),
        _img("v2", age_days=2),
        _img("v1", age_days=3),
    ]
    assert _deleted_versions(images, keep_last=2) == {"v1"}


def test_older_than_only() -> None:
    images = [_img("new", age_days=5), _img("old", age_days=40)]
    assert _deleted_versions(images, older_than=timedelta(days=30)) == {"old"}


def test_keep_last_and_older_than_intersect() -> None:
    # Pool is older-than-30d; newest 3 protected. Only old-and-not-newest-3 go.
    images = [
        _img("v5", age_days=1),
        _img("v4", age_days=10),
        _img("v3", age_days=40),
        _img("v2", age_days=50),
        _img("v1", age_days=60),
    ]
    # older than 30d: v3, v2, v1. keep newest 3 overall: v5, v4, v3 -> delete v2, v1.
    assert _deleted_versions(images, keep_last=3, older_than=timedelta(days=30)) == {
        "v2",
        "v1",
    }


def test_keep_active_protects_in_use() -> None:
    images = [
        _img("v2", age_days=1),  # newest, protected by keep_last
        _img("v1", age_days=2, in_use=True),  # old but still running
    ]
    # keep_last=1 protects v2; keep_active additionally protects the running v1.
    assert _deleted_versions(images, keep_last=1, keep_active=True) == set()
    # Without keep_active the old running version would be deleted.
    assert _deleted_versions(images, keep_last=1) == {"v1"}


def test_dangling_only_deletes_dangling() -> None:
    images = [
        _img("v1", age_days=1),
        _img(None, dangling=True, image_id="sha256:dangle"),
    ]
    plan = select_prune_targets(images, include_dangling=True, now=NOW)
    assert [i.image_id for i in plan.deleted] == ["sha256:dangle"]


def test_keep_last_is_per_target() -> None:
    images = [
        _img("v2", target="flowmesh_worker_cpu", age_days=1),
        _img("v1", target="flowmesh_worker_cpu", age_days=2),
        _img("v2", target="flowmesh_worker_gpu", age_days=1),
        _img("v1", target="flowmesh_worker_gpu", age_days=2),
    ]
    plan = select_prune_targets(images, keep_last=1, now=NOW)
    deleted = {(i.target, i.version) for i in plan.deleted}
    assert deleted == {
        ("flowmesh_worker_cpu", "v1"),
        ("flowmesh_worker_gpu", "v1"),
    }


def test_keep_last_above_available_deletes_nothing() -> None:
    images = [_img("v2", age_days=1), _img("v1", age_days=2)]
    assert _deleted_versions(images, keep_last=5) == set()


def test_unparsed_tag_never_deleted() -> None:
    stray = _img("custom", target=None, age_days=99)
    stray.tag = "ghcr.io/mlsys-io/flowmesh_server:custom"  # tagged but unattributed
    stray.dangling = False
    images = [stray, _img("v1", age_days=99)]
    # older-than-30d and keep-last both leave the target=None tag untouched.
    assert _deleted_versions(images, older_than=timedelta(days=30)) == {"v1"}
    assert _deleted_versions(images, keep_last=0) == {"v1"}


def test_negative_keep_last_raises() -> None:
    with pytest.raises(ValueError):
        select_prune_targets([_img("v1")], keep_last=-1, now=NOW)


def test_older_than_skips_undeterminable_created() -> None:
    images = [_img("old", age_days=40), _img("unknown", age_days=None)]
    assert _deleted_versions(images, older_than=timedelta(days=30)) == {"old"}


def test_keep_last_protects_undeterminable_created() -> None:
    images = [
        _img("new", age_days=1),
        _img("old", age_days=5),
        _img("unknown", age_days=None),
    ]
    assert _deleted_versions(images, keep_last=1) == {"old"}


def test_keep_active_protects_dangling_by_id() -> None:
    dangle = _img(None, dangling=True, image_id="sha256:held", in_use=True)
    images = [dangle]
    plan = select_prune_targets(
        images, include_dangling=True, keep_active=True, now=NOW
    )
    assert plan.deleted == []
    assert plan.protected[0][1] == "keep-active"
