"""Tests for the ``flowmesh stack image`` command group."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import typer
from flowmesh_cli_stack import image as image_module
from flowmesh_stack.docker import ManagedImage, RemovalResult

REGISTRY = "ghcr.io/mlsys-io"


def _img(
    target: str | None,
    version: str | None,
    *,
    dangling: bool = False,
    in_use: bool = False,
) -> ManagedImage:
    repo = f"{REGISTRY}/flowmesh_server"
    tag = None if dangling else f"{repo}:{version}"
    return ManagedImage(
        repo=repo if not dangling else "<none>",
        tag=tag,
        target=None if dangling else target,
        version=None if dangling else version,
        image_id=f"sha256:{version}",
        size_bytes=10,
        created=datetime(2026, 7, 1, tzinfo=UTC),
        dangling=dangling,
        in_use=in_use,
    )


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_module, "ensure_docker_available", lambda: None)
    monkeypatch.setattr(image_module, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(image_module, "container_image_refs", lambda: set())
    monkeypatch.setenv("FLOWMESH_REGISTRY", REGISTRY)


def test_list_json_shape(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(
        image_module,
        "list_managed_images",
        lambda *a, **k: [_img("flowmesh_server", "dev")],
    )
    image_module.list_images(
        targets=None,
        version=None,
        in_use=False,
        dangling=False,
        include_builder=False,
        as_json=True,
        env_file=Path(".env"),
    )
    out = capsys.readouterr().out
    assert '"target": "flowmesh_server"' in out
    assert '"version": "dev"' in out


def test_prune_requires_policy() -> None:
    with pytest.raises(typer.Exit):
        image_module.prune(
            keep_last=None,
            keep_active=False,
            keep=None,
            older_than=None,
            dangling=False,
            targets=None,
            include_builder=False,
            dry_run=False,
            yes=False,
            as_json=False,
            env_file=Path(".env"),
        )


def test_prune_dangling_with_target_refuses() -> None:
    with pytest.raises(typer.Exit):
        image_module.prune(
            keep_last=None,
            keep_active=False,
            keep=None,
            older_than=None,
            dangling=True,
            targets=["flowmesh_server"],
            include_builder=False,
            dry_run=False,
            yes=True,
            as_json=False,
            env_file=Path(".env"),
        )


def test_prune_dry_run_removes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_module,
        "list_managed_images",
        lambda *a, **k: [
            _img("flowmesh_server", "v2"),
            _img("flowmesh_server", "v1"),
        ],
    )
    called = {"removed": False}

    def fake_remove(*_a: Any, **_k: Any) -> list[RemovalResult]:
        called["removed"] = True
        return []

    monkeypatch.setattr(image_module, "remove_images", fake_remove)
    image_module.prune(
        keep_last=1,
        keep_active=False,
        keep=None,
        older_than=None,
        dangling=False,
        targets=None,
        include_builder=False,
        dry_run=True,
        yes=False,
        as_json=False,
        env_file=Path(".env"),
    )
    assert called["removed"] is False


def test_prune_yes_skips_confirm_and_removes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_module,
        "list_managed_images",
        lambda *a, **k: [
            _img("flowmesh_server", "v2"),
            _img("flowmesh_server", "v1"),
        ],
    )
    removed_refs: list[str] = []

    def fake_remove(refs: list[str], **_: Any) -> list[RemovalResult]:
        removed_refs.extend(refs)
        return [RemovalResult(ref=r, ok=True) for r in refs]

    monkeypatch.setattr(image_module, "remove_images", fake_remove)
    monkeypatch.setattr(
        image_module.typer, "confirm", lambda *a, **k: pytest.fail("prompted")
    )
    image_module.prune(
        keep_last=1,
        keep_active=False,
        keep=None,
        older_than=None,
        dangling=False,
        targets=None,
        include_builder=False,
        dry_run=False,
        yes=True,
        as_json=False,
        env_file=Path(".env"),
    )
    assert removed_refs == [f"{REGISTRY}/flowmesh_server:v1"]


def test_prune_exit_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_module,
        "list_managed_images",
        lambda *a, **k: [_img("flowmesh_server", "v1")],
    )
    monkeypatch.setattr(
        image_module,
        "remove_images",
        lambda refs, **k: [RemovalResult(ref=refs[0], ok=False, error="in use")],
    )
    with pytest.raises(typer.Exit):
        image_module.prune(
            keep_last=0,
            keep_active=False,
            keep=None,
            older_than=None,
            dangling=False,
            targets=None,
            include_builder=False,
            dry_run=False,
            yes=True,
            as_json=False,
            env_file=Path(".env"),
        )


def test_prune_json_without_yes_refuses() -> None:
    with pytest.raises(typer.Exit):
        image_module.prune(
            keep_last=1,
            keep_active=False,
            keep=None,
            older_than=None,
            dangling=False,
            targets=None,
            include_builder=False,
            dry_run=False,
            yes=False,
            as_json=True,
            env_file=Path(".env"),
        )


def test_rm_json_without_yes_refuses() -> None:
    with pytest.raises(typer.Exit):
        image_module.rm(
            versions=["v1"],
            targets=None,
            include_builder=False,
            force=False,
            dry_run=False,
            yes=False,
            as_json=True,
            env_file=Path(".env"),
        )


def test_rm_resolves_present_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_module,
        "list_managed_images",
        lambda *a, **k: [
            _img("flowmesh_server", "v1"),
        ],
    )
    removed_refs: list[str] = []

    def fake_remove(refs: list[str], **_: Any) -> list[RemovalResult]:
        removed_refs.extend(refs)
        return [RemovalResult(ref=r, ok=True) for r in refs]

    monkeypatch.setattr(image_module, "remove_images", fake_remove)
    image_module.rm(
        versions=["v1"],
        targets=["flowmesh_server"],
        include_builder=False,
        force=False,
        dry_run=False,
        yes=True,
        as_json=False,
        env_file=Path(".env"),
    )
    assert removed_refs == [f"{REGISTRY}/flowmesh_server:v1"]


def test_rm_unknown_target_exits() -> None:
    with pytest.raises(typer.Exit):
        image_module.rm(
            versions=["v1"],
            targets=["bogus"],
            include_builder=False,
            force=False,
            dry_run=False,
            yes=True,
            as_json=False,
            env_file=Path(".env"),
        )


def test_resolve_targets_builder_toggle() -> None:
    assert "flowmesh_worker_gpu_builder" not in image_module._resolve_targets(
        None, False
    )
    assert "flowmesh_worker_gpu_builder" in image_module._resolve_targets(None, True)
