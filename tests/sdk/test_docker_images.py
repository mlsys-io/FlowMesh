"""Tests for image discovery helpers in ``flowmesh_stack.docker``."""

import json
from subprocess import CompletedProcess
from typing import Any

import pytest
from flowmesh_stack import docker as docker_module

REGISTRY = "ghcr.io/mlsys-io"


def _ok(stdout: str) -> CompletedProcess[str]:
    return CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _lines(*rows: dict[str, Any] | str) -> str:
    return "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows)


@pytest.fixture(autouse=True)
def _no_docker_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_module, "ensure_docker_available", lambda: None)


def test_list_managed_images_attributes_and_enriches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ls_out = _lines(
        {"Repository": f"{REGISTRY}/flowmesh_server", "Tag": "dev", "ID": "sha256:aaa"},
        {
            "Repository": f"{REGISTRY}/flowmesh_worker",
            "Tag": "dev-gpu",
            "ID": "sha256:bbb",
        },
        {"Repository": "ubuntu", "Tag": "22.04", "ID": "sha256:ccc"},
        {
            "Repository": f"{REGISTRY}/flowmesh_server",
            "Tag": "<none>",
            "ID": "sha256:ddd",
        },
        "{ this is not valid json",
    )
    inspect_out = _lines(
        {"Id": "sha256:aaa", "Size": 1000, "Created": "2026-07-01T10:00:00.123456789Z"},
        {"Id": "sha256:bbb", "Size": 2000, "Created": "2026-07-02T10:00:00Z"},
    )

    def fake_run(args: list[str], **_: Any) -> CompletedProcess[str]:
        if args[:3] == ["docker", "image", "ls"]:
            return _ok(ls_out)
        if args[:3] == ["docker", "image", "inspect"]:
            return _ok(inspect_out)
        raise AssertionError(f"unexpected: {args}")

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)
    images = docker_module.list_managed_images(REGISTRY, in_use_ids={"sha256:aaa"})

    by_id = {img.image_id: img for img in images}
    assert set(by_id) == {"sha256:aaa", "sha256:bbb"}  # ubuntu + <none> + junk dropped
    assert by_id["sha256:aaa"].target == "flowmesh_server"
    assert by_id["sha256:aaa"].version == "dev"
    assert by_id["sha256:aaa"].size_bytes == 1000
    assert by_id["sha256:aaa"].in_use is True
    assert by_id["sha256:bbb"].target == "flowmesh_worker_gpu"
    assert by_id["sha256:bbb"].in_use is False
    assert by_id["sha256:aaa"].created.year == 2026


def test_list_managed_images_includes_dangling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tagged = _lines(
        {"Repository": f"{REGISTRY}/flowmesh_server", "Tag": "dev", "ID": "sha256:aaa"}
    )
    dangling = _lines({"Repository": "<none>", "Tag": "<none>", "ID": "sha256:zzz"})
    inspect_out = _lines(
        {"Id": "sha256:aaa", "Size": 1, "Created": "2026-07-01T10:00:00Z"},
        {"Id": "sha256:zzz", "Size": 2, "Created": "2026-06-01T10:00:00Z"},
    )

    def fake_run(args: list[str], **_: Any) -> CompletedProcess[str]:
        if args[:3] == ["docker", "image", "ls"]:
            return _ok(dangling if "dangling=true" in args else tagged)
        if args[:3] == ["docker", "image", "inspect"]:
            return _ok(inspect_out)
        raise AssertionError(f"unexpected: {args}")

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)
    images = docker_module.list_managed_images(REGISTRY, include_dangling=True)

    dangle = next(i for i in images if i.dangling)
    assert dangle.image_id == "sha256:zzz"
    assert dangle.tag is None
    assert dangle.removal_ref == "sha256:zzz"
    assert dangle.size_bytes == 2


def test_container_image_refs_empty_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: Any) -> CompletedProcess[str]:
        calls.append(args)
        return _ok("")

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)
    assert docker_module.container_image_refs() == set()
    assert calls == [["docker", "ps", "-aq"]]  # no inspect when no containers


def test_container_image_refs_collects_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **_: Any) -> CompletedProcess[str]:
        if args == ["docker", "ps", "-aq"]:
            return _ok("c1\nc2\n")
        if args[:3] == ["docker", "container", "inspect"]:
            return _ok("sha256:aaa\nsha256:bbb\n")
        raise AssertionError(f"unexpected: {args}")

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)
    assert docker_module.container_image_refs() == {"sha256:aaa", "sha256:bbb"}


def test_remove_images_reports_each_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **_: Any) -> CompletedProcess[str]:
        ref = args[-1]
        if ref == "bad":
            return CompletedProcess(args=args, returncode=1, stdout="", stderr="in use")
        return CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)
    results = docker_module.remove_images(["good", "bad"], force=True)
    assert results[0].ok is True
    assert results[1].ok is False
    assert results[1].error == "in use"
