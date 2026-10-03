"""Result bundle helper tests."""

import tarfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from server.routers.v1.results import (
    _BUNDLE_SECTIONS_DEFAULT,
    _create_result_bundle_archive,
    _resolve_bundle_sections,
)
from shared.schemas.result import BaseExecutorResult, ResultEnvelope


def _populate_task_dir(task_dir: Path) -> None:
    (task_dir / "artifacts" / "images").mkdir(parents=True)
    (task_dir / "logs").mkdir(parents=True)
    envelope = ResultEnvelope(task_id=task_dir.name, result=BaseExecutorResult())
    (task_dir / "results.json").write_text(envelope.model_dump_json(), encoding="utf-8")
    (task_dir / "manifest.json").write_text('{"ok": true}', encoding="utf-8")
    (task_dir / "artifacts" / "images" / "a.png").write_bytes(b"aaa")
    (task_dir / "logs" / "logs.jsonl").write_text("line\n", encoding="utf-8")


def _members(bundle_path: Path) -> set[str]:
    # tarfile with mode="r:*" transparently handles the gzip-wrapped tar.
    with tarfile.open(bundle_path, mode="r:*") as archive:
        return set(archive.getnames())


def test_default_bundle_contains_result_and_artifacts_not_logs(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task-1"
    _populate_task_dir(task_dir)

    bundle_path = _create_result_bundle_archive(
        "task-1", task_dir, sections=_BUNDLE_SECTIONS_DEFAULT
    )
    try:
        names = _members(bundle_path)
    finally:
        bundle_path.unlink(missing_ok=True)

    assert "task-1/results.json" in names
    assert "task-1/artifacts/images/a.png" in names
    assert "task-1/logs/logs.jsonl" not in names
    # manifest.json is never shipped.
    assert not any("manifest.json" in n for n in names)


def test_result_only_bundle_has_single_file(tmp_path: Path) -> None:
    task_dir = tmp_path / "task-1"
    _populate_task_dir(task_dir)

    bundle_path = _create_result_bundle_archive(
        "task-1", task_dir, sections=("results",)
    )
    try:
        names = _members(bundle_path)
    finally:
        bundle_path.unlink(missing_ok=True)

    assert names == {"task-1/results.json"}


def test_all_bundle_includes_every_concrete_section(tmp_path: Path) -> None:
    task_dir = tmp_path / "task-1"
    _populate_task_dir(task_dir)

    bundle_path = _create_result_bundle_archive(
        "task-1", task_dir, sections=("results", "artifacts", "logs")
    )
    try:
        names = _members(bundle_path)
    finally:
        bundle_path.unlink(missing_ok=True)

    assert "task-1/results.json" in names
    assert "task-1/artifacts/images/a.png" in names
    assert "task-1/logs/logs.jsonl" in names


def test_resolve_bundle_sections_empty_uses_default() -> None:
    assert _resolve_bundle_sections([]) == _BUNDLE_SECTIONS_DEFAULT


def test_resolve_bundle_sections_all_expands_to_concrete_set() -> None:
    assert _resolve_bundle_sections(["all"]) == ("results", "artifacts", "logs")


def test_resolve_bundle_sections_dedupes_and_orders() -> None:
    assert _resolve_bundle_sections(["artifacts", "logs", "results"]) == (
        "results",
        "artifacts",
        "logs",
    )


def test_resolve_bundle_sections_rejects_unknown() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _resolve_bundle_sections(["artifacts", "bogus"])
    assert exc_info.value.status_code == 400
