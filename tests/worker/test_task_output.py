"""Tests for worker output manifest generation and artifact tracking."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from worker.runner import Runner
from worker.utils.manifest import (
    ARTIFACTS_DIR,
    LOGS_DIR,
    MANIFEST_NAME,
    RESULTS_NAME,
    prepare_output_dir,
    sync_manifest,
)


class TestPrepareOutputDir:
    def test_creates_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        prepare_output_dir(out)
        assert out.exists()
        assert (out / LOGS_DIR).exists()
        assert (out / ARTIFACTS_DIR).exists()

    def test_idempotent(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        prepare_output_dir(out)
        prepare_output_dir(out)  # should not raise
        assert out.exists()


class TestSyncManifest:
    def test_basic_manifest(self, tmp_path: Path) -> None:
        """Sync with no declared artifacts produces default entries."""
        manifest = sync_manifest(tmp_path, "t-1", expected=[])
        assert manifest["task_id"] == "t-1"
        names = {e["name"] for e in manifest["entries"]}
        assert RESULTS_NAME in names
        assert LOGS_DIR in names
        assert ARTIFACTS_DIR in names

    def test_present_file(self, tmp_path: Path) -> None:
        """A declared artifact that exists is marked 'present' with sha256."""
        prepare_output_dir(tmp_path)
        (tmp_path / "results.json").write_text('{"ok": true}')
        manifest = sync_manifest(tmp_path, "t-1", expected=[])
        resp_entry = next(e for e in manifest["entries"] if e["name"] == RESULTS_NAME)
        assert resp_entry["status"] == "present"
        assert resp_entry["type"] == "result"
        assert "sha256" in resp_entry
        assert resp_entry["size"] > 0

    def test_missing_artifact(self, tmp_path: Path) -> None:
        """A declared artifact that doesn't exist is marked 'missing'."""
        manifest = sync_manifest(tmp_path, "t-1", expected=["model.bin"])
        model_entry = next(e for e in manifest["entries"] if e["name"] == "model.bin")
        assert model_entry["status"] == "missing"

    def test_extra_files_captured(self, tmp_path: Path) -> None:
        """Files not in the expected list but present on disk are included."""
        prepare_output_dir(tmp_path)
        (tmp_path / "extra.txt").write_text("hello")
        manifest = sync_manifest(tmp_path, "t-1", expected=[])
        names = {e["name"] for e in manifest["entries"]}
        assert "extra.txt" in names

    def test_manifest_written_to_disk(self, tmp_path: Path) -> None:
        sync_manifest(tmp_path, "t-1", expected=[])
        manifest_path = tmp_path / MANIFEST_NAME
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["task_id"] == "t-1"

    def test_expected_names_are_normalized_in_manifest(self, tmp_path: Path) -> None:
        manifest = sync_manifest(tmp_path, "t-1", expected=["  file.txt  ", "./dir/"])
        names = {e["name"] for e in manifest["entries"]}
        assert "file.txt" in names
        assert "dir" in names

    def test_manifest_entry_types_reflect_real_paths(self, tmp_path: Path) -> None:
        prepare_output_dir(tmp_path)
        (tmp_path / LOGS_DIR / "task.jsonl").write_text("line\n")
        (tmp_path / ARTIFACTS_DIR / "output.png").write_text("png")
        (tmp_path / "subdir").mkdir()

        manifest = sync_manifest(tmp_path, "t-1", expected=["subdir"])
        entries = {entry["name"]: entry for entry in manifest["entries"]}

        assert entries[LOGS_DIR]["type"] == "logs"
        assert entries[ARTIFACTS_DIR]["type"] == "artifact"
        assert entries[ARTIFACTS_DIR]["status"] == "present"
        assert entries[ARTIFACTS_DIR]["file_count"] == 1
        assert entries["subdir"]["type"] == "directory"


class TestRunnerOutputDir:
    def test_resolve_output_dir_uses_canonical_path(self, tmp_path: Path) -> None:
        """Test that _resolve_output_dir creates and returns the canonical path
        `<results_dir>/<task_id>`."""
        results_dir = tmp_path / "results"
        runner = Runner(
            lifecycle=MagicMock(),
            task_stream=[],
            results_dir=results_dir,
            hardware=MagicMock(),
            executors={},
            default_executor=MagicMock(),
            logger=MagicMock(),
        )

        task_id = "task-123"
        out_dir = runner._resolve_output_dir(task_id)

        assert out_dir == results_dir / task_id
        assert out_dir.exists()
        assert out_dir.is_dir()
