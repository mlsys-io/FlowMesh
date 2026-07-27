"""Tests for the worker artifact helpers in checkpoints."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from shared.schemas.artifact import ArtifactContext
from worker.executors.base_executor import TaskReference
from worker.executors.utils import checkpoints


def _task(
    dest_type: str = "http", dest_url: str = "http://host:8010/api/v1/results"
) -> TaskReference:
    if dest_type == "http":
        destination = SimpleNamespace(
            type="http",
            method="POST",
            url=dest_url,
            headers={},
            timeoutSec=30,
        )
    elif dest_type == "local":
        destination = SimpleNamespace(type="local")
    else:
        destination = None
    spec = SimpleNamespace(output=SimpleNamespace(destination=destination))
    return cast(TaskReference, SimpleNamespace(task_id="task-1", spec=spec))


class TestBuildArtifactContext:
    def test_http_destination_strips_api_suffix(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "task-1"
        out_dir.mkdir()
        ctx = checkpoints.build_artifact_context(_task().spec, out_dir)
        assert ctx == ArtifactContext(
            base_dir=out_dir.resolve().as_posix(), base_url="http://host:8010"
        )

    def test_local_destination_leaves_base_url_none(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "task-1"
        out_dir.mkdir()
        ctx = checkpoints.build_artifact_context(_task("local").spec, out_dir)
        assert ctx == ArtifactContext(
            base_dir=out_dir.resolve().as_posix(), base_url=None
        )


class TestMaybeUploadArtifacts:
    def test_uploads_with_filename_relative_to_artifacts_dir(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        out_dir = tmp_path / "task-1"
        artifacts_dir = out_dir / "artifacts"
        (artifacts_dir / "images" / "nested").mkdir(parents=True)
        first = artifacts_dir / "images" / "nested" / "a.png"
        second = artifacts_dir / "final_model.tar.gz"
        first.write_text("first", encoding="utf-8")
        second.write_text("second", encoding="utf-8")

        uploaded: list[tuple[str, str]] = []

        class _Response:
            def raise_for_status(self) -> None:
                return None

        def fake_request(method, url, files, headers, timeout):
            assert method == "POST"
            assert url == "http://host:8010/api/v1/results/task-1/files"
            remote_name, _fh, _content_type = files["file"]
            uploaded.append((remote_name, url))
            return _Response()

        monkeypatch.setattr(checkpoints.requests, "request", fake_request)

        rel_paths = checkpoints.maybe_upload_artifacts(_task(), out_dir)
        assert set(rel_paths) == {"images/nested/a.png", "final_model.tar.gz"}
        assert {name for name, _url in uploaded} == {
            "images/nested/a.png",
            "final_model.tar.gz",
        }

    def test_local_destination_is_noop(self, tmp_path: Path, monkeypatch) -> None:
        out_dir = tmp_path / "task-1"
        (out_dir / "artifacts").mkdir(parents=True)
        (out_dir / "artifacts" / "a.png").write_text("x", encoding="utf-8")

        def fake_request(*args, **kwargs):
            raise AssertionError("should not be called for local destination")

        monkeypatch.setattr(checkpoints.requests, "request", fake_request)
        assert checkpoints.maybe_upload_artifacts(_task("local"), out_dir) == []

    def test_empty_artifacts_dir_returns_empty_list(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        out_dir = tmp_path / "task-1"
        out_dir.mkdir()  # no artifacts/ subdir

        def fake_request(*args, **kwargs):
            raise AssertionError("should not be called")

        monkeypatch.setattr(checkpoints.requests, "request", fake_request)
        assert checkpoints.maybe_upload_artifacts(_task(), out_dir) == []


@pytest.mark.parametrize(
    ("url", "expects_auth"),
    [
        ("https://flowmesh.example/api/v1/results/tsk-1/files/model.bin", True),
        ("https://external.example/model.bin", False),
    ],
)
def test_download_and_unpack_only_authenticates_flowmesh_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    expects_auth: bool,
) -> None:
    monkeypatch.setenv("FLOWMESH_API_KEY", "flm-test")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh.example")
    captured_calls: list[dict[str, Any]] = []

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [b"checkpoint"]

    def _fake_get(
        request_url: str,
        *,
        headers: dict[str, str],
        stream: bool,
        timeout: float,
    ) -> _Response:
        captured_calls.append(
            {
                "url": request_url,
                "headers": headers,
                "stream": stream,
                "timeout": timeout,
            }
        )
        return _Response()

    monkeypatch.setattr(checkpoints.requests, "get", _fake_get)

    path = checkpoints.download_and_unpack({"url": url, "timeoutSec": 12}, tmp_path)

    assert path.read_bytes() == b"checkpoint"
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["timeout"] == 12
    assert call["stream"] is True
    if expects_auth:
        assert call["headers"]["Authorization"] == "Bearer flm-test"
    else:
        assert "Authorization" not in call["headers"]


class TestArchiveModelDir:
    def test_pigz_bin_empty_string_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that an empty MODEL_ARCHIVE_PIGZ_BIN env var gracefully falls back to
        'pigz' instead of passing an empty string to shutil.which()."""
        # Create a dummy model dir
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        # Mock shutil.which to verify what it receives and pretend pigz doesn't exist
        # to quickly exit the compression flow without actually compressing
        calls = []

        def mock_which(cmd, *args, **kwargs):
            calls.append(cmd)
            return None

        monkeypatch.setattr(checkpoints.shutil, "which", mock_which)

        # Explicitly set the env var to an empty string (or whitespace)
        monkeypatch.setenv("MODEL_ARCHIVE_PIGZ_BIN", "   ")
        monkeypatch.setenv("MODEL_ARCHIVE_TAR_BIN", "")

        # Also need to mock _should_use_pigz to True so it reaches the bin resolution
        monkeypatch.setattr(checkpoints, "_should_use_pigz", lambda: True)

        # The compression logic will fall back to tarfile if binaries are not found
        archive_path = checkpoints.archive_model_dir(model_dir)

        # Ensure shutil.which was called with the default 'pigz' and 'tar',
        # not empty strings
        assert "pigz" in calls
        assert "tar" not in calls  # Because pigz failed, it shouldn't even check tar

        assert archive_path.exists()
        assert archive_path.name == "model.tar.gz"

    def test_pigz_bin_custom_path_honored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that a custom valid string is passed to shutil.which."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        calls = []

        def mock_which(cmd, *args, **kwargs):
            calls.append(cmd)
            return None

        monkeypatch.setattr(checkpoints.shutil, "which", mock_which)
        monkeypatch.setenv("MODEL_ARCHIVE_PIGZ_BIN", "/usr/local/bin/mypigz")
        monkeypatch.setattr(checkpoints, "_should_use_pigz", lambda: True)

        checkpoints.archive_model_dir(model_dir)

        assert "/usr/local/bin/mypigz" in calls
