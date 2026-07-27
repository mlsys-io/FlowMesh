"""Tests for `worker.executors.utils.artifacts`.

Pure-logic helpers (no pandas/datasets/PIL) — runnable in CI without
the worker-side extras.
"""

from pathlib import Path
from typing import Any

import pytest

from shared.schemas.artifact import ArtifactContext
from shared.schemas.result import BaseExecutorResult
from worker.executors.base_executor import ExecutionError
from worker.executors.utils import artifacts as artifacts_module
from worker.executors.utils.artifacts import (
    artifact_to_source,
    is_flowmesh_origin_url,
    maybe_resolve_artifact_ref,
    resolve_artifact,
)


class TestArtifactToSource:
    def test_url_resolution(self, tmp_path: Path) -> None:
        upstream = BaseExecutorResult(
            _artifacts=ArtifactContext(
                base_dir=(tmp_path / "producer-tid").as_posix(),
                base_url="http://host:8010",
            ),
        )
        url = artifact_to_source({"path": "a.png"}, {"producer": upstream}, "producer")
        assert url == "http://host:8010/api/v1/results/producer-tid/files/a.png"

    def test_local_file_takes_fast_path(self, tmp_path: Path) -> None:
        task_root = tmp_path / "producer-tid"
        (task_root / "artifacts").mkdir(parents=True)
        (task_root / "artifacts" / "a.png").write_bytes(b"\x89PNG")
        upstream = BaseExecutorResult(
            _artifacts=ArtifactContext(
                base_dir=task_root.as_posix(),
                base_url="http://host:8010",
            )
        )
        resolved = artifact_to_source(
            {"path": "a.png"}, {"producer": upstream}, "producer"
        )
        assert resolved == (task_root / "artifacts" / "a.png").as_posix()

    def test_local_only_upstream_returns_local_path(self, tmp_path: Path) -> None:
        task_root = tmp_path / "producer-tid"
        upstream = BaseExecutorResult(
            _artifacts=ArtifactContext(base_dir=task_root.as_posix())
        )
        resolved = artifact_to_source(
            {"path": "a.png"}, {"producer": upstream}, "producer"
        )
        assert resolved == (task_root / "artifacts" / "a.png").as_posix()

    def test_missing_context_raises(self) -> None:
        with pytest.raises(ExecutionError, match="_artifacts context is missing"):
            artifact_to_source(
                {"path": "a.png"}, {"producer": BaseExecutorResult()}, "producer"
            )

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ExecutionError, match="non-empty 'path' field"):
            artifact_to_source({}, {"producer": BaseExecutorResult()}, "producer")


class TestMaybeResolveArtifactRef:
    def test_passes_through_non_dict(self) -> None:
        assert maybe_resolve_artifact_ref("hello", None, None) == "hello"
        assert maybe_resolve_artifact_ref(42, None, None) == 42
        assert maybe_resolve_artifact_ref([1, 2], None, None) == [1, 2]

    def test_passes_through_dict_without_path(self) -> None:
        value = {"url": "http://x"}
        assert maybe_resolve_artifact_ref(value, None, None) is value

    def test_resolves_path_dict(self, tmp_path: Path) -> None:
        upstream = BaseExecutorResult(
            _artifacts=ArtifactContext(
                base_url="http://host:8010",
                base_dir=(tmp_path / "producer-tid").as_posix(),
            )
        )
        out = maybe_resolve_artifact_ref(
            {"path": "a.png"}, {"producer": upstream}, "producer"
        )
        assert out == "http://host:8010/api/v1/results/producer-tid/files/a.png"


class TestResolveArtifact:
    def test_local_path_still_symlinks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail_if_called(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("requests.get must not be called for a local path")

        monkeypatch.setattr(artifacts_module.requests, "get", _fail_if_called)

        src = tmp_path / "a.png"
        src.write_bytes(b"\x89PNG")

        local_path = resolve_artifact(src.as_posix())

        assert local_path.is_symlink()
        assert local_path.resolve() == src.resolve()

    def test_external_url_sends_no_auth_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_API_KEY", "s3cr3t-token")
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://worker-b.internal")

        captured: dict[str, Any] = {}

        class _FakeResponse:
            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
                return [b"external-bytes"]

        def _fake_get(
            url: str, headers: dict[str, str] | None, stream: bool, timeout: float
        ) -> _FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse()

        monkeypatch.setattr(artifacts_module.requests, "get", _fake_get)

        resolve_artifact("https://attacker.example/x.png")

        assert captured["url"] == "https://attacker.example/x.png"
        assert captured["headers"] is None

    def test_origin_url_sends_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLOWMESH_API_KEY", "s3cr3t-token")
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://worker-b.internal")

        captured: dict[str, Any] = {}

        class _FakeResponse:
            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
                return [b"origin-bytes"]

        def _fake_get(
            url: str, headers: dict[str, str] | None, stream: bool, timeout: float
        ) -> _FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse()

        monkeypatch.setattr(artifacts_module.requests, "get", _fake_get)

        resolve_artifact("https://worker-b.internal/api/v1/results/tsk-1/files/a.png")

        assert captured["headers"] is not None
        assert captured["headers"].get("Authorization") == "Bearer s3cr3t-token"


class TestIsFlowmeshOriginUrl:
    def test_default_port_equivalence_target_implicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh:443")
        assert is_flowmesh_origin_url("https://flowmesh/api/v1/results") is True

    def test_default_port_equivalence_base_implicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh")
        assert is_flowmesh_origin_url("https://flowmesh:443/api/v1/results") is True

    def test_different_host_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh")
        assert is_flowmesh_origin_url("https://attacker.example/x") is False

    def test_different_explicit_non_default_port_is_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh:8443")
        assert is_flowmesh_origin_url("https://flowmesh:9443/x") is False

    def test_empty_base_url_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FLOWMESH_BASE_URL", raising=False)
        assert is_flowmesh_origin_url("https://flowmesh/x") is False
