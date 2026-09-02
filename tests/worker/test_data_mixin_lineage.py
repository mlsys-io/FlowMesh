"""DataMixin tests: span emission + asset/lineage row JSONL writes."""

import io
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from shared.schemas.result import BaseExecutorResult
from worker.executors.mixins import data as data_mixin_module
from worker.executors.mixins.data import DataMixin
from worker.executors.utils import artifacts


class _Mixin(DataMixin):
    """Bare-bones DataMixin instance for unit testing."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _spans_for_task(out_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(out_dir / "logs" / "spans.jsonl")


def test_task_span_emits_root_with_compute_kind(tmp_path: Path) -> None:
    mixin = _Mixin()

    out_dir = tmp_path / "task"
    with mixin._task_span("tsk-1", "wfl-fbad6be5c4434181a2d394eac830dea1", out_dir):
        mixin._log_event("queuing for execution", data_id="tsk-1")

    spans = _spans_for_task(out_dir)
    names = [s["name"] for s in spans]
    assert "task" in names
    assert "queuing for execution" in names
    task_row = next(s for s in spans if s["name"] == "task")
    assert task_row["attributes"]["data_id"] == "tsk-1"
    assert task_row["attributes"]["flowmesh.type"] == "compute"
    assert {s["context"]["trace_id"] for s in spans} == {
        "0xfbad6be5c4434181a2d394eac830dea1"
    }


def test_record_asset_and_lineage(tmp_path: Path) -> None:
    mixin = _Mixin()
    out_dir = tmp_path / "task"
    with mixin._task_span("tsk-1", "wfl-1", out_dir):
        mixin._record_asset(
            data_id="tsk-1", asset_guid="g-1", version=1, user_id="alice"
        )
        mixin._record_lineage("tsk-1", ["upstream-a", "upstream-b"])

    base = out_dir / "logs"
    assets = _read_jsonl(base / "assets.jsonl")
    assert len(assets) == 1
    assert assets[0]["asset_guid"] == "g-1"
    assert assets[0]["user_id"] == "alice"

    lineage = _read_jsonl(base / "lineage.jsonl")
    assert len(lineage) == 2
    assert {row["source_data_id"] for row in lineage} == {
        "upstream-a",
        "upstream-b",
    }


def test_record_output_emits_dump_span_and_rows(tmp_path: Path) -> None:
    mixin = _Mixin()
    out_dir = tmp_path / "task-up"
    with mixin._task_span("tsk-up", "wfl-1", out_dir, owner_id="alice"):
        mixin._record_output(
            data_id="tsk-up",
            data={"items": [{"output": "ok"}]},
            source_data_ids=["tsk-source-a"],
        )

    base = out_dir / "logs"
    assets = _read_jsonl(base / "assets.jsonl")
    assert assets and assets[0]["data_id"] == "tsk-up"
    assert assets[0]["user_id"] == "alice"

    lineage = _read_jsonl(base / "lineage.jsonl")
    assert len(lineage) == 1
    assert lineage[0]["data_id"] == "tsk-up"
    assert lineage[0]["source_data_id"] == "tsk-source-a"

    spans = _spans_for_task(out_dir)
    dump = [s for s in spans if s["name"] == "dump to storage"]
    assert dump
    assert dump[0]["attributes"].get("data_id") == "tsk-up"
    assert dump[0]["attributes"]["flowmesh.type"] == "network"
    assert dump[0]["attributes"].get("payload_bytes", 0) > 0


def test_dump_to_governance_with_merged_children(tmp_path: Path) -> None:
    mixin = _Mixin()
    out_dir = tmp_path / "task"
    with mixin._task_span("tsk-parent", "wfl-1", out_dir, owner_id="alice"):
        result = BaseExecutorResult.model_validate(
            {
                "ok": True,
                "items": [{"output": "p"}],
                "children": {
                    "tsk-c1": {"items": [{"output": "c1"}]},
                    "tsk-c2": {"items": [{"output": "c2"}]},
                },
            }
        )
        deps = {
            "tsk-parent": ["tsk-up-a"],
            "tsk-c1": ["tsk-up-b"],
            "tsk-c2": ["tsk-up-c"],
        }
        mixin._dump_to_governance(
            task_id="tsk-parent",
            result=result,
            dependencies_by_task=deps,
        )

    base = out_dir / "logs"
    assets = _read_jsonl(base / "assets.jsonl")
    assert {row["data_id"] for row in assets} == {
        "tsk-parent",
        "tsk-c1",
        "tsk-c2",
    }
    assert all(row["user_id"] == "alice" for row in assets)

    lineage = _read_jsonl(base / "lineage.jsonl")
    edges = {(row["data_id"], row["source_data_id"]) for row in lineage}
    assert edges == {
        ("tsk-parent", "tsk-up-a"),
        ("tsk-c1", "tsk-up-b"),
        ("tsk-c2", "tsk-up-c"),
    }


def test_extract_source_data_ids_from_upstream_artifacts(tmp_path: Path) -> None:
    """Upstream task IDs are recovered from each result's _artifacts.base_dir
    (the producer's output dir), since the payload carries no ID itself."""
    mixin = _Mixin()
    upstream_a = BaseExecutorResult.model_validate(
        {"_artifacts": {"base_dir": (tmp_path / "results" / "tsk-up-a").as_posix()}}
    )
    upstream_b = BaseExecutorResult.model_validate(
        {"_artifacts": {"base_dir": (tmp_path / "results" / "tsk-up-b").as_posix()}}
    )
    no_ctx = BaseExecutorResult()
    spec = cast(
        Any,
        SimpleNamespace(
            upstreamResults={"a": upstream_a, "b": upstream_b, "skipped": no_ctx}
        ),
    )

    ids = mixin._extract_source_data_ids(spec)

    assert ids == ["tsk-up-a", "tsk-up-b"]


def test_collect_prompts_resolves_grouped_image_artifact_refs_after_flatten(
    tmp_path: Path,
) -> None:
    mixin = _Mixin()
    upstream_dir = tmp_path / "upstream-task"
    artifacts_dir = upstream_dir / "artifacts" / "images"
    artifacts_dir.mkdir(parents=True)
    for name, color in (("a.png", "red"), ("b.png", "green"), ("c.png", "blue")):
        Image.new("RGB", (2, 2), color=color).save(artifacts_dir / name)

    result = BaseExecutorResult.model_validate(
        {
            "images": [
                [{"path": "images/a.png"}, {"path": "images/b.png"}],
                [{"path": "images/c.png"}],
            ],
            "_artifacts": {"base_dir": upstream_dir.as_posix()},
        }
    )
    spec = cast(
        Any,
        SimpleNamespace(
            data={"type": "list", "expr": "vision.images"},
            inference={},
            upstreamResults={"vision": result},
        ),
    )

    entry = mixin._collect_prompts_for_spec(spec, "tsk-vision", fetch_images=True)

    assert entry.image_group_sizes == [2, 1]
    assert len(entry.images) == 3
    assert all(image is not None for image in entry.images)
    assert [image.size for image in entry.images if image is not None] == [
        (2, 2),
        (2, 2),
        (2, 2),
    ]


def _png_bytes(color: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_collect_prompts_flowmesh_results_url_sends_auth_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare http(s) image URL item that targets this worker's own
    FlowMesh server (FLOWMESH_BASE_URL) must go through the same
    auth-aware fetch path as artifact refs (resolve_artifact), not an
    unauthenticated `requests.get`. Regression test for a cross-worker
    results URL (`{base_url}/api/v1/results/{task_id}/files/{rel_path}`)
    being fetched without the `Authorization` header and failing image
    decoding."""
    monkeypatch.setenv("FLOWMESH_API_KEY", "s3cr3t-token")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "https://worker-b.internal")

    png_bytes = _png_bytes("purple")
    captured_calls: list[dict[str, Any]] = []

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
            yield png_bytes

    def _fake_get(
        url: str, headers: dict[str, str], stream: bool, timeout: float
    ) -> _FakeResponse:
        captured_calls.append({"url": url, "headers": headers})
        return _FakeResponse()

    monkeypatch.setattr(artifacts.requests, "get", _fake_get)

    mixin = _Mixin()
    spec = cast(
        Any,
        SimpleNamespace(
            data={
                "type": "list",
                "items": [
                    "https://worker-b.internal/api/v1/results/tsk-1/files/img.png"
                ],
            },
            inference={},
        ),
    )

    entry = mixin._collect_prompts_for_spec(spec, "tsk-http", fetch_images=True)

    assert len(captured_calls) == 1
    assert captured_calls[0]["headers"].get("Authorization") == "Bearer s3cr3t-token"
    assert len(entry.images) == 1
    assert entry.images[0] is not None
    assert entry.images[0].size == (2, 2)


def test_collect_prompts_external_url_sends_no_auth_and_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An http(s) image item that does NOT target this worker's configured
    FlowMesh server must never receive the worker's bearer token, even when
    FLOWMESH_API_KEY is set, and must use a short bounded timeout rather
    than resolve_artifact's long default (a hung/malicious external host
    must not stall the worker for ~30 minutes).

    `requests` is a single shared module object, so `data.requests` and
    `artifacts.requests` are the same object — patching one patches both.
    A single fake therefore both proves no credential leaked AND (via the
    call signature: no `headers`/`stream` kwargs, short `timeout`) that the
    call bypassed resolve_artifact's auth-aware download path entirely.
    """
    monkeypatch.setenv("FLOWMESH_API_KEY", "s3cr3t-token")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "https://worker-b.internal")

    png_bytes = _png_bytes("orange")
    captured_calls: list[dict[str, Any]] = []

    class _FakeResponse:
        content = png_bytes

        def raise_for_status(self) -> None:
            return None

    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        captured_calls.append({"url": url, "kwargs": kwargs})
        return _FakeResponse()

    monkeypatch.setattr(data_mixin_module.requests, "get", _fake_get)

    mixin = _Mixin()
    spec = cast(
        Any,
        SimpleNamespace(
            data={
                "type": "list",
                "items": ["http://attacker.example/x.png"],
            },
            inference={},
        ),
    )

    entry = mixin._collect_prompts_for_spec(spec, "tsk-http-ext", fetch_images=True)

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["url"] == "http://attacker.example/x.png"
    assert "headers" not in call["kwargs"]
    assert "stream" not in call["kwargs"]
    assert 0 < call["kwargs"].get("timeout", 0) <= 15
    assert len(entry.images) == 1
    assert entry.images[0] is not None
    assert entry.images[0].size == (2, 2)


def test_collect_prompts_external_dict_url_sends_no_auth_and_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLOWMESH_API_KEY", "s3cr3t-token")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "https://worker-b.internal")

    png_bytes = _png_bytes("orange")
    captured_calls: list[dict[str, Any]] = []

    class _FakeResponse:
        content = png_bytes

        def raise_for_status(self) -> None:
            return None

    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        captured_calls.append({"url": url, "kwargs": kwargs})
        return _FakeResponse()

    monkeypatch.setattr(data_mixin_module.requests, "get", _fake_get)

    mixin = _Mixin()
    spec = cast(
        Any,
        SimpleNamespace(
            data={
                "type": "list",
                "items": [{"url": "http://attacker.example/x.png"}],
            },
            inference={},
        ),
    )

    entry = mixin._collect_prompts_for_spec(spec, "tsk-http-ext", fetch_images=True)

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["url"] == "http://attacker.example/x.png"
    assert "headers" not in call["kwargs"]
    assert "stream" not in call["kwargs"]
    assert 0 < call["kwargs"].get("timeout", 0) <= 15
    assert len(entry.images) == 1
    assert entry.images[0] is not None
    assert entry.images[0].size == (2, 2)


def test_collect_prompts_dict_local_path_uses_artifact_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dict item whose 'url' is a bare local filesystem path (no scheme)
    must go through _load_image_from_artifact, not the external-http path.
    Regression test: urlparse("/local/x.png").scheme == "", which used to
    fall into the http(s)-only branch and call requests.get on a path,
    raising MissingSchema."""
    monkeypatch.delenv("FLOWMESH_API_KEY", raising=False)
    monkeypatch.delenv("FLOWMESH_BASE_URL", raising=False)

    img_path = tmp_path / "local.png"
    Image.new("RGB", (2, 2), color="red").save(img_path)

    def _fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("requests.get must not be called for a local path")

    monkeypatch.setattr(data_mixin_module.requests, "get", _fail_if_called)

    mixin = _Mixin()
    spec = cast(
        Any,
        SimpleNamespace(
            data={
                "type": "list",
                "items": [{"url": img_path.as_posix()}],
            },
            inference={},
        ),
    )

    entry = mixin._collect_prompts_for_spec(spec, "tsk-dict-local", fetch_images=True)

    assert len(entry.images) == 1
    assert entry.images[0] is not None
    assert entry.images[0].size == (2, 2)


def test_collect_prompts_dict_origin_url_sends_auth_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dict item whose 'url' targets this worker's own FlowMesh server
    must go through the auth-aware resolve_artifact path, same as the
    equivalent bare-string item."""
    monkeypatch.setenv("FLOWMESH_API_KEY", "s3cr3t-token")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "https://worker-b.internal")

    png_bytes = _png_bytes("purple")
    captured_calls: list[dict[str, Any]] = []

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
            yield png_bytes

    def _fake_get(
        url: str, headers: dict[str, str], stream: bool, timeout: float
    ) -> _FakeResponse:
        captured_calls.append({"url": url, "headers": headers})
        return _FakeResponse()

    monkeypatch.setattr(artifacts.requests, "get", _fake_get)

    mixin = _Mixin()
    spec = cast(
        Any,
        SimpleNamespace(
            data={
                "type": "list",
                "items": [
                    {
                        "url": (
                            "https://worker-b.internal/api/v1/results/tsk-1/files/img.png"
                        )
                    }
                ],
            },
            inference={},
        ),
    )

    entry = mixin._collect_prompts_for_spec(spec, "tsk-dict-origin", fetch_images=True)

    assert len(captured_calls) == 1
    assert captured_calls[0]["headers"].get("Authorization") == "Bearer s3cr3t-token"
    assert len(entry.images) == 1
    assert entry.images[0] is not None
    assert entry.images[0].size == (2, 2)
