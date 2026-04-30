"""Tests for the DataMixin: span emission + asset/lineage rows + Redis transport."""

import json
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from worker.executors.mixins.data import DataMixin


class _Mixin(DataMixin):
    """Bare-bones DataMixin instance for unit testing."""

    def __init__(self, fake: fakeredis.FakeRedis) -> None:
        super().__init__()
        self._redis_client = fake


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _spans_for_task(out_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(out_dir / "artifacts" / "logs" / "spans.jsonl")


def _names_with_attr(spans: list[dict[str, Any]], **wanted: Any) -> list[str]:
    """Return span names whose attributes match every key in `wanted`."""
    out: list[str] = []
    for s in spans:
        attrs = s.get("attributes") or {}
        if all(attrs.get(k) == v for k, v in wanted.items()):
            out.append(s["name"])
    return out


def test_task_span_emits_root_with_compute_kind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)

    out_dir = tmp_path / "task"
    with mixin._task_span("tsk-1", "wfl-fbad6be5c4434181a2d394eac830dea1", out_dir):
        mixin._instant("queuing for execution", data_id="tsk-1")

    spans = _spans_for_task(out_dir)
    names = [s["name"] for s in spans]
    assert "task" in names
    assert "queuing for execution" in names
    task_row = next(s for s in spans if s["name"] == "task")
    assert task_row["attributes"]["data_id"] == "tsk-1"
    assert task_row["attributes"]["flowmesh.kind"] == "compute"
    # All spans share the same trace_id pinned from the workflow_id.
    assert {s["context"]["trace_id"] for s in spans} == {
        "0xfbad6be5c4434181a2d394eac830dea1"
    }


def test_record_asset_and_lineage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)
    out_dir = tmp_path / "task"
    with mixin._task_span("tsk-1", "wfl-1", out_dir):
        mixin._record_asset(
            data_id="tsk-1", asset_guid="g-1", version=1, user_id="alice"
        )
        mixin._record_lineage("tsk-1", ["upstream-a", "upstream-b"])

    base = out_dir / "artifacts" / "logs"
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


def test_redis_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)

    writer = _Mixin(fake)
    with writer._task_span("tsk-up", "wfl-1", tmp_path / "task-up"):
        writer._write_data(
            data_id="tsk-up",
            data={"items": [{"output": "hello"}], "ok": True},
            source_data_ids=[],
            governance_spec={"user_id": "alice"},
        )

    reader = _Mixin(fake)
    with reader._task_span("tsk-down", "wfl-1", tmp_path / "task-down"):
        fetched = reader._fetch_data("tsk-up")

    assert fetched == {"items": [{"output": "hello"}], "ok": True}


def test_cache_hit_emits_marker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)

    writer = _Mixin(fake)
    with writer._task_span("tsk-up", "wfl-1", tmp_path / "task-up"):
        writer._write_data(
            data_id="tsk-up",
            data={"items": [{"output": "hello"}]},
            source_data_ids=[],
            governance_spec={"user_id": "alice"},
        )

    reader = _Mixin(fake)
    out_dir = tmp_path / "task-down"
    with reader._task_span("tsk-down", "wfl-1", out_dir):
        payload = {"items": [{"output": "hello"}]}
        reader._write_cache("tsk-up", payload)
        fake.flushall()
        fetched = reader._fetch_data("tsk-up")
    assert fetched == payload

    spans = _spans_for_task(out_dir)
    cache_hit_spans = [
        s
        for s in spans
        if s["name"] == "cache hit" and s["attributes"].get("data_id") == "tsk-up"
    ]
    assert cache_hit_spans, "expected a 'cache hit' marker span for tsk-up"
    assert cache_hit_spans[0]["attributes"]["flowmesh.kind"] == "marker"


def test_dump_to_governance_with_merged_children(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)
    out_dir = tmp_path / "task"
    with mixin._task_span("tsk-parent", "wfl-1", out_dir):
        result = {
            "ok": True,
            "items": [{"output": "p"}],
            "children": {
                "tsk-c1": {"items": [{"output": "c1"}]},
                "tsk-c2": {"items": [{"output": "c2"}]},
            },
        }
        deps = {
            "tsk-parent": ["tsk-up-a"],
            "tsk-c1": ["tsk-up-b"],
            "tsk-c2": ["tsk-up-c"],
        }
        mixin._dump_to_governance(
            governance_spec={"user_id": "alice"},
            task_id="tsk-parent",
            result=result,
            dependencies_by_task=deps,
        )

    assert fake.exists("flowmesh:data:tsk-parent")
    assert fake.exists("flowmesh:data:tsk-c1")
    assert fake.exists("flowmesh:data:tsk-c2")

    base = out_dir / "artifacts" / "logs"
    assets = _read_jsonl(base / "assets.jsonl")
    assert {row["data_id"] for row in assets} == {
        "tsk-parent",
        "tsk-c1",
        "tsk-c2",
    }

    lineage = _read_jsonl(base / "lineage.jsonl")
    edges = {(row["data_id"], row["source_data_id"]) for row in lineage}
    assert edges == {
        ("tsk-parent", "tsk-up-a"),
        ("tsk-c1", "tsk-up-b"),
        ("tsk-c2", "tsk-up-c"),
    }


def test_fetch_data_falls_back_to_server_on_redis_miss(
    tmp_path: Path, monkeypatch
) -> None:
    """Redis is a cache; the durable copy lives on the server."""
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://server.test")
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)

    server_payload = {"items": [{"output": "from-server"}], "ok": True}
    monkeypatch.setattr(
        mixin,
        "_fetch_from_server",
        lambda data_id: server_payload if data_id == "tsk-missing-in-redis" else None,
    )

    out_dir = tmp_path / "task-down"
    with mixin._task_span("tsk-down", "wfl-1", out_dir):
        fetched = mixin._fetch_data("tsk-missing-in-redis")
    assert fetched == server_payload

    spans = _spans_for_task(out_dir)
    read_spans = [
        s
        for s in spans
        if s["name"] == "read"
        and s["attributes"].get("data_id") == "tsk-missing-in-redis"
    ]
    assert read_spans, "expected a 'read' span for the fetched data_id"
    assert read_spans[0]["attributes"].get("source") == "server"
    assert read_spans[0]["attributes"]["flowmesh.kind"] == "network"


def test_fetch_data_missing_in_redis_and_server_raises(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)
    monkeypatch.setattr(mixin, "_fetch_from_server", lambda data_id: None)

    with mixin._task_span("tsk-down", "wfl-1", tmp_path / "task-down"):
        with pytest.raises(Exception) as excinfo:
            mixin._fetch_data("tsk-missing")
    assert "tsk-missing" in str(excinfo.value)


def test_write_data_tolerates_redis_failure(tmp_path: Path, monkeypatch) -> None:
    """Server upload is the source of truth; Redis is best-effort."""
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))

    class _BrokenRedis:
        def set(self, *_a, **_kw):  # noqa: ANN001
            import redis as _redis

            raise _redis.RedisError("connection lost")

    mixin = _Mixin(_BrokenRedis())  # type: ignore[arg-type]
    out_dir = tmp_path / "task-up"
    with mixin._task_span("tsk-up", "wfl-1", out_dir):
        mixin._write_data(
            data_id="tsk-up",
            data={"items": [{"output": "ok"}]},
            source_data_ids=[],
            governance_spec={"user_id": "alice"},
        )
    assets = _read_jsonl(out_dir / "artifacts" / "logs" / "assets.jsonl")
    assert assets and assets[0]["data_id"] == "tsk-up"
