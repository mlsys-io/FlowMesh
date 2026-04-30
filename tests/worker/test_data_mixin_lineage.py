"""DataMixin tests: span emission, asset/lineage rows, supervisor cache."""

import json
from pathlib import Path
from typing import Any

import pytest

from worker.executors.mixins.data import DataMixin


class _FakeSupervisorData:
    """In-memory stand-in for SupervisorDataClient.

    Mirrors the supervisor's FetchData / PublishData behavior over a dict.
    Lets tests exercise cache hits / misses without spinning up gRPC.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.fail_publish = False

    def fetch(self, data_id: str) -> bytes | None:
        return self.store.get(data_id)

    def publish(self, data_id: str, payload: bytes, ttl_sec: int) -> bool:
        if self.fail_publish:
            return False
        self.store[data_id] = payload
        return True

    def close(self) -> None:
        return None


class _Mixin(DataMixin):
    """Bare-bones DataMixin instance for unit testing."""

    def __init__(self, supervisor: _FakeSupervisorData) -> None:
        super().__init__()
        self._supervisor_data = supervisor  # type: ignore[assignment]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _spans_for_task(out_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(out_dir / "artifacts" / "logs" / "spans.jsonl")


def test_task_span_emits_root_with_compute_kind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    mixin = _Mixin(_FakeSupervisorData())

    out_dir = tmp_path / "task"
    with mixin._task_span("tsk-1", "wfl-fbad6be5c4434181a2d394eac830dea1", out_dir):
        mixin._log_event("queuing for execution", data_id="tsk-1")

    spans = _spans_for_task(out_dir)
    names = [s["name"] for s in spans]
    assert "task" in names
    assert "queuing for execution" in names
    task_row = next(s for s in spans if s["name"] == "task")
    assert task_row["attributes"]["data_id"] == "tsk-1"
    assert task_row["attributes"]["flowmesh.kind"] == "compute"
    assert {s["context"]["trace_id"] for s in spans} == {
        "0xfbad6be5c4434181a2d394eac830dea1"
    }


def test_record_asset_and_lineage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    mixin = _Mixin(_FakeSupervisorData())
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


def test_supervisor_cache_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    supervisor = _FakeSupervisorData()

    writer = _Mixin(supervisor)
    with writer._task_span("tsk-up", "wfl-1", tmp_path / "task-up"):
        writer._write_data(
            data_id="tsk-up",
            data={"items": [{"output": "hello"}], "ok": True},
            source_data_ids=[],
            governance_spec={"user_id": "alice"},
        )

    reader = _Mixin(supervisor)
    with reader._task_span("tsk-down", "wfl-1", tmp_path / "task-down"):
        fetched = reader._fetch_data("tsk-up")

    assert fetched == {"items": [{"output": "hello"}], "ok": True}
    assert "tsk-up" in supervisor.store


def test_cache_hit_recorded_on_read_span(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    supervisor = _FakeSupervisorData()

    writer = _Mixin(supervisor)
    with writer._task_span("tsk-up", "wfl-1", tmp_path / "task-up"):
        writer._write_data(
            data_id="tsk-up",
            data={"items": [{"output": "hello"}]},
            source_data_ids=[],
            governance_spec={"user_id": "alice"},
        )

    reader = _Mixin(supervisor)
    out_dir = tmp_path / "task-down"
    with reader._task_span("tsk-down", "wfl-1", out_dir):
        payload = {"items": [{"output": "hello"}]}
        reader._write_cache("tsk-up", payload)
        supervisor.store.clear()
        fetched = reader._fetch_data("tsk-up")
    assert fetched == payload

    spans = _spans_for_task(out_dir)
    read_spans = [
        s
        for s in spans
        if s["name"] == "read" and s["attributes"].get("data_id") == "tsk-up"
    ]
    assert read_spans, "expected a 'read' span for tsk-up"
    assert read_spans[0]["attributes"].get("source") == "cache"
    assert read_spans[0]["attributes"].get("cache_hit") is True
    assert read_spans[0]["attributes"]["flowmesh.kind"] == "network"


def test_dump_to_governance_with_merged_children(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    supervisor = _FakeSupervisorData()
    mixin = _Mixin(supervisor)
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

    assert {"tsk-parent", "tsk-c1", "tsk-c2"} <= supervisor.store.keys()

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


def test_fetch_data_falls_back_to_server_on_supervisor_miss(
    tmp_path: Path, monkeypatch
) -> None:
    """The supervisor cache is best-effort; the server has the durable copy."""
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://server.test")
    mixin = _Mixin(_FakeSupervisorData())

    server_payload = {"items": [{"output": "from-server"}], "ok": True}
    monkeypatch.setattr(
        mixin,
        "_fetch_from_server",
        lambda data_id: server_payload if data_id == "tsk-missing-in-cache" else None,
    )

    out_dir = tmp_path / "task-down"
    with mixin._task_span("tsk-down", "wfl-1", out_dir):
        fetched = mixin._fetch_data("tsk-missing-in-cache")
    assert fetched == server_payload

    spans = _spans_for_task(out_dir)
    read_spans = [
        s
        for s in spans
        if s["name"] == "read"
        and s["attributes"].get("data_id") == "tsk-missing-in-cache"
    ]
    assert read_spans, "expected a 'read' span for the fetched data_id"
    assert read_spans[0]["attributes"].get("source") == "server"
    assert read_spans[0]["attributes"]["flowmesh.kind"] == "network"


def test_fetch_data_missing_in_cache_and_server_raises(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    mixin = _Mixin(_FakeSupervisorData())
    monkeypatch.setattr(mixin, "_fetch_from_server", lambda data_id: None)

    with mixin._task_span("tsk-down", "wfl-1", tmp_path / "task-down"):
        with pytest.raises(Exception) as excinfo:
            mixin._fetch_data("tsk-missing")
    assert "tsk-missing" in str(excinfo.value)


def test_write_data_tolerates_supervisor_publish_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """Server upload is the source of truth; supervisor publish is best-effort."""
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))

    supervisor = _FakeSupervisorData()
    supervisor.fail_publish = True
    mixin = _Mixin(supervisor)
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
    spans = _spans_for_task(out_dir)
    dump_spans = [s for s in spans if s["name"] == "dump to storage"]
    assert dump_spans
    assert dump_spans[0]["attributes"].get("cache_hit") is False
