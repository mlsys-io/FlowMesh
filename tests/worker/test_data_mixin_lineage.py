"""Tests for the merged DataMixin: lineage JSONL writes + Redis transport."""

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


def test_log_event_appends_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)
    mixin._init_task_lineage("tsk-1", tmp_path / "task")

    mixin._log_event(event_type="queuing for execution")
    mixin._log_event(event_type="upstream fetch", event_data="from cache")

    events_path = tmp_path / "task" / "artifacts" / "logs" / "events.jsonl"
    rows = _read_jsonl(events_path)
    assert len(rows) == 2
    assert rows[0]["event_type"] == "queuing for execution"
    assert rows[0]["data_id"] == "tsk-1"
    assert rows[1]["event_data"] == "from cache"


def test_record_asset_and_lineage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)
    mixin._init_task_lineage("tsk-1", tmp_path / "task")

    mixin._record_asset(data_id="tsk-1", asset_guid="g-1", version=1, user_id="alice")
    mixin._record_lineage("tsk-1", ["upstream-a", "upstream-b"])

    base = tmp_path / "task" / "artifacts" / "logs"
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
    writer._init_task_lineage("tsk-up", tmp_path / "task-up")
    writer._write_data(
        data_id="tsk-up",
        data={"items": [{"output": "hello"}], "ok": True},
        source_data_ids=[],
        governance_spec={"user_id": "alice"},
    )

    reader = _Mixin(fake)
    reader._init_task_lineage("tsk-down", tmp_path / "task-down")
    fetched = reader._fetch_data("tsk-up")

    assert fetched == {"items": [{"output": "hello"}], "ok": True}


def test_cache_hit_avoids_redis(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)

    writer = _Mixin(fake)
    writer._init_task_lineage("tsk-up", tmp_path / "task-up")
    writer._write_data(
        data_id="tsk-up",
        data={"items": [{"output": "hello"}]},
        source_data_ids=[],
        governance_spec={"user_id": "alice"},
    )

    # Pre-populate the cache for a *different* worker by writing the file
    # directly: simulates "second task on same worker reads same upstream."
    reader = _Mixin(fake)
    reader._init_task_lineage("tsk-down", tmp_path / "task-down")
    payload = {"items": [{"output": "hello"}]}
    reader._write_cache("tsk-up", payload)

    # Wipe Redis so a real fetch would fail.
    fake.flushall()

    fetched = reader._fetch_data("tsk-up")
    assert fetched == payload
    events = _read_jsonl(tmp_path / "task-down" / "artifacts" / "logs" / "events.jsonl")
    assert any(
        row["event_type"] == "read cache hit" and row["data_id"] == "tsk-up"
        for row in events
    )


def test_dump_to_governance_with_merged_children(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)
    mixin._init_task_lineage("tsk-parent", tmp_path / "task")

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

    # All three data_ids land in Redis.
    assert fake.exists("flowmesh:data:tsk-parent")
    assert fake.exists("flowmesh:data:tsk-c1")
    assert fake.exists("flowmesh:data:tsk-c2")

    base = tmp_path / "task" / "artifacts" / "logs"
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
    mixin._init_task_lineage("tsk-down", tmp_path / "task-down")

    server_payload = {"items": [{"output": "from-server"}], "ok": True}
    monkeypatch.setattr(
        mixin,
        "_fetch_from_server",
        lambda data_id: server_payload if data_id == "tsk-missing-in-redis" else None,
    )

    fetched = mixin._fetch_data("tsk-missing-in-redis")
    assert fetched == server_payload

    events = _read_jsonl(tmp_path / "task-down" / "artifacts" / "logs" / "events.jsonl")
    assert any(
        row["event_type"] == "read response transfer"
        and "source=server" in str(row.get("event_data", ""))
        for row in events
    )


def test_fetch_data_missing_in_redis_and_server_raises(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("WORKER_CACHE_DIR", str(tmp_path / "wc"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    mixin = _Mixin(fake)
    mixin._init_task_lineage("tsk-down", tmp_path / "task-down")
    monkeypatch.setattr(mixin, "_fetch_from_server", lambda data_id: None)

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
    mixin._init_task_lineage("tsk-up", tmp_path / "task-up")

    # Should not raise — write proceeds even though Redis publish failed.
    mixin._write_data(
        data_id="tsk-up",
        data={"items": [{"output": "ok"}]},
        source_data_ids=[],
        governance_spec={"user_id": "alice"},
    )
    assets = _read_jsonl(tmp_path / "task-up" / "artifacts" / "logs" / "assets.jsonl")
    assert assets and assets[0]["data_id"] == "tsk-up"
