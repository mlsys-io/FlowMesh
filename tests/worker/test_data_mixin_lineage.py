"""DataMixin tests: span emission + asset/lineage row JSONL writes."""

import json
from pathlib import Path
from typing import Any

from worker.executors.mixins.data import DataMixin


class _Mixin(DataMixin):
    """Bare-bones DataMixin instance for unit testing."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _spans_for_task(out_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(out_dir / "artifacts" / "logs" / "spans.jsonl")


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
    assert task_row["attributes"]["flowmesh.kind"] == "compute"
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


def test_write_data_emits_dump_span_and_rows(tmp_path: Path) -> None:
    mixin = _Mixin()
    out_dir = tmp_path / "task-up"
    with mixin._task_span("tsk-up", "wfl-1", out_dir, owner_id="alice"):
        mixin._write_data(
            data_id="tsk-up",
            data={"items": [{"output": "ok"}]},
            source_data_ids=["tsk-source-a"],
        )

    base = out_dir / "artifacts" / "logs"
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
    assert dump[0]["attributes"]["flowmesh.kind"] == "network"
    assert dump[0]["attributes"].get("payload_bytes", 0) > 0


def test_dump_to_governance_with_merged_children(tmp_path: Path) -> None:
    mixin = _Mixin()
    out_dir = tmp_path / "task"
    with mixin._task_span("tsk-parent", "wfl-1", out_dir, owner_id="alice"):
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
            task_id="tsk-parent",
            result=result,
            dependencies_by_task=deps,
        )

    base = out_dir / "artifacts" / "logs"
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
