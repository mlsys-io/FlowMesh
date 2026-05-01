from typing import Any

from shared.governance import analyze, to_mermaid


def _span(
    name: str,
    *,
    data_id: str,
    start: str,
    end: str,
    kind: str,
    parent_id: str | None = None,
    span_id: str = "0xa3f1e9d2c5b40678",
    batch_id: str | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {"data_id": data_id, "flowmesh.kind": kind}
    if batch_id:
        attributes["batch_id"] = batch_id
    return {
        "name": name,
        "context": {
            "trace_id": "0xfbad6be5c4434181a2d394eac830dea1",
            "span_id": span_id,
        },
        "parent_id": parent_id,
        "start_time": start,
        "end_time": end,
        "status": {"status_code": "OK"},
        "attributes": attributes,
    }


def _spans() -> list[dict[str, Any]]:
    """
    Two parallel branches (tsk-1, tsk-2) feeding a synthesis (tsk-3).
    Same shape the original event-fixture exercised, now as spans.
    """
    return [
        _span(
            "task",
            data_id="tsk-1",
            start="2026-04-29T00:00:00+00:00",
            end="2026-04-29T00:00:02+00:00",
            kind="compute",
            span_id="0x1111111111111111",
        ),
        _span(
            "model load",
            data_id="tsk-1",
            start="2026-04-29T00:00:00+00:00",
            end="2026-04-29T00:00:01+00:00",
            kind="compute",
            parent_id="0x1111111111111111",
            span_id="0x1111000000000001",
            batch_id="tsk-1",
        ),
        _span(
            "dump to storage",
            data_id="tsk-1",
            start="2026-04-29T00:00:01+00:00",
            end="2026-04-29T00:00:02+00:00",
            kind="network",
            parent_id="0x1111111111111111",
            span_id="0x1111000000000002",
        ),
        _span(
            "task",
            data_id="tsk-2",
            start="2026-04-29T00:00:00+00:00",
            end="2026-04-29T00:00:04+00:00",
            kind="compute",
            span_id="0x2222222222222222",
        ),
        _span(
            "model load",
            data_id="tsk-2",
            start="2026-04-29T00:00:00+00:00",
            end="2026-04-29T00:00:03+00:00",
            kind="compute",
            parent_id="0x2222222222222222",
            span_id="0x2222000000000001",
            batch_id="tsk-2",
        ),
        _span(
            "dump to storage",
            data_id="tsk-2",
            start="2026-04-29T00:00:03+00:00",
            end="2026-04-29T00:00:04+00:00",
            kind="network",
            parent_id="0x2222222222222222",
            span_id="0x2222000000000002",
        ),
        _span(
            "task",
            data_id="tsk-3",
            start="2026-04-29T00:00:05+00:00",
            end="2026-04-29T00:00:06+00:00",
            kind="compute",
            span_id="0x3333333333333333",
        ),
        _span(
            "read",
            data_id="tsk-3",
            start="2026-04-29T00:00:05+00:00",
            end="2026-04-29T00:00:05.500000+00:00",
            kind="network",
            parent_id="0x3333333333333333",
            span_id="0x3333000000000001",
        ),
        _span(
            "dump to storage",
            data_id="tsk-3",
            start="2026-04-29T00:00:05.500000+00:00",
            end="2026-04-29T00:00:06+00:00",
            kind="network",
            parent_id="0x3333333333333333",
            span_id="0x3333000000000002",
        ),
    ]


def _assets() -> list[dict]:
    return [
        {
            "data_id": "tsk-1",
            "asset_guid": "g-1",
            "version": 1,
            "user_id": "alice",
            "created_at": "2026-04-29T00:00:02+00:00",
        },
        {
            "data_id": "tsk-2",
            "asset_guid": "g-2",
            "version": 1,
            "user_id": "alice",
            "created_at": "2026-04-29T00:00:04+00:00",
        },
        {
            "data_id": "tsk-3",
            "asset_guid": "g-3",
            "version": 1,
            "user_id": "alice",
            "created_at": "2026-04-29T00:00:06+00:00",
        },
    ]


def _lineage() -> list[dict]:
    return [
        {
            "data_id": "tsk-3",
            "source_data_id": "tsk-1",
            "created_at": "2026-04-29T00:00:06+00:00",
        },
        {
            "data_id": "tsk-3",
            "source_data_id": "tsk-2",
            "created_at": "2026-04-29T00:00:06+00:00",
        },
    ]


def test_e2e_breakdown_workflow_duration_and_network_union() -> None:
    summary = analyze(_spans(), _assets(), _lineage())
    e2e = summary.e2e_breakdown
    assert e2e.workflow_duration_seconds == 6.0
    assert e2e.total_network_seconds > 0


def test_e2e_hardware_summary_lists_compute_spans() -> None:
    summary = analyze(_spans(), _assets(), _lineage())
    hw = summary.e2e_breakdown.hardware_summary
    types = set(hw.event_type)
    assert "model load" in types
    assert "read" not in types
    assert "dump to storage" not in types
    assert "task" not in types


def test_e2e_network_summary_includes_transfers() -> None:
    summary = analyze(_spans(), _assets(), _lineage())
    net = summary.e2e_breakdown.network_summary
    types = set(net.event_type)
    assert "dump to storage" in types
    assert "read" in types


def test_critical_path_picks_synthesis_chain() -> None:
    summary = analyze(_spans(), _assets(), _lineage())
    cp = summary.critical_path
    assert cp is not None
    assert cp.path == ["tsk-2", "tsk-3"]
    awb = cp.active_wait_breakdown
    assert awb.data_id == ["tsk-2", "tsk-3"]
    assert awb.active_seconds[1] == 1.0
    assert awb.wait_seconds[1] == 1.0


def test_per_data_id_queuing_delays() -> None:
    summary = analyze(_spans(), _assets(), _lineage())
    per_id = {t.data_id: t for t in summary.per_data_id}
    assert per_id["tsk-1"].queuing_delay_seconds == 0.0
    assert per_id["tsk-2"].queuing_delay_seconds == 0.0
    assert per_id["tsk-3"].queuing_delay_seconds == 1.0
    assert per_id["tsk-3"].blocking_parent_data_id == "tsk-2"
    assert per_id["tsk-3"].duration_seconds == 1.0


def test_to_mermaid_includes_edges() -> None:
    summary = analyze(_spans(), _assets(), _lineage())
    rendered = to_mermaid(summary)
    assert rendered.startswith("graph TD")
    assert "tsk_1" in rendered
    assert "tsk_3" in rendered
    assert "-->" in rendered


def test_analyze_handles_empty_spans() -> None:
    summary = analyze([], _assets(), _lineage())
    assert summary.event_count == 0
    assert summary.data_ids == []
    assert summary.per_data_id == []
    assert summary.e2e_breakdown.workflow_duration_seconds == 0.0
    assert summary.critical_path is None
