from shared.profile import analyze, to_mermaid


def _events() -> list[dict]:
    # Two parallel branches (same batch_id="bat-1") + one downstream synthesis.
    return [
        # Branch tsk-1
        {
            "timestamp": "2026-04-29T00:00:00+00:00",
            "event_type": "queuing for execution",
            "data_id": "tsk-1",
            "batch_id": "bat-1",
        },
        {
            "timestamp": "2026-04-29T00:00:01+00:00",
            "event_type": "model initialization",
            "data_id": "tsk-1",
            "batch_id": "bat-1",
        },
        {
            "timestamp": "2026-04-29T00:00:02+00:00",
            "event_type": "write request transfer",
            "data_id": "tsk-1",
            "batch_id": "bat-1",
        },
        {
            "timestamp": "2026-04-29T00:00:02+00:00",
            "event_type": "dump to storage",
            "data_id": "tsk-1",
            "batch_id": "bat-1",
        },
        # Branch tsk-2 (parallel)
        {
            "timestamp": "2026-04-29T00:00:00+00:00",
            "event_type": "queuing for execution",
            "data_id": "tsk-2",
            "batch_id": "bat-1",
        },
        {
            "timestamp": "2026-04-29T00:00:03+00:00",
            "event_type": "model initialization",
            "data_id": "tsk-2",
            "batch_id": "bat-1",
        },
        {
            "timestamp": "2026-04-29T00:00:04+00:00",
            "event_type": "write request transfer",
            "data_id": "tsk-2",
            "batch_id": "bat-1",
        },
        {
            "timestamp": "2026-04-29T00:00:04+00:00",
            "event_type": "dump to storage",
            "data_id": "tsk-2",
            "batch_id": "bat-1",
        },
        # Synthesis tsk-3 (depends on tsk-1, tsk-2)
        {
            "timestamp": "2026-04-29T00:00:05+00:00",
            "event_type": "read response transfer",
            "data_id": "tsk-3",
            "batch_id": "bat-3",
        },
        {
            "timestamp": "2026-04-29T00:00:06+00:00",
            "event_type": "write request transfer",
            "data_id": "tsk-3",
            "batch_id": "bat-3",
        },
        {
            "timestamp": "2026-04-29T00:00:06+00:00",
            "event_type": "dump to storage",
            "data_id": "tsk-3",
            "batch_id": "bat-3",
        },
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
    summary = analyze(_events(), _assets(), _lineage())
    e2e = summary.e2e_breakdown
    # Span: 00:00:00 → 00:00:06 = 6 sec.
    assert e2e.workflow_duration_seconds == 6.0
    # Network events: 3 transfers (write, write, read, write). Their elapsed
    # intervals overlap with each other across data_ids, but merging them
    # gives a non-zero, deterministic active span.
    assert e2e.total_network_seconds > 0


def test_e2e_hardware_summary_lists_event_types() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    hw = summary.e2e_breakdown.hardware_summary
    types = set(hw.event_type)
    # `dump to storage` is hardware-side (not in NETWORK_EVENT_TYPES).
    assert "dump to storage" in types
    assert "model initialization" in types
    # Network event_types must NOT appear in hardware.
    assert "write request transfer" not in types
    assert "read response transfer" not in types


def test_e2e_network_summary_includes_transfers() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    net = summary.e2e_breakdown.network_summary
    types = set(net.event_type)
    assert "write request transfer" in types
    assert "read response transfer" in types


def test_critical_path_picks_synthesis_chain() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    cp = summary.critical_path
    assert cp is not None
    # Sink is tsk-3 (latest finish at T6). It depends on both branches; the
    # later-finishing one is tsk-2 (T4). Path: tsk-2 → tsk-3.
    assert cp.path == ["tsk-2", "tsk-3"]
    # active_seconds for tsk-3: from first event at T5 to dump at T6 = 1s.
    awb = cp.active_wait_breakdown
    assert awb.data_id == ["tsk-2", "tsk-3"]
    assert awb.active_seconds[1] == 1.0
    # wait_seconds for tsk-3 between tsk-2 finish (T4) and tsk-3 start (T5).
    assert awb.wait_seconds[1] == 1.0


def test_to_mermaid_includes_edges() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    rendered = to_mermaid(summary)
    assert rendered.startswith("graph TD")
    assert "tsk_1" in rendered
    assert "tsk_3" in rendered
    assert "-->" in rendered


def test_analyze_handles_empty_events() -> None:
    summary = analyze([], _assets(), _lineage())
    assert summary.event_count == 0
    assert summary.data_ids == []
    assert summary.e2e_breakdown.workflow_duration_seconds == 0.0
    assert summary.critical_path is None
