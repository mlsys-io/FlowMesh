from shared.profile import analyze, render_mermaid, render_table


def _events() -> list[dict]:
    return [
        {
            "timestamp": "2026-04-29T00:00:00+00:00",
            "event_type": "write request transfer",
            "data_id": "tsk-1",
        },
        {
            "timestamp": "2026-04-29T00:00:01+00:00",
            "event_type": "read request initiated",
            "data_id": "tsk-1",
        },
        {
            "timestamp": "2026-04-29T00:00:02+00:00",
            "event_type": "read cache hit",
            "data_id": "tsk-1",
        },
        {
            "timestamp": "2026-04-29T00:00:03+00:00",
            "event_type": "write request transfer",
            "data_id": "tsk-2",
        },
    ]


def _assets() -> list[dict]:
    return [
        {
            "data_id": "tsk-1",
            "asset_guid": "g-1",
            "version": 1,
            "user_id": "alice",
            "created_at": "2026-04-29T00:00:00+00:00",
        },
        {
            "data_id": "tsk-2",
            "asset_guid": "g-1",
            "version": 2,
            "user_id": "alice",
            "created_at": "2026-04-29T00:00:03+00:00",
        },
    ]


def _lineage() -> list[dict]:
    return [
        {
            "data_id": "tsk-2",
            "source_data_id": "tsk-1",
            "created_at": "2026-04-29T00:00:03+00:00",
        }
    ]


def test_analyze_counts_events() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    assert summary.total_events == 4
    assert summary.total_assets == 1  # one guid spans two versions
    assert summary.total_lineage_edges == 1
    assert summary.cache_hit_count == 1
    assert summary.read_count == 2
    assert summary.write_count == 2


def test_analyze_per_data_id() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    by_id = {entry.data_id: entry for entry in summary.data_ids}
    assert by_id["tsk-1"].read_count == 2
    assert by_id["tsk-1"].cache_hit_count == 1
    assert by_id["tsk-1"].asset_guid == "g-1"
    assert by_id["tsk-2"].source_data_ids == ["tsk-1"]
    assert by_id["tsk-2"].version == 2
    assert by_id["tsk-2"].duration_sec == 0.0


def test_analyze_asset_versions() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    asset = summary.assets[0]
    assert asset.asset_guid == "g-1"
    assert asset.versions == 2
    assert asset.latest_version == 2
    assert asset.latest_data_id == "tsk-2"


def test_render_mermaid_includes_edges() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    rendered = render_mermaid(summary)
    assert rendered.startswith("graph TD")
    assert "tsk_1" in rendered
    assert "tsk_2" in rendered
    assert "tsk_1 --> tsk_2" in rendered


def test_render_table_smoke() -> None:
    summary = analyze(_events(), _assets(), _lineage())
    table = render_table(summary)
    assert "data_id" in table
    assert "tsk-1" in table
    assert "tsk-2" in table
    assert "events=4" in table


def test_phase_timings_aggregate_across_data_ids() -> None:
    events = [
        {
            "timestamp": "2026-04-29T00:00:00+00:00",
            "event_type": "queuing",
            "data_id": "tsk-1",
        },
        {
            "timestamp": "2026-04-29T00:00:01+00:00",
            "event_type": "model init",
            "data_id": "tsk-1",
        },
        {
            "timestamp": "2026-04-29T00:00:04+00:00",
            "event_type": "inference",
            "data_id": "tsk-1",
        },
        {
            "timestamp": "2026-04-29T00:00:00+00:00",
            "event_type": "queuing",
            "data_id": "tsk-2",
        },
        {
            "timestamp": "2026-04-29T00:00:03+00:00",
            "event_type": "model init",
            "data_id": "tsk-2",
        },
        {
            "timestamp": "2026-04-29T00:00:08+00:00",
            "event_type": "inference",
            "data_id": "tsk-2",
        },
    ]
    summary = analyze(events, [], [])
    by_phase = {p.event_type: p for p in summary.phase_timings}

    # "model init" should sum the (T1-T0) deltas from both data_ids: 1s + 3s = 4s.
    assert by_phase["model init"].count == 2
    assert by_phase["model init"].total_sec == 4.0
    assert by_phase["model init"].min_sec == 1.0
    assert by_phase["model init"].max_sec == 3.0
    assert by_phase["model init"].avg_sec == 2.0

    # "inference" sums (T2-T1) deltas: 3s + 5s = 8s.
    assert by_phase["inference"].total_sec == 8.0
    # Sorted descending by total — inference (8s) should come before model init.
    assert summary.phase_timings[0].event_type == "inference"

    # workflow_wall is global span: T0=00:00, last T2=00:08 → 8s.
    assert summary.workflow_wall_sec == 8.0


def test_analyze_handles_missing_rows() -> None:
    # No events but has assets + lineage — should still build data_ids.
    summary = analyze([], _assets(), _lineage())
    assert summary.total_events == 0
    assert summary.total_assets == 1
    assert summary.total_lineage_edges == 1
    assert {entry.data_id for entry in summary.data_ids} == {"tsk-1", "tsk-2"}
