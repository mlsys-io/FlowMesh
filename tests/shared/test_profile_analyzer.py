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


def test_analyze_handles_missing_rows() -> None:
    # No events but has assets + lineage — should still build data_ids.
    summary = analyze([], _assets(), _lineage())
    assert summary.total_events == 0
    assert summary.total_assets == 1
    assert summary.total_lineage_edges == 1
    assert {entry.data_id for entry in summary.data_ids} == {"tsk-1", "tsk-2"}
