from shared.governance import AssetRow, EventRow, LineageRow, data_key


def test_event_row_round_trip() -> None:
    row = EventRow(
        timestamp="2026-04-29T00:00:00+00:00",
        event_type="write request transfer",
        data_id="tsk-1",
        user_id="alice",
        batch_id="tsk-1",
        event_data={"size": 42},
    )
    serialized = row.model_dump()
    parsed = EventRow.model_validate(serialized)
    assert parsed.event_type == "write request transfer"
    assert parsed.event_data == {"size": 42}


def test_asset_row_defaults() -> None:
    row = AssetRow(
        data_id="tsk-1",
        asset_guid="guid-1",
        created_at="2026-04-29T00:00:00+00:00",
    )
    assert row.version == 1
    assert row.user_id == ""


def test_lineage_row_required_fields() -> None:
    row = LineageRow(
        data_id="tsk-2",
        source_data_id="tsk-1",
        created_at="2026-04-29T00:00:00+00:00",
    )
    assert row.data_id == "tsk-2"
    assert row.source_data_id == "tsk-1"


def test_data_key_format() -> None:
    assert data_key("tsk-abc") == "flowmesh:data:tsk-abc"
