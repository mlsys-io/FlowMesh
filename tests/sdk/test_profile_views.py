"""SDK profile-view helpers (mermaid renderer)."""

from flowmesh.profile_views import to_mermaid


def test_to_mermaid_renders_lineage_edges() -> None:
    summary = {
        "workflow_id": "wfl-1",
        "event_count": 0,
        "data_ids": ["tsk-1", "tsk-2", "tsk-3"],
        "assets": [],
        "lineage": [
            {
                "data_id": "tsk-3",
                "source_data_id": "tsk-1",
                "created_at": "2026-04-29T00:00:00+00:00",
            },
            {
                "data_id": "tsk-3",
                "source_data_id": "tsk-2",
                "created_at": "2026-04-29T00:00:00+00:00",
            },
        ],
        "e2e_breakdown": {
            "hardware_summary": {
                "event_type": [],
                "count": [],
                "total_seconds": [],
                "avg_seconds": [],
                "min_seconds": [],
                "max_seconds": [],
            },
            "network_summary": {
                "event_type": [],
                "count": [],
                "total_seconds": [],
                "avg_seconds": [],
                "min_seconds": [],
                "max_seconds": [],
            },
            "workflow_duration_seconds": 0.0,
            "total_network_seconds": 0.0,
        },
        "per_data_id": [],
        "critical_path": None,
    }
    rendered = to_mermaid(summary)
    assert rendered.startswith("graph TD")
    assert "tsk_1" in rendered
    assert "tsk_3" in rendered
    assert "-->" in rendered
