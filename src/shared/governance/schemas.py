"""Row schemas for the per-task lineage JSONL files.

Three files live under each task's `logs/` directory after upload:
- `events.jsonl`  — one row per recorded event (read/write/cache hit/etc.)
- `assets.jsonl`  — one row per data_id produced (asset_guid, version)
- `lineage.jsonl` — one row per (data_id, source_data_id) edge
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictRow(BaseModel):
    model_config = ConfigDict(extra="allow")


class EventRow(_StrictRow):
    timestamp: str
    event_type: str
    data_id: str
    user_id: str = ""
    batch_id: str | None = None
    event_data: Any = Field(default="")


class AssetRow(_StrictRow):
    data_id: str
    asset_guid: str
    version: int = 1
    user_id: str = ""
    created_at: str


class LineageRow(_StrictRow):
    data_id: str
    source_data_id: str
    created_at: str
