"""Row schemas for the per-task lineage JSONL files.

Three files live under each task's `logs/` directory after upload:
- `spans.jsonl`   — one OTel span per row (see `spans.py` for the parser)
- `assets.jsonl`  — one row per data_id produced (asset_guid, version)
- `lineage.jsonl` — one row per (data_id, source_data_id) edge
"""

from pydantic import BaseModel, ConfigDict


class _StrictRow(BaseModel):
    model_config = ConfigDict(extra="allow")


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
