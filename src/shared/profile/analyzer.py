"""Pure analyzer over (events, assets, lineage) JSONL rows.

Takes parsed rows in, returns a structured `ProfileSummary` out. No I/O —
callers (server endpoint, SDK consumer, downstream lumilake hosts) own row
loading and rendering.
"""

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class _ProfileBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetSummary(_ProfileBase):
    asset_guid: str
    latest_data_id: str
    latest_version: int
    user_id: str
    versions: int
    created_at: str | None = None


class LineageEdge(_ProfileBase):
    data_id: str
    source_data_id: str
    created_at: str | None = None


class DataIdSummary(_ProfileBase):
    data_id: str
    asset_guid: str | None = None
    version: int | None = None
    user_id: str = ""
    read_count: int = 0
    write_count: int = 0
    cache_hit_count: int = 0
    first_event_at: str | None = None
    last_event_at: str | None = None
    duration_sec: float | None = None
    source_data_ids: list[str] = []


class ProfileSummary(_ProfileBase):
    total_events: int
    total_assets: int
    total_lineage_edges: int
    cache_hit_count: int
    read_count: int
    write_count: int
    assets: list[AssetSummary]
    data_ids: list[DataIdSummary]
    lineage: list[LineageEdge]


def analyze(
    events: Iterable[dict[str, Any]],
    assets: Iterable[dict[str, Any]],
    lineage: Iterable[dict[str, Any]],
) -> ProfileSummary:
    event_rows = list(events)
    asset_rows = list(assets)
    lineage_rows = list(lineage)

    by_data_id: dict[str, DataIdSummary] = {}
    sources_by_data_id: dict[str, list[str]] = defaultdict(list)

    for row in lineage_rows:
        data_id = str(row.get("data_id") or "")
        source = str(row.get("source_data_id") or "")
        if not data_id or not source:
            continue
        sources_by_data_id[data_id].append(source)

    asset_versions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in asset_rows:
        guid = str(row.get("asset_guid") or "")
        if not guid:
            continue
        asset_versions[guid].append(row)

    asset_summaries: list[AssetSummary] = []
    asset_by_data_id: dict[str, dict[str, Any]] = {}
    for guid, rows in asset_versions.items():
        rows_sorted = sorted(rows, key=lambda r: int(r.get("version") or 0))
        latest = rows_sorted[-1]
        latest_data_id = str(latest.get("data_id") or "")
        asset_summaries.append(
            AssetSummary(
                asset_guid=guid,
                latest_data_id=latest_data_id,
                latest_version=int(latest.get("version") or 0),
                user_id=str(latest.get("user_id") or ""),
                versions=len(rows_sorted),
                created_at=str(latest.get("created_at") or "") or None,
            )
        )
        for row in rows_sorted:
            data_id = str(row.get("data_id") or "")
            if data_id:
                asset_by_data_id[data_id] = row

    read_count = 0
    write_count = 0
    cache_hit_count = 0
    for row in event_rows:
        data_id = str(row.get("data_id") or "")
        if not data_id:
            continue
        ts = _parse_ts(row.get("timestamp"))
        summary = by_data_id.setdefault(
            data_id, DataIdSummary(data_id=data_id)
        )
        et = str(row.get("event_type") or "")
        if "read" in et:
            summary.read_count += 1
            read_count += 1
            if "cache" in et and "hit" in et:
                summary.cache_hit_count += 1
                cache_hit_count += 1
        elif "write" in et:
            summary.write_count += 1
            write_count += 1
        if ts is not None:
            iso = ts.isoformat()
            if summary.first_event_at is None or iso < summary.first_event_at:
                summary.first_event_at = iso
            if summary.last_event_at is None or iso > summary.last_event_at:
                summary.last_event_at = iso

    for data_id, summary in by_data_id.items():
        if (asset_row := asset_by_data_id.get(data_id)) is not None:
            summary.asset_guid = str(asset_row.get("asset_guid") or "") or None
            summary.version = int(asset_row.get("version") or 0) or None
            summary.user_id = str(asset_row.get("user_id") or "")
        summary.source_data_ids = sources_by_data_id.get(data_id, [])
        if summary.first_event_at and summary.last_event_at:
            first = _parse_ts(summary.first_event_at)
            last = _parse_ts(summary.last_event_at)
            if first and last:
                summary.duration_sec = round(
                    (last - first).total_seconds(), 6
                )

    for data_id in asset_by_data_id:
        if data_id not in by_data_id:
            asset_row = asset_by_data_id[data_id]
            by_data_id[data_id] = DataIdSummary(
                data_id=data_id,
                asset_guid=str(asset_row.get("asset_guid") or "") or None,
                version=int(asset_row.get("version") or 0) or None,
                user_id=str(asset_row.get("user_id") or ""),
                source_data_ids=sources_by_data_id.get(data_id, []),
            )

    lineage_edges = [
        LineageEdge(
            data_id=str(row.get("data_id") or ""),
            source_data_id=str(row.get("source_data_id") or ""),
            created_at=str(row.get("created_at") or "") or None,
        )
        for row in lineage_rows
        if row.get("data_id") and row.get("source_data_id")
    ]

    return ProfileSummary(
        total_events=len(event_rows),
        total_assets=len(asset_summaries),
        total_lineage_edges=len(lineage_edges),
        cache_hit_count=cache_hit_count,
        read_count=read_count,
        write_count=write_count,
        assets=asset_summaries,
        data_ids=sorted(by_data_id.values(), key=lambda s: s.data_id),
        lineage=lineage_edges,
    )
