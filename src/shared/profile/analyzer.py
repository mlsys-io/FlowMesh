"""Trace analyzer over (events, assets, lineage) JSONL rows.

Mirrors the analysis shape used by lumilake's `TraceAnalyzer`: an end-to-end
breakdown (hardware + network) across all data_ids in a workflow, plus a
critical-path summary that walks the lineage DAG from the latest-finishing
sink back to the root, breaking down active vs. wait time at each step.

A `dump to storage` event marks data as ready (the READY marker). Network
events (`read response transfer`, `write request transfer`) carry their own
elapsed-derived intervals, and total network time is the union of those
intervals so overlapping reads / writes don't double-count.
"""

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict

READY_EVENT_TYPE = "dump to storage"
NETWORK_EVENT_TYPES = frozenset(
    {
        "read request transfer",
        "read response transfer",
        "write request transfer",
        "read response retrieval",
    }
)
SKIPPED_EVENT_TYPES = frozenset({"read request initiated"})


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
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


class HardwareSummary(_ProfileBase):
    event_type: list[str]
    count: list[int]
    total_hardware_time_seconds: list[float | None]
    avg_time_seconds: list[float]
    min_time_seconds: list[float]
    max_time_seconds: list[float]


class NetworkSummary(_ProfileBase):
    event_type: list[str]
    count: list[int]
    total_active_seconds: list[float]
    avg_time_seconds: list[float]
    min_time_seconds: list[float]
    max_time_seconds: list[float]


class E2EBreakdown(_ProfileBase):
    hardware_summary: HardwareSummary
    network_summary: NetworkSummary
    workflow_duration_seconds: float
    total_network_seconds: float


class ActiveWaitBreakdown(_ProfileBase):
    data_id: list[str]
    active_seconds: list[float]
    wait_seconds: list[float]


class CriticalPathSummary(_ProfileBase):
    path: list[str]
    critical_path_seconds: float
    active_wait_breakdown: ActiveWaitBreakdown
    hardware_summary: HardwareSummary
    network_summary: NetworkSummary
    total_network_seconds: float


class ProfileSummary(_ProfileBase):
    workflow_id: str | None = None
    event_count: int
    data_ids: list[str]
    assets: list[AssetSummary]
    lineage: list[LineageEdge]
    e2e_breakdown: E2EBreakdown
    critical_path: CriticalPathSummary | None = None


def analyze(
    events: Iterable[dict[str, Any]],
    assets: Iterable[dict[str, Any]],
    lineage: Iterable[dict[str, Any]],
    workflow_id: str | None = None,
) -> ProfileSummary:
    event_rows = [e for e in events if isinstance(e, dict)]
    asset_rows = [a for a in assets if isinstance(a, dict)]
    lineage_rows = [le for le in lineage if isinstance(le, dict)]

    asset_summaries = _asset_summaries(asset_rows)
    data_ids = sorted({str(e.get("data_id") or "") for e in event_rows} - {""})
    dep_map = _dep_map(lineage_rows)
    lineage_edges = [
        LineageEdge(
            data_id=str(le.get("data_id") or ""),
            source_data_id=str(le.get("source_data_id") or ""),
            created_at=str(le.get("created_at") or "") or None,
        )
        for le in lineage_rows
        if le.get("data_id") and le.get("source_data_id")
    ]

    grouped = _group_events(event_rows, data_ids, dep_map)
    e2e_breakdown = _obtain_breakdown(grouped)
    critical_path = _compute_critical_path(grouped, dep_map) if data_ids else None

    return ProfileSummary(
        workflow_id=workflow_id,
        event_count=len(event_rows),
        data_ids=data_ids,
        assets=asset_summaries,
        lineage=lineage_edges,
        e2e_breakdown=E2EBreakdown.model_validate(e2e_breakdown),
        critical_path=(
            CriticalPathSummary.model_validate(critical_path)
            if critical_path is not None
            else None
        ),
    )


def _asset_summaries(rows: list[dict[str, Any]]) -> list[AssetSummary]:
    asset_versions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        guid = str(row.get("asset_guid") or "")
        if not guid:
            continue
        asset_versions[guid].append(row)
    summaries: list[AssetSummary] = []
    for guid, items in asset_versions.items():
        items_sorted = sorted(items, key=lambda r: int(r.get("version") or 0))
        latest = items_sorted[-1]
        summaries.append(
            AssetSummary(
                asset_guid=guid,
                latest_data_id=str(latest.get("data_id") or ""),
                latest_version=int(latest.get("version") or 0),
                user_id=str(latest.get("user_id") or ""),
                versions=len(items_sorted),
                created_at=str(latest.get("created_at") or "") or None,
            )
        )
    return summaries


def _dep_map(lineage_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    dep_map: dict[str, list[str]] = defaultdict(list)
    for row in lineage_rows:
        target = str(row.get("data_id") or "")
        source = str(row.get("source_data_id") or "")
        if target and source:
            dep_map[target].append(source)
    return dict(dep_map)


def _group_events(
    events: list[dict[str, Any]],
    data_ids: list[str],
    dep_map: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evt in events:
        data_id = str(evt.get("data_id") or "")
        if not data_id:
            continue
        grouped[data_id].append(evt)

    finish_ts: dict[str, datetime] = {}
    for data_id, evts in grouped.items():
        ready_times: list[datetime] = []
        for e in evts:
            if e.get("event_type") != READY_EVENT_TYPE:
                continue
            parsed = _parse_ts(e.get("timestamp"))
            if parsed is not None:
                ready_times.append(parsed)
        if ready_times:
            finish_ts[data_id] = max(ready_times)

    for data_id in data_ids:
        evts = grouped.get(data_id) or []
        evts.sort(key=lambda e: _parse_ts(e.get("timestamp")) or datetime.min)
        prev_ts: datetime | None = None
        parents = dep_map.get(data_id)
        for idx, evt in enumerate(evts):
            ts = _parse_ts(evt.get("timestamp"))
            if ts is None:
                evt["elapsed"] = timedelta(0)
                continue
            baseline: datetime | None = prev_ts
            if baseline is None and parents and idx == 0:
                parent_finishes = [finish_ts[p] for p in parents if p in finish_ts]
                if parent_finishes:
                    baseline = max(parent_finishes)
            if baseline is None:
                elapsed = timedelta(0)
            elif ts < baseline:
                elapsed = timedelta(0)
            else:
                elapsed = ts - baseline
            evt["elapsed"] = elapsed
            prev_ts = ts
    return dict(grouped)


def _merge_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    sorted_ivl = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[datetime, datetime]] = []
    cur_start, cur_end = sorted_ivl[0]
    for start, end in sorted_ivl[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _avg_min_max(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return sum(values) / len(values), min(values), max(values)


def _obtain_breakdown(
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_type: dict[str, list[float]] = defaultdict(list)
    by_type_batch: dict[str, dict[str, timedelta]] = defaultdict(
        lambda: defaultdict(timedelta)
    )
    network_intervals: list[tuple[datetime, datetime]] = []
    network_intervals_by_type: dict[str, list[tuple[datetime, datetime]]] = defaultdict(
        list
    )
    network_gap_seconds: dict[str, list[float]] = defaultdict(list)
    all_timestamps: list[datetime] = []

    for evts in grouped.values():
        for evt in evts:
            event_type = str(evt.get("event_type") or "")
            if event_type in SKIPPED_EVENT_TYPES:
                continue
            ts = _parse_ts(evt.get("timestamp"))
            if ts is None:
                continue
            all_timestamps.append(ts)
            elapsed = evt.get("elapsed")
            if not isinstance(elapsed, timedelta):
                elapsed = timedelta(0)

            if event_type in NETWORK_EVENT_TYPES:
                interval = (ts - elapsed, ts)
                network_intervals.append(interval)
                network_intervals_by_type[event_type].append(interval)
                network_gap_seconds[event_type].append(elapsed.total_seconds())
            else:
                by_type[event_type].append(elapsed.total_seconds())
                batch_id = str(evt.get("batch_id") or "")
                if batch_id:
                    by_type_batch[event_type][batch_id] += elapsed

    workflow_duration = (
        max(all_timestamps) - min(all_timestamps)
        if len(all_timestamps) >= 2
        else timedelta(0)
    )
    total_network = sum(
        (end - start for start, end in _merge_intervals(network_intervals)),
        timedelta(0),
    )

    hw_event_types = list(by_type.keys())
    hardware_summary = {
        "event_type": hw_event_types,
        "count": [len(by_type[t]) for t in hw_event_types],
        "total_hardware_time_seconds": [
            (
                sum(
                    (d for d in by_type_batch[t].values()), timedelta(0)
                ).total_seconds()
                if by_type_batch[t]
                else None
            )
            for t in hw_event_types
        ],
        "avg_time_seconds": [_avg_min_max(by_type[t])[0] for t in hw_event_types],
        "min_time_seconds": [_avg_min_max(by_type[t])[1] for t in hw_event_types],
        "max_time_seconds": [_avg_min_max(by_type[t])[2] for t in hw_event_types],
    }

    net_event_types = list(network_intervals_by_type.keys())
    network_summary = {
        "event_type": net_event_types,
        "count": [len(network_gap_seconds[t]) for t in net_event_types],
        "total_active_seconds": [
            sum(
                (
                    end - start
                    for start, end in _merge_intervals(network_intervals_by_type[t])
                ),
                timedelta(0),
            ).total_seconds()
            for t in net_event_types
        ],
        "avg_time_seconds": [
            _avg_min_max(network_gap_seconds[t])[0] for t in net_event_types
        ],
        "min_time_seconds": [
            _avg_min_max(network_gap_seconds[t])[1] for t in net_event_types
        ],
        "max_time_seconds": [
            _avg_min_max(network_gap_seconds[t])[2] for t in net_event_types
        ],
    }

    return {
        "hardware_summary": hardware_summary,
        "network_summary": network_summary,
        "workflow_duration_seconds": workflow_duration.total_seconds(),
        "total_network_seconds": total_network.total_seconds(),
    }


def _compute_critical_path(
    grouped: dict[str, list[dict[str, Any]]],
    dep_map: dict[str, list[str]],
) -> dict[str, Any] | None:
    start_times: dict[str, datetime] = {}
    finish_times: dict[str, datetime] = {}
    for data_id, evts in grouped.items():
        ts_values: list[datetime] = []
        ready_times: list[datetime] = []
        for e in evts:
            parsed = _parse_ts(e.get("timestamp"))
            if parsed is None:
                continue
            ts_values.append(parsed)
            if e.get("event_type") == READY_EVENT_TYPE:
                ready_times.append(parsed)
        if not ts_values:
            continue
        start_times[data_id] = min(ts_values)
        finish_times[data_id] = max(ready_times) if ready_times else max(ts_values)

    if not finish_times:
        return None

    sink = max(finish_times, key=lambda k: finish_times[k])
    path_rev: list[str] = [sink]
    cursor = sink
    while parents := dep_map.get(cursor):
        eligible = [(p, finish_times[p]) for p in parents if p in finish_times]
        if not eligible:
            break
        latest_parent = max(eligible, key=lambda x: x[1])[0]
        path_rev.append(latest_parent)
        cursor = latest_parent
    critical_path = list(reversed(path_rev))

    actives: list[float] = []
    waits: list[float] = []
    cp_duration = timedelta(0)
    for i, nid in enumerate(critical_path):
        st = start_times[nid]
        ft = finish_times[nid]
        active = ft - st
        wait = timedelta(0)
        baseline_finish: datetime | None = None
        if parents := dep_map.get(nid):
            finishes = [finish_times[p] for p in parents if p in finish_times]
            if finishes:
                baseline_finish = max(finishes)
        if baseline_finish is None and i > 0:
            baseline_finish = finish_times.get(critical_path[i - 1])
        if baseline_finish is not None and st > baseline_finish:
            wait = st - baseline_finish
        cp_duration += active + wait
        actives.append(active.total_seconds())
        waits.append(wait.total_seconds())

    cp_breakdown = _obtain_breakdown(
        {nid: grouped[nid] for nid in critical_path if nid in grouped}
    )
    cp_breakdown.pop("workflow_duration_seconds", None)

    return {
        "path": critical_path,
        "critical_path_seconds": cp_duration.total_seconds(),
        "active_wait_breakdown": {
            "data_id": critical_path,
            "active_seconds": actives,
            "wait_seconds": waits,
        },
        "hardware_summary": cp_breakdown["hardware_summary"],
        "network_summary": cp_breakdown["network_summary"],
        "total_network_seconds": cp_breakdown["total_network_seconds"],
    }
