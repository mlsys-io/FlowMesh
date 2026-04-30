"""Trace analyzer over (spans, assets, lineage) JSONL rows.

Spans arrive in OTLP/JSON shape (one ``ReadableSpan.to_json()`` per line).
``flowmesh.kind`` on each span attribute classifies it as ``compute`` /
``network`` / ``marker``; the analyzer never has to maintain its own
event-type whitelist.

Per-data_id timing:

- ``start_time`` / ``end_time`` come from the ``"task"`` root span emitted by
  :meth:`worker.executors.mixins.data.DataMixin._task_span`.
- A zero-duration ``"dump to storage"`` marker stamps when data is durably
  persisted; its ``end_time`` is the data-ready timestamp.
- ``queuing_delay`` is the gap between a task's ``start_time`` and the latest
  parent's ``"dump to storage"`` end_time. Surfaced for every data_id so
  imbalance shows up off the critical path too.
"""

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict

from shared.governance.spans import FlowMeshSpanKind, Span

READY_SPAN_NAME = "dump to storage"
TASK_SPAN_NAME = "task"


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


class TaskTiming(_ProfileBase):
    data_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    queuing_delay_seconds: float
    parent_data_ids: list[str]
    blocking_parent_data_id: str | None = None


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
    per_data_id: list[TaskTiming]
    critical_path: CriticalPathSummary | None = None


def analyze(
    spans: Iterable[dict[str, Any]],
    assets: Iterable[dict[str, Any]],
    lineage: Iterable[dict[str, Any]],
    workflow_id: str | None = None,
) -> ProfileSummary:
    parsed: list[Span] = []
    for raw in spans:
        if not isinstance(raw, dict):
            continue
        try:
            parsed.append(Span.parse_otel_json(raw))
        except (ValueError, TypeError):
            continue

    asset_rows = [a for a in assets if isinstance(a, dict)]
    lineage_rows = [le for le in lineage if isinstance(le, dict)]

    asset_summaries = _asset_summaries(asset_rows)
    data_ids = sorted({s.data_id for s in parsed if s.data_id})
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

    grouped = _group_spans(parsed)
    e2e_breakdown = _obtain_breakdown(grouped)
    per_data_id = _per_data_id_timings(grouped, dep_map, data_ids)
    critical_path = (
        _compute_critical_path(grouped, dep_map, per_data_id) if data_ids else None
    )

    return ProfileSummary(
        workflow_id=workflow_id,
        event_count=len(parsed),
        data_ids=data_ids,
        assets=asset_summaries,
        lineage=lineage_edges,
        e2e_breakdown=E2EBreakdown.model_validate(e2e_breakdown),
        per_data_id=per_data_id,
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


def _group_spans(spans: list[Span]) -> dict[str, list[Span]]:
    grouped: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        if span.data_id:
            grouped[span.data_id].append(span)
    for data_id, items in grouped.items():
        items.sort(key=lambda s: s.start_time)
    return dict(grouped)


def _ready_finish(spans: list[Span]) -> datetime | None:
    ready_times = [s.end_time for s in spans if s.name == READY_SPAN_NAME]
    return max(ready_times) if ready_times else None


def _task_span(spans: list[Span]) -> Span | None:
    for span in spans:
        if span.name == TASK_SPAN_NAME and span.parent_span_id is None:
            return span
    for span in spans:
        if span.name == TASK_SPAN_NAME:
            return span
    return None


def _per_data_id_timings(
    grouped: dict[str, list[Span]],
    dep_map: dict[str, list[str]],
    data_ids: list[str],
) -> list[TaskTiming]:
    finish_ts: dict[str, datetime] = {}
    for data_id, spans in grouped.items():
        ready = _ready_finish(spans)
        if ready is not None:
            finish_ts[data_id] = ready

    timings: list[TaskTiming] = []
    for data_id in data_ids:
        spans = grouped.get(data_id) or []
        if not spans:
            continue
        task = _task_span(spans)
        if task is not None:
            start = task.start_time
            end = task.end_time
        else:
            start = min(s.start_time for s in spans)
            end = max(s.end_time for s in spans)

        parents = dep_map.get(data_id) or []
        eligible = [(p, finish_ts[p]) for p in parents if p in finish_ts]
        if eligible:
            blocking_parent, blocking_finish = max(eligible, key=lambda x: x[1])
            wait = max((start - blocking_finish).total_seconds(), 0.0)
        else:
            blocking_parent = None
            wait = 0.0

        timings.append(
            TaskTiming(
                data_id=data_id,
                start_time=start,
                end_time=end,
                duration_seconds=(end - start).total_seconds(),
                queuing_delay_seconds=wait,
                parent_data_ids=list(parents),
                blocking_parent_data_id=blocking_parent,
            )
        )
    return timings


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
    grouped: dict[str, list[Span]],
) -> dict[str, Any]:
    by_type: dict[str, list[float]] = defaultdict(list)
    by_type_batch: dict[str, dict[str, timedelta]] = defaultdict(
        lambda: defaultdict(timedelta)
    )
    network_intervals: list[tuple[datetime, datetime]] = []
    network_intervals_by_type: dict[str, list[tuple[datetime, datetime]]] = defaultdict(
        list
    )
    network_active_seconds: dict[str, list[float]] = defaultdict(list)
    all_starts: list[datetime] = []
    all_ends: list[datetime] = []

    for spans in grouped.values():
        for span in spans:
            kind = span.flowmesh_kind
            if kind == FlowMeshSpanKind.MARKER:
                continue
            if span.name == TASK_SPAN_NAME:
                all_starts.append(span.start_time)
                all_ends.append(span.end_time)
                continue

            duration = span.duration_seconds

            if kind == FlowMeshSpanKind.NETWORK:
                interval = (span.start_time, span.end_time)
                network_intervals.append(interval)
                network_intervals_by_type[span.name].append(interval)
                network_active_seconds[span.name].append(duration)
            elif kind == FlowMeshSpanKind.COMPUTE:
                by_type[span.name].append(duration)
                if span.batch_id:
                    by_type_batch[span.name][span.batch_id] += span.end_time - (
                        span.start_time
                    )

    if all_starts and all_ends:
        workflow_duration = max(all_ends) - min(all_starts)
    else:
        workflow_duration = timedelta(0)
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
        "count": [len(network_active_seconds[t]) for t in net_event_types],
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
            _avg_min_max(network_active_seconds[t])[0] for t in net_event_types
        ],
        "min_time_seconds": [
            _avg_min_max(network_active_seconds[t])[1] for t in net_event_types
        ],
        "max_time_seconds": [
            _avg_min_max(network_active_seconds[t])[2] for t in net_event_types
        ],
    }

    return {
        "hardware_summary": hardware_summary,
        "network_summary": network_summary,
        "workflow_duration_seconds": workflow_duration.total_seconds(),
        "total_network_seconds": total_network.total_seconds(),
    }


def _compute_critical_path(
    grouped: dict[str, list[Span]],
    dep_map: dict[str, list[str]],
    per_data_id: list[TaskTiming],
) -> dict[str, Any] | None:
    by_id: dict[str, TaskTiming] = {t.data_id: t for t in per_data_id}
    finish_times: dict[str, datetime] = {}
    for data_id, spans in grouped.items():
        ready = _ready_finish(spans)
        if ready is not None:
            finish_times[data_id] = ready
        elif data_id in by_id:
            finish_times[data_id] = by_id[data_id].end_time

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
    for nid in critical_path:
        timing = by_id.get(nid)
        if timing is None:
            actives.append(0.0)
            waits.append(0.0)
            continue
        active = timing.duration_seconds
        wait = timing.queuing_delay_seconds
        cp_duration += timedelta(seconds=active + wait)
        actives.append(active)
        waits.append(wait)

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
