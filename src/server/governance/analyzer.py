"""Trace analyzer over (spans, assets, lineage) JSONL rows.

Per-data_id timing comes from the ``"task"`` root span; ``"dump to storage"``
end_time is the data-ready timestamp; ``queuing_delay`` =
task.start − max(parent.dump_to_storage.end).
"""

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict

from shared.governance.spans import (
    READY_SPAN_NAME,
    TASK_SPAN_NAME,
    FlowMeshSpanKind,
)

from .spans import Span


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


class EventSummary(_ProfileBase):
    """Per-event-type duration aggregates as parallel lists.

    For ``compute`` spans, ``total_seconds[i]`` is the per-batch sum (so
    parallel spans within a batch collapse). For ``network`` spans it is the
    merged-interval wall-clock time (so overlapping reads/writes collapse).
    """

    event_type: list[str]
    count: list[int]
    total_seconds: list[float]
    avg_seconds: list[float]
    min_seconds: list[float]
    max_seconds: list[float]


class E2EBreakdown(_ProfileBase):
    hardware_summary: EventSummary
    network_summary: EventSummary
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
    hardware_summary: EventSummary
    network_summary: EventSummary
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
    """Build a :class:`ProfileSummary` from raw JSONL rows for a single workflow.

    ``spans`` rows are parsed via :class:`Span`; malformed entries are dropped.
    ``assets`` and ``lineage`` rows are passed straight through as dicts. The
    returned summary contains the asset rollup, full DAG edges, an end-to-end
    breakdown, per-data_id timings (with queuing delay + blocking parent),
    and a critical-path subset.
    """
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
    data_ids = sorted({did for s in parsed if (did := s.attributes.data_id)})
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
    """Group asset rows by ``asset_guid`` and emit one summary per asset.

    Each summary points at the highest-version row (``latest_*``) and reports
    the total version count.
    """
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
    """Build a ``data_id -> [source_data_id, ...]`` adjacency from lineage edges."""
    dep_map: dict[str, list[str]] = defaultdict(list)
    for row in lineage_rows:
        target = str(row.get("data_id") or "")
        source = str(row.get("source_data_id") or "")
        if target and source:
            dep_map[target].append(source)
    return dict(dep_map)


def _group_spans(spans: list[Span]) -> dict[str, list[Span]]:
    """Bucket spans by ``attributes.data_id`` and sort each bucket by start time.

    Spans without a ``data_id`` (e.g. third-party telemetry) are dropped here.
    """
    grouped: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        if data_id := span.attributes.data_id:
            grouped[data_id].append(span)
    for data_id, items in grouped.items():
        items.sort(key=lambda s: s.start_time)
    return dict(grouped)


def _ready_finish(spans: list[Span]) -> datetime | None:
    """Latest ``"dump to storage"`` span ``end_time`` — the data-ready boundary
    for a task. ``None`` if no such span exists (e.g. failed task)."""
    ready_times = [s.end_time for s in spans if s.name == READY_SPAN_NAME]
    return max(ready_times) if ready_times else None


def _task_span(spans: list[Span]) -> Span | None:
    """Pick the root ``"task"`` span, preferring one without a parent."""
    for span in spans:
        if span.name == TASK_SPAN_NAME and span.parent_id is None:
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
    """Per-data_id start/end timestamps + queuing delay against parents.

    ``queuing_delay = task.start - max(parent.dump_to_storage.end)``. Falls
    back to ``min(span.start) / max(span.end)`` when a data_id has no root
    ``"task"`` span (e.g. merged children that only emit per-task markers).
    """
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
                parent_data_ids=parents.copy(),
                blocking_parent_data_id=blocking_parent,
            )
        )
    return timings


def _merge_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Collapse overlapping ``(start, end)`` intervals so concurrent network
    spans count once toward total active time."""
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
    """``(avg, min, max)`` over ``values``; zeros when empty."""
    if not values:
        return 0.0, 0.0, 0.0
    return sum(values) / len(values), min(values), max(values)


def _obtain_breakdown(
    grouped: dict[str, list[Span]],
) -> dict[str, Any]:
    """Aggregate per-event-type compute / network / wall stats over a span set.

    Compute totals are summed per ``batch_id`` then across batches (parallel
    spans within a batch collapse). Network totals use merged-interval
    wall-clock so concurrent reads/writes count once. The ``"task"`` root
    span and ``MARKER`` kind spans are excluded from event-type aggregates;
    ``"task"`` start/end bound ``workflow_duration_seconds``.
    """
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
            kind = span.attributes.flowmesh_kind
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
                if batch_id := span.attributes.batch_id:
                    by_type_batch[span.name][batch_id] += span.end_time - (
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
        "total_seconds": [
            sum((d for d in by_type_batch[t].values()), timedelta(0)).total_seconds()
            for t in hw_event_types
        ],
        "avg_seconds": [_avg_min_max(by_type[t])[0] for t in hw_event_types],
        "min_seconds": [_avg_min_max(by_type[t])[1] for t in hw_event_types],
        "max_seconds": [_avg_min_max(by_type[t])[2] for t in hw_event_types],
    }

    net_event_types = list(network_intervals_by_type.keys())
    network_summary = {
        "event_type": net_event_types,
        "count": [len(network_active_seconds[t]) for t in net_event_types],
        "total_seconds": [
            sum(
                (
                    end - start
                    for start, end in _merge_intervals(network_intervals_by_type[t])
                ),
                timedelta(0),
            ).total_seconds()
            for t in net_event_types
        ],
        "avg_seconds": [
            _avg_min_max(network_active_seconds[t])[0] for t in net_event_types
        ],
        "min_seconds": [
            _avg_min_max(network_active_seconds[t])[1] for t in net_event_types
        ],
        "max_seconds": [
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
    """Walk back from the latest-finishing data_id, picking the slowest parent
    at each hop, to surface the bottleneck chain.

    ``critical_path_seconds`` is the sum of active + wait along the chain.
    The CP-restricted breakdown also pulls in spans from any merge-parent
    ``batch_id`` referenced by CP nodes — otherwise shared work (model load,
    generation) emitted under the merge parent's data_id would be missed
    when a merged-child branch is on the path.
    """
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

    # Expand CP membership through merged-execution batch_ids so the breakdown
    # captures shared work (model load, generation) emitted under the merge
    # parent's data_id when a merged-child branch lands on the path.
    cp_data_ids: set[str] = set(critical_path)
    for nid in critical_path:
        for span in grouped.get(nid, []):
            batch_id = span.attributes.batch_id
            if batch_id and batch_id != nid and batch_id in grouped:
                cp_data_ids.add(batch_id)

    cp_breakdown = _obtain_breakdown(
        {nid: grouped[nid] for nid in cp_data_ids if nid in grouped}
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
