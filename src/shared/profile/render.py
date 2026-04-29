"""Renderers for `ProfileSummary`.

Two views match lumilake's analyzer:
- `render_e2e` — end-to-end hardware + network breakdown across all data_ids.
- `render_critical_path` — the longest dep-chain from sink to root, with
  per-node active/wait timing and hardware + network breakdowns.

`render_mermaid` draws the lineage DAG.
"""

import re

from .analyzer import (
    HardwareSummary,
    NetworkSummary,
    ProfileSummary,
)

_MERMAID_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def _mermaid_node_id(value: str) -> str:
    cleaned = _MERMAID_SAFE.sub("_", value).strip("_")
    return cleaned or "node"


def render_mermaid(summary: ProfileSummary) -> str:
    """Mermaid `graph TD` of the lineage DAG (data_ids as nodes)."""
    lines = ["graph TD"]
    seen: set[str] = set()
    for edge in summary.lineage:
        src = _mermaid_node_id(edge.source_data_id)
        dst = _mermaid_node_id(edge.data_id)
        if src not in seen:
            lines.append(f'    {src}["{edge.source_data_id}"]')
            seen.add(src)
        if dst not in seen:
            lines.append(f'    {dst}["{edge.data_id}"]')
            seen.add(dst)
        lines.append(f"    {src} --> {dst}")
    for data_id in summary.data_ids:
        node = _mermaid_node_id(data_id)
        if node not in seen:
            lines.append(f'    {node}["{data_id}"]')
            seen.add(node)
    return "\n".join(lines)


def _format_row(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths))


def _render_hardware(hw: HardwareSummary) -> str:
    headers = ["event_type", "n", "total_sec", "avg_sec", "min_sec", "max_sec"]
    rows: list[list[str]] = []
    pairs = list(
        zip(
            hw.event_type,
            hw.count,
            hw.total_hardware_time_seconds,
            hw.avg_time_seconds,
            hw.min_time_seconds,
            hw.max_time_seconds,
        )
    )
    pairs.sort(key=lambda r: (r[2] if r[2] is not None else 0.0), reverse=True)
    for event_type, count, total, avg, mn, mx in pairs:
        rows.append(
            [
                event_type,
                str(count),
                "" if total is None else f"{total:.3f}",
                f"{avg:.3f}",
                f"{mn:.3f}",
                f"{mx:.3f}",
            ]
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows) if rows else (0,))
        for i in range(len(headers))
    ]
    out = [
        _format_row(headers, widths),
        _format_row(["-" * w for w in widths], widths),
    ]
    out.extend(_format_row(row, widths) for row in rows)
    return "\n".join(out)


def _render_network(net: NetworkSummary) -> str:
    headers = [
        "event_type",
        "n",
        "active_sec",
        "avg_sec",
        "min_sec",
        "max_sec",
    ]
    rows: list[list[str]] = []
    pairs = list(
        zip(
            net.event_type,
            net.count,
            net.total_active_seconds,
            net.avg_time_seconds,
            net.min_time_seconds,
            net.max_time_seconds,
        )
    )
    pairs.sort(key=lambda r: r[2], reverse=True)
    for event_type, count, total, avg, mn, mx in pairs:
        rows.append(
            [
                event_type,
                str(count),
                f"{total:.3f}",
                f"{avg:.3f}",
                f"{mn:.3f}",
                f"{mx:.3f}",
            ]
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows) if rows else (0,))
        for i in range(len(headers))
    ]
    out = [
        _format_row(headers, widths),
        _format_row(["-" * w for w in widths], widths),
    ]
    out.extend(_format_row(row, widths) for row in rows)
    return "\n".join(out)


def render_e2e(summary: ProfileSummary) -> str:
    """End-to-end hardware + network breakdown across all data_ids."""
    e2e = summary.e2e_breakdown
    sections = [
        f"workflow_duration={e2e.workflow_duration_seconds:.3f}s  "
        f"total_network={e2e.total_network_seconds:.3f}s  "
        f"data_ids={len(summary.data_ids)}  "
        f"events={summary.event_count}",
        "",
        "Hardware time (per event_type, summed elapsed):",
        _render_hardware(e2e.hardware_summary),
        "",
        "Network time (per event_type, merged-interval active):",
        _render_network(e2e.network_summary),
    ]
    return "\n".join(sections)


def render_critical_path(summary: ProfileSummary) -> str:
    """Critical path summary: longest dep chain with active/wait + breakdowns."""
    cp = summary.critical_path
    if cp is None:
        return "(no critical path: workflow has no events with timestamps)"

    active_wait_lines = ["data_id  active_sec  wait_sec"]
    awb = cp.active_wait_breakdown
    for data_id, active, wait in zip(awb.data_id, awb.active_seconds, awb.wait_seconds):
        active_wait_lines.append(f"{data_id}  {active:.3f}  {wait:.3f}")

    sections = [
        f"critical_path_seconds={cp.critical_path_seconds:.3f}s  "
        f"network={cp.total_network_seconds:.3f}s  "
        f"path_length={len(cp.path)}",
        "",
        "Path (sink ← root):",
        " → ".join(cp.path),
        "",
        "Per-node active vs wait:",
        "\n".join(active_wait_lines),
        "",
        "Hardware time on critical path:",
        _render_hardware(cp.hardware_summary),
        "",
        "Network time on critical path:",
        _render_network(cp.network_summary),
    ]
    return "\n".join(sections)
