"""Renderers for `ProfileSummary` — Mermaid graphs and terminal tables."""

import re

from .analyzer import ProfileSummary

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
    for entry in summary.data_ids:
        node = _mermaid_node_id(entry.data_id)
        if node not in seen:
            lines.append(f'    {node}["{entry.data_id}"]')
            seen.add(node)
    return "\n".join(lines)


def _format_row(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths))


def render_table(summary: ProfileSummary) -> str:
    """Terminal-friendly per-data_id summary table."""
    headers = [
        "data_id",
        "asset_guid",
        "ver",
        "reads",
        "writes",
        "cache_hits",
        "duration_sec",
        "sources",
    ]
    rows: list[list[str]] = []
    for entry in summary.data_ids:
        rows.append(
            [
                entry.data_id,
                entry.asset_guid or "",
                str(entry.version) if entry.version is not None else "",
                str(entry.read_count),
                str(entry.write_count),
                str(entry.cache_hit_count),
                (
                    f"{entry.duration_sec:.3f}"
                    if entry.duration_sec is not None
                    else ""
                ),
                ",".join(entry.source_data_ids),
            ]
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows) if rows else (0,))
        for i in range(len(headers))
    ]
    out_lines = [
        _format_row(headers, widths),
        _format_row(["-" * w for w in widths], widths),
    ]
    out_lines.extend(_format_row(row, widths) for row in rows)
    out_lines.append("")
    out_lines.append(
        f"events={summary.total_events}  assets={summary.total_assets}  "
        f"edges={summary.total_lineage_edges}  cache_hits={summary.cache_hit_count}"
    )
    return "\n".join(out_lines)
