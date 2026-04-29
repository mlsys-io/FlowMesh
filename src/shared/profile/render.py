"""Plain-text renderers for `ProfileSummary` shapes that don't depend on Rich.

Lives in `shared` so any consumer (server, SDK, CLI) can produce a Mermaid
graph without pulling a renderer dep. Rich-based table/tree rendering for
interactive terminals is in the CLI; pandas-based dataframe views are in the
SDK.
"""

import re

from .analyzer import ProfileSummary

_MERMAID_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def _mermaid_node_id(value: str) -> str:
    cleaned = _MERMAID_SAFE.sub("_", value).strip("_")
    return cleaned or "node"


def to_mermaid(summary: ProfileSummary) -> str:
    """Mermaid `graph TD` source for the lineage DAG (data_ids as nodes)."""
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
