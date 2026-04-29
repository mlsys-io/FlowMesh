"""`flowmesh profile fetch ...` — analyze a workflow's lineage."""

from collections import defaultdict
from typing import Any

import typer
from flowmesh.exceptions import FlowMeshError
from rich.box import SIMPLE
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

from shared.profile import (
    HardwareSummary,
    NetworkSummary,
    ProfileSummary,
)

from ..core import logging
from ..core.runtime import flowmesh_client_from_config
from ..core.typer import get_typer

app = get_typer(help="Profile a workflow's events / assets / lineage.")
console = Console()


def _to_summary(payload: dict[str, Any]) -> ProfileSummary:
    return ProfileSummary.model_validate(payload)


def _short(data_id: str) -> str:
    if len(data_id) <= 16:
        return data_id
    return f"{data_id[:8]}…{data_id[-4:]}"


def _compute_table(hw: HardwareSummary, title: str) -> Table:
    """Render the per-event-type compute (GPU/CPU) time table.

    Schema field is `hardware_summary` for lumilake compatibility, but
    "Compute time" is clearer in user-facing UI than "Hardware time".
    """
    table = Table(title=title, box=SIMPLE, header_style="bold cyan", title_style="bold")
    table.add_column("event_type", style="cyan", no_wrap=True)
    table.add_column("n", justify="right")
    table.add_column("total_sec", justify="right", style="bold")
    table.add_column("avg", justify="right", style="dim")
    table.add_column("min", justify="right", style="dim")
    table.add_column("max", justify="right", style="dim")
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
    max_total = max((p[2] or 0.0 for p in pairs), default=0.0)
    for event_type, count, total, avg, mn, mx in pairs:
        total_str = "" if total is None else f"{total:.3f}"
        if total is not None and total > 0 and max_total > 0:
            ratio = total / max_total
            color = "red" if ratio > 0.5 else "yellow" if ratio > 0.1 else "green"
            total_str = f"[{color}]{total_str}[/{color}]"
        table.add_row(
            event_type, str(count), total_str, f"{avg:.3f}", f"{mn:.3f}", f"{mx:.3f}"
        )
    return table


def _network_table(net: NetworkSummary, title: str) -> Table:
    table = Table(
        title=title, box=SIMPLE, header_style="bold magenta", title_style="bold"
    )
    table.add_column("event_type", style="magenta", no_wrap=True)
    table.add_column("n", justify="right")
    table.add_column("active_sec", justify="right", style="bold")
    table.add_column("avg", justify="right", style="dim")
    table.add_column("min", justify="right", style="dim")
    table.add_column("max", justify="right", style="dim")
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
        table.add_row(
            event_type,
            str(count),
            f"{total:.3f}",
            f"{avg:.3f}",
            f"{mn:.3f}",
            f"{mx:.3f}",
        )
    return table


def _lineage_tree(summary: ProfileSummary) -> Tree:
    """DAG rendered as a Rich tree rooted at sinks (leaves = upstream sources).

    Cycles aren't possible in a lineage DAG; same data_id can appear under
    multiple parents and we render it each time it appears.
    """
    children_of: dict[str, list[str]] = defaultdict(list)
    parents_of: dict[str, list[str]] = defaultdict(list)
    for edge in summary.lineage:
        children_of[edge.source_data_id].append(edge.data_id)
        parents_of[edge.data_id].append(edge.source_data_id)

    cp_set: set[str] = (
        set(summary.critical_path.path) if summary.critical_path else set()
    )
    sinks = [d for d in summary.data_ids if d not in children_of]

    root = Tree("[bold]lineage DAG[/bold]")
    seen_in_branch: set[str] = set()

    def _label(data_id: str) -> str:
        marker = " [bold red]◆ critical path[/bold red]" if data_id in cp_set else ""
        return f"[cyan]{data_id}[/cyan]{marker}"

    def _walk(parent_node: Tree, data_id: str, branch: frozenset[str]) -> None:
        node = parent_node.add(_label(data_id))
        for upstream in parents_of.get(data_id, []):
            if upstream in branch:
                node.add(f"[dim]↺ {upstream} (cycle skipped)[/dim]")
                continue
            _walk(node, upstream, branch | {upstream})

    if not sinks:
        return root.add("[dim](no events with timestamps)[/dim]")
    for sink in sinks:
        _walk(root, sink, frozenset({sink}))
        seen_in_branch.add(sink)
    in_any_edge: set[str] = set(children_of) | set(parents_of)
    for orphan in summary.data_ids:
        if orphan not in in_any_edge:
            root.add(_label(orphan))
    _ = seen_in_branch  # kept for future filtering; unused now
    return root


def _critical_path_tree(summary: ProfileSummary) -> Tree:
    cp = summary.critical_path
    if cp is None:
        tree = Tree("[bold]critical path[/bold]")
        tree.add("[dim](no path: workflow has no events with timestamps)[/dim]")
        return tree

    tree = Tree(
        f"[bold]critical path[/bold]  "
        f"[green]{cp.critical_path_seconds:.3f}s[/green]"
        f"  network=[magenta]{cp.total_network_seconds:.3f}s[/magenta]"
        f"  length=[cyan]{len(cp.path)}[/cyan]"
    )
    awb = cp.active_wait_breakdown
    cursor = tree
    for data_id, active, wait in zip(awb.data_id, awb.active_seconds, awb.wait_seconds):
        wait_part = f"  [yellow]wait {wait:.3f}s[/yellow]" if wait > 0 else ""
        cursor = cursor.add(
            f"[bold red]◆[/bold red] [cyan]{data_id}[/cyan]"
            f"  [green]active {active:.3f}s[/green]{wait_part}"
        )
    return tree


def _print_header(summary: ProfileSummary) -> None:
    e2e = summary.e2e_breakdown
    headline = (
        f"[bold]workflow:[/bold] {summary.workflow_id or '(unnamed)'}\n"
        f"wall=[bold green]{e2e.workflow_duration_seconds:.3f}s[/bold green]"
        f"  network=[bold magenta]{e2e.total_network_seconds:.3f}s[/bold magenta]"
        f"  data_ids=[bold cyan]{len(summary.data_ids)}[/bold cyan]"
        f"  events=[bold cyan]{summary.event_count}[/bold cyan]"
        f"  assets=[bold cyan]{len(summary.assets)}[/bold cyan]"
    )
    console.print(Panel(headline, title="profile", border_style="cyan"))


def _print_critical_path(summary: ProfileSummary) -> None:
    if summary.critical_path is None:
        console.print("[dim](no critical path: no events with timestamps)[/dim]")
        return
    console.print(_critical_path_tree(summary))
    console.print(
        _compute_table(
            summary.critical_path.hardware_summary, "Compute time (critical path)"
        )
    )
    console.print(
        _network_table(
            summary.critical_path.network_summary, "Network time (critical path)"
        )
    )


def _print_e2e(summary: ProfileSummary) -> None:
    e2e = summary.e2e_breakdown
    console.print(_compute_table(e2e.hardware_summary, "Compute time (end-to-end)"))
    console.print(_network_table(e2e.network_summary, "Network time (end-to-end)"))


def _print_dag(summary: ProfileSummary) -> None:
    console.print(_lineage_tree(summary))


_FORMAT_HELP = (
    "Output format: rich (default; everything), critical-path (cp), " "e2e, dag, json"
)
_VIEW_ALIASES = {
    "rich": "rich",
    "all": "rich",
    "default": "rich",
    "critical-path": "critical-path",
    "critical_path": "critical-path",
    "cp": "critical-path",
    "e2e": "e2e",
    "compute": "e2e",
    "summary": "e2e",
    "dag": "dag",
    "lineage": "dag",
    "graph": "dag",
    "json": "json",
}


@app.command("fetch")
def fetch(
    workflow_id: str = typer.Argument(..., help="Workflow identifier"),
    fmt: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help=_FORMAT_HELP,
        case_sensitive=False,
    ),
) -> None:
    """Run the trace analyzer on a workflow and render the result."""
    client = flowmesh_client_from_config()
    try:
        payload = client.profile.fetch(workflow_id)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)

    view = _VIEW_ALIASES.get(fmt.lower())
    if view is None:
        logging.error(
            f"Unknown format '{fmt}'; expected one of: "
            "rich, critical-path, e2e, dag, json"
        )
        raise typer.Exit(code=2)

    if view == "json":
        console.print(RichJSON.from_data(payload))
        return

    summary = _to_summary(payload)
    _print_header(summary)

    if view == "rich":
        if summary.critical_path is not None:
            _print_critical_path(summary)
            console.print(Rule(style="dim"))
        _print_e2e(summary)
        console.print(Rule(style="dim"))
        _print_dag(summary)
        return
    if view == "critical-path":
        _print_critical_path(summary)
        return
    if view == "e2e":
        _print_e2e(summary)
        return
    if view == "dag":
        _print_dag(summary)
        return
