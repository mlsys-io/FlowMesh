"""`flowmesh profile fetch ...` — analyze a workflow's lineage."""

import json
from typing import Any

import typer
from flowmesh.exceptions import FlowMeshError

from shared.profile import (
    ProfileSummary,
    render_mermaid,
    render_phase_timings,
    render_table,
)

from ..core import logging
from ..core.runtime import flowmesh_client_from_config
from ..core.typer import get_typer

app = get_typer(help="Profile a workflow's events / assets / lineage.")


def _to_summary(payload: dict[str, Any]) -> ProfileSummary:
    return ProfileSummary.model_validate(payload)


@app.command("fetch")
def fetch(
    workflow_id: str = typer.Argument(..., help="Workflow identifier"),
    fmt: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json, table, mermaid, phases",
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

    fmt_lower = fmt.lower()
    if fmt_lower == "json":
        logging.log(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if fmt_lower == "mermaid":
        logging.log(render_mermaid(_to_summary(payload)))
        return
    if fmt_lower == "table":
        logging.log(render_table(_to_summary(payload)))
        return
    if fmt_lower == "phases":
        logging.log(render_phase_timings(_to_summary(payload)))
        return
    logging.error(
        f"Unknown format '{fmt}'; expected one of: json, table, mermaid, phases"
    )
    raise typer.Exit(code=2)
