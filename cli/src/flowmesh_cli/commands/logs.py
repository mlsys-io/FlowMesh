"""`flowmesh logs fetch ...` — workflow-scoped lineage retrieval."""

import json
from pathlib import Path

import typer
from flowmesh.exceptions import FlowMeshError

from ..core import logging
from ..core.runtime import flowmesh_client_from_config
from ..core.typer import get_typer

app = get_typer(help="Fetch workflow lineage rows (spans / assets / lineage).")


def _fetch_kind(workflow_id: str, kind: str, output: Path | None) -> None:
    client = flowmesh_client_from_config()
    try:
        rows = client.logs.fetch(workflow_id, kind)  # type: ignore[arg-type]
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)

    if output is None:
        for row in rows:
            logging.log(json.dumps(row, ensure_ascii=False))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    logging.log(f"Wrote {count} {kind} rows to {output}")


@app.command("fetch")
def fetch(
    kind: str = typer.Argument(
        ..., help="One of: spans, assets, lineage", metavar="KIND"
    ),
    workflow_id: str = typer.Argument(..., help="Workflow identifier"),
    output: Path | None = typer.Option(
        None, "--out", "-o", help="Write rows to this JSONL file (default: stdout)"
    ),
) -> None:
    """Fetch JSONL rows for a workflow's spans / assets / lineage."""
    if kind not in {"spans", "assets", "lineage"}:
        logging.error(f"Unknown kind '{kind}'; expected one of: spans, assets, lineage")
        raise typer.Exit(code=2)
    _fetch_kind(workflow_id, kind, output)
