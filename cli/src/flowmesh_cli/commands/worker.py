import json

import typer
from flowmesh import FlowMesh
from flowmesh.exceptions import FlowMeshError
from flowmesh.params import append_param

from ..core import logging
from ..core.query import parse_query_filters
from ..core.typer import get_typer

app = get_typer(help="Query and manage workers across all servers via the server API.")


@app.command()
def info(worker_id: str = typer.Argument(..., help="Worker identifier")) -> None:
    """Retrieve information for a specific worker."""
    client = FlowMesh()
    try:
        worker = client.workers.retrieve(worker_id)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(worker.model_dump_json(indent=2))


@app.command("list")
def list_workers(
    worker_id: str | None = typer.Option(None, "--id", help="Filter by worker id"),
    alias: str | None = typer.Option(None, "--alias", help="Filter by worker alias"),
    namespace: str | None = typer.Option(
        None, "--namespace", help="Filter by worker namespace"
    ),
    cluster: str | None = typer.Option(
        None, "--cluster", help="Filter by worker cluster"
    ),
    node_id: str | None = typer.Option(
        None, "--node-id", help="Filter by owning node id"
    ),
    node_alias: str | None = typer.Option(
        None, "--node-alias", help="Filter by owning node alias"
    ),
    status: list[str] | None = typer.Option(
        None, "--status", "-s", help="Filter by status (repeatable)"
    ),
    tag: list[str] | None = typer.Option(
        None, "--tag", help="Filter by tag (repeatable)"
    ),
    stale: bool | None = typer.Option(
        None, "--stale/--not-stale", help="Filter by stale heartbeat state"
    ),
    query: list[str] | None = typer.Option(
        None, "--query", "-q", help="Filter workers by key=value pairs"
    ),
) -> None:
    """List all workers."""
    client = FlowMesh()
    query_params = parse_query_filters(query)
    append_param(query_params, "node_id", node_id)
    append_param(query_params, "node_alias", node_alias)
    try:
        workers = client.workers.list(
            worker_id=worker_id,
            alias=alias,
            namespace=namespace,
            cluster=cluster,
            status=status or None,
            tags=tag or None,
            stale=stale,
            query_params=query_params,
        )
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps([w.model_dump(mode="json") for w in workers], indent=2))
