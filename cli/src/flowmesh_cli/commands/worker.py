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


def _set_cordon(
    cordoned: bool,
    worker_id: str | None,
    node_alias: str | None,
    alias: str | None,
) -> None:
    client = FlowMesh()
    try:
        if worker_id is not None and node_alias is None and alias is None:
            set_by_id = client.workers.cordon if cordoned else client.workers.uncordon
            result = set_by_id(worker_id)
        elif worker_id is None and node_alias and alias:
            set_by_alias = (
                client.workers.cordon_alias
                if cordoned
                else client.workers.uncordon_alias
            )
            result = set_by_alias(node_alias, alias)
        else:
            logging.error("Pass a WORKER_ID, or both --node-alias and --alias.")
            raise typer.Exit(code=2)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(result.model_dump_json(indent=2))


@app.command()
def cordon(
    worker_id: str | None = typer.Argument(None, help="Worker identifier"),
    node_alias: str | None = typer.Option(
        None, "--node-alias", help="Node alias, with --alias instead of WORKER_ID"
    ),
    alias: str | None = typer.Option(
        None, "--alias", help="Worker alias, with --node-alias instead of WORKER_ID"
    ),
) -> None:
    """Stop offering new tasks to a worker without stopping it."""
    _set_cordon(True, worker_id, node_alias, alias)


@app.command()
def uncordon(
    worker_id: str | None = typer.Argument(None, help="Worker identifier"),
    node_alias: str | None = typer.Option(
        None, "--node-alias", help="Node alias, with --alias instead of WORKER_ID"
    ),
    alias: str | None = typer.Option(
        None, "--alias", help="Worker alias, with --node-alias instead of WORKER_ID"
    ),
) -> None:
    """Allow a cordoned worker to receive tasks again."""
    _set_cordon(False, worker_id, node_alias, alias)


@app.command("cordons")
def list_cordons() -> None:
    """List cordons, including those with no worker registered."""
    client = FlowMesh()
    try:
        cordons = client.workers.list_cordons()
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps([c.model_dump(mode="json") for c in cordons], indent=2))
