import json

import typer
from flowmesh import FlowMesh
from flowmesh.exceptions import FlowMeshError
from flowmesh.params import append_param, extend_params

from ..core import logging
from ..core.query import parse_query_filters
from ..core.typer import get_typer

app = get_typer(help="Manage nodes registered with FlowMesh.")


@app.command()
def info(node_id: str = typer.Argument(..., help="Node identifier")) -> None:
    """Retrieve information for a specific node."""
    client = FlowMesh()
    try:
        node = client.nodes.retrieve(node_id)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(node.model_dump_json(indent=2))


@app.command("list")
def list_nodes(
    node_id: str | None = typer.Option(None, "--id", help="Filter by node id"),
    namespace: str | None = typer.Option(
        None, "--namespace", help="Filter by node namespace"
    ),
    cluster: str | None = typer.Option(
        None, "--cluster", help="Filter by node cluster"
    ),
    alias: str | None = typer.Option(None, "--alias", help="Filter by node alias"),
    tag: list[str] | None = typer.Option(
        None, "--tag", help="Filter by tag (repeatable)"
    ),
    query: list[str] | None = typer.Option(
        None, "--query", "-q", help="Filter nodes by key=value pairs"
    ),
) -> None:
    """List all nodes registered with FlowMesh."""
    client = FlowMesh()
    query_params = parse_query_filters(query)
    try:
        nodes = client.nodes.list(
            node_id=node_id,
            namespace=namespace,
            cluster=cluster,
            alias=alias,
            tags=tag or None,
            query_params=query_params,
        )
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps([s.model_dump(mode="json") for s in nodes], indent=2))


worker_app = get_typer(help="Manage workers on nodes.")
app.add_typer(worker_app, name="worker")


@worker_app.command("list")
def list_workers(
    node_id: str | None = typer.Argument(None, help="Node identifier"),
    worker_id: str | None = typer.Option(None, "--id", help="Filter by worker id"),
    name: str | None = typer.Option(None, "--name", help="Filter by worker name"),
    namespace: str | None = typer.Option(
        None, "--namespace", help="Filter by worker namespace"
    ),
    cluster: str | None = typer.Option(
        None, "--cluster", help="Filter by worker cluster"
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Filter by worker provider"
    ),
    status: list[str] | None = typer.Option(
        None, "--status", "-s", help="Filter by status (repeatable)"
    ),
    node_id_filter: str | None = typer.Option(
        None, "--node-id", help="Filter by associated node id"
    ),
    node_alias: str | None = typer.Option(
        None, "--node-alias", help="Filter by associated node alias"
    ),
    query: list[str] | None = typer.Option(
        None, "--query", "-q", help="Filter workers by key=value pairs"
    ),
) -> None:
    """List workers on a specific node or all nodes."""
    client = FlowMesh()
    query_params = parse_query_filters(query)
    append_param(query_params, "id", worker_id)
    append_param(query_params, "name", name)
    append_param(query_params, "namespace", namespace)
    append_param(query_params, "cluster", cluster)
    append_param(query_params, "provider", provider)
    extend_params(query_params, "status", status)
    append_param(query_params, "node_id", node_id_filter)
    append_param(query_params, "node_alias", node_alias)
    try:
        if node_id is None:
            workers = client.nodes.list_all_workers(query_params=query_params)
        else:
            workers = client.nodes.list_workers(node_id, query_params=query_params)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps([w.model_dump(mode="json") for w in workers], indent=2))


@worker_app.command("start")
def start_worker(
    node_id: str = typer.Argument(..., help="Node identifier"),
    worker_name: str = typer.Argument(..., help="Worker name"),
) -> None:
    """Start a worker on a specific node."""
    client = FlowMesh()
    try:
        client.nodes.start_worker(node_id, worker_name)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.success(f"Worker '{worker_name}' started on node '{node_id}'")


@worker_app.command("stop")
def stop_worker(
    node_id: str = typer.Argument(..., help="Node identifier"),
    worker_name: str = typer.Argument(..., help="Worker name"),
) -> None:
    """Stop a worker on a specific node."""
    client = FlowMesh()
    try:
        client.nodes.stop_worker(node_id, worker_name)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.success(f"Worker '{worker_name}' stopped on node '{node_id}'")
