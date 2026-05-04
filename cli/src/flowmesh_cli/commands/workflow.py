import json
import time
from pathlib import Path

import typer
from flowmesh import FlowMesh
from flowmesh.exceptions import FlowMeshError
from flowmesh.models.common import TERMINAL_WORKFLOW_STATUSES, WorkflowStatus
from flowmesh.resources.workflows import WorkflowFormat

from ..core import logging
from ..core.query import parse_query_filters
from ..core.typer import get_typer

app = get_typer(help="Submit and manage workflows.")


def _format_log_event(event: dict) -> str:
    ts = event.get("ts", "")
    task_id = event.get("task_id", "")
    message = str(event.get("message", "")).rstrip("\n")

    if ts and task_id:
        prefix = f"[{ts} | {task_id}] "
    elif ts:
        prefix = f"[{ts}] "
    elif task_id:
        prefix = f"[{task_id}] "
    else:
        prefix = ""
    return prefix + message


@app.command()
def submit(
    template: Path = typer.Argument(..., help="Path to workflow template YAML"),
    workflow_format: WorkflowFormat = typer.Option(
        "native", "--format", help="Format of the workflow template (native|n8n)"
    ),
) -> None:
    """Submit a workflow definition from a YAML template to the FlowMesh server."""
    if not template.exists():
        logging.error(f"Template not found: {template}")
        raise typer.Exit(code=1)
    workflow_text = template.read_text()
    client = FlowMesh()
    try:
        result = client.workflows.submit(workflow_text, workflow_format=workflow_format)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(result.model_dump_json(indent=2))


@app.command()
def validate(
    template: Path = typer.Argument(..., help="Path to workflow template YAML"),
    workflow_format: WorkflowFormat = typer.Option(
        "native", "--format", help="Format of the workflow template (native|n8n)"
    ),
) -> None:
    """Validate a workflow definition from a YAML template without submitting."""
    if not template.exists():
        logging.error(f"Template not found: {template}")
        raise typer.Exit(code=1)
    workflow_text = template.read_text()
    client = FlowMesh()
    try:
        result = client.workflows.validate(
            workflow_text, workflow_format=workflow_format
        )
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(result.model_dump_json(indent=2))


@app.command()
def info(
    workflow_id: str = typer.Argument(..., help="ID of the workflow to retrieve")
) -> None:
    """Retrieve information for a specific workflow."""
    client = FlowMesh()
    try:
        wf = client.workflows.retrieve(workflow_id)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(wf.model_dump_json(indent=2))


@app.command("list")
def list_workflows(
    workflow_id: str | None = typer.Option(None, "--id", help="Filter by workflow id"),
    status: list[str] | None = typer.Option(
        None, "--status", "-s", help="Filter by status (repeatable)"
    ),
    task_id: list[str] | None = typer.Option(
        None,
        "--task-id",
        help="Filter by task id in workflow (repeatable)",
    ),
    query: list[str] | None = typer.Option(
        None, "--query", "-q", help="Filter workflows by key=value pairs"
    ),
) -> None:
    """List all workflows submitted to the FlowMesh server."""
    client = FlowMesh()
    query_params = parse_query_filters(query)
    try:
        workflows = client.workflows.list(
            workflow_id=workflow_id,
            status=status or None,
            task_ids=task_id or None,
            query_params=query_params,
        )
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps([w.model_dump(mode="json") for w in workflows], indent=2))


@app.command()
def watch(
    workflow_id: str = typer.Argument(..., help="ID of the workflow to monitor"),
    interval: float = typer.Option(2.0, help="Polling interval in seconds"),
) -> None:
    """Monitor a workflow's progress by polling until completion."""
    client = FlowMesh()
    last_status: str | None = None
    try:
        while True:
            wf = client.workflows.retrieve(workflow_id)
            status_value = WorkflowStatus(wf.status)
            if status_value != last_status:
                logging.log(f"[{workflow_id}] status: {status_value}")
                last_status = status_value
            if status_value in TERMINAL_WORKFLOW_STATUSES:
                logging.log(wf.model_dump_json(indent=2))
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.warning("Cancelled by user.")
        raise typer.Exit(code=1)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)


@app.command()
def cancel(
    workflow_id: str = typer.Argument(..., help="ID of the workflow to cancel")
) -> None:
    """Request cancellation of a running workflow and its associated tasks."""
    client = FlowMesh()
    try:
        wf = client.workflows.cancel(workflow_id)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(wf.model_dump_json(indent=2))


logs_app = get_typer(help="Query and monitor workflow logs.")
app.add_typer(logs_app, name="logs")


@logs_app.command("show")
def show_logs(
    workflow_id: str = typer.Argument(..., help="Workflow identifier"),
    limit: int = typer.Option(500, help="Maximum number of entries to return"),
    before: str | None = typer.Option(None, help="Return entries before this cursor"),
    after: str | None = typer.Option(None, help="Return entries after this cursor"),
    json_output: bool = typer.Option(
        False, "--json", help="Print raw JSON response instead of formatted lines"
    ),
) -> None:
    """Query recent workflow logs from the server."""
    client = FlowMesh()
    try:
        result = client.workflows.get_logs(
            workflow_id, limit=limit, before=before, after=after
        )
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    payload = result.model_dump(mode="json")
    if json_output:
        logging.log(json.dumps(payload, indent=2))
        return
    entries = payload.get("entries") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        event = entry.get("event")
        if not isinstance(event, dict):
            continue
        logging.log(_format_log_event(event))
    next_cursor = payload.get("next_cursor")
    prev_cursor = payload.get("prev_cursor")
    if next_cursor or prev_cursor:
        logging.log(
            json.dumps({"next_cursor": next_cursor, "prev_cursor": prev_cursor})
        )


@logs_app.command("stream")
def stream_logs(
    workflow_id: str = typer.Argument(..., help="Workflow identifier"),
    cursor: str | None = typer.Option(None, help="Start streaming after this cursor"),
) -> None:
    """Stream workflow logs via SSE."""
    client = FlowMesh()
    try:
        for entry in client.workflows.stream_logs(workflow_id, cursor=cursor):
            event = entry.event.model_dump(mode="json")
            logging.log(_format_log_event(event))
    except KeyboardInterrupt:
        logging.warning("Cancelled by user.")
        raise typer.Exit(code=1)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)


@logs_app.command("download")
def download_logs(
    workflow_id: str = typer.Argument(..., help="Workflow identifier"),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="Directory to save task logs"
    ),
) -> None:
    client = FlowMesh()
    error = True
    try:
        for path in client.workflows.download_logs(workflow_id, output_dir):
            logging.log(f"Downloaded logs to {path}")
            error = False
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    if error:
        logging.info(f"No logs downloaded for workflow {workflow_id}.")
