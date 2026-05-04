"""Public FlowMesh task commands."""

import json
import time
from pathlib import Path

import typer
from flowmesh import FlowMesh
from flowmesh.exceptions import FlowMeshError
from flowmesh.models.common import TERMINAL_TASK_STATUSES, TaskStatus

from ..core import logging
from ..core.query import parse_query_filters
from ..core.task import wait_for_task_completion
from ..core.typer import get_typer

app = get_typer(help="Query and monitor tasks executing on FlowMesh workers.")


def _format_log_event(event: dict) -> str:
    ts = event.get("ts", "")
    message = str(event.get("message", "")).rstrip("\n")
    prefix = f"[{ts}] " if ts else ""
    return prefix + message


def _log_ssh_connection_instructions(
    task_id: str, latest_update: dict, client: FlowMesh
) -> None:
    ssh_info = latest_update.get("ssh")
    if not isinstance(ssh_info, dict):
        return
    mode = str(ssh_info.get("mode") or "direct")
    logging.log(f"[{task_id}] SSH connection instructions ({mode}):")
    for label, command in client.ssh.connection_commands(task_id, ssh_info):
        logging.log(f"  {label}:")
        logging.log(f"    {command}")
    if str(ssh_info.get("mode") or "direct") == "proxy":
        logging.log("  note:")
        logging.log("    proxy ssh command requires `websocat` and $FLOWMESH_API_KEY")


@app.command()
def info(task_id: str = typer.Argument(..., help="Task identifier")) -> None:
    """Retrieve the current status and metadata for a specific task."""
    client = FlowMesh()
    try:
        task = client.tasks.retrieve(task_id)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(task.model_dump_json(indent=2))


@app.command("list")
def list_tasks(
    task_id: str | None = typer.Option(None, "--id", help="Filter by task id"),
    workflow_id: str | None = typer.Option(
        None, "--workflow-id", help="Filter by workflow id"
    ),
    status: list[str] | None = typer.Option(
        None, "--status", "-s", help="Filter by status (repeatable)"
    ),
    category: str | None = typer.Option(None, "--category", help="Filter by category"),
    task_type: str | None = typer.Option(None, "--type", help="Filter by task type"),
    assigned_worker: str | None = typer.Option(
        None, "--assigned-worker", help="Filter by assigned worker id"
    ),
    graph_node_name: str | None = typer.Option(
        None, "--graph-node", help="Filter by graph node name"
    ),
    completed: bool | None = typer.Option(
        None,
        "--completed/--not-completed",
        help="Filter by completion state",
    ),
    failed: bool | None = typer.Option(
        None,
        "--failed/--not-failed",
        help="Filter by failure state",
    ),
    query: list[str] | None = typer.Option(
        None, "--query", "-q", help="Filter tasks by key=value pairs"
    ),
) -> None:
    """List all tasks registered in the FlowMesh server."""
    client = FlowMesh()
    query_params = parse_query_filters(query)
    try:
        tasks = client.tasks.list(
            task_id=task_id,
            workflow_id=workflow_id,
            status=status or None,
            category=category,
            task_type=task_type,
            assigned_worker=assigned_worker,
            graph_node_name=graph_node_name,
            completed=completed,
            failed=failed,
            query_params=query_params,
        )
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps([t.model_dump(mode="json") for t in tasks], indent=2))


@app.command()
def stop(
    task_id: str = typer.Argument(..., help="Task identifier"),
    interval: float = typer.Option(2.0, help="Polling interval in seconds"),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Return immediately after requesting stop"
    ),
) -> None:
    """Stop a running task."""
    client = FlowMesh()
    try:
        client.tasks.stop(task_id)
    except FlowMeshError as exc:
        logging.error(f"Failed to stop task: {exc}")
        raise typer.Exit(code=1)

    message = f"Stop requested for task {task_id}."
    if no_wait:
        logging.log(message)
        return

    logging.info(message)
    status, error = wait_for_task_completion(task_id, interval)
    match status:
        case TaskStatus.DONE:
            logging.success("Task stopped successfully.")
        case TaskStatus.FAILED | TaskStatus.CANCELLED:
            logging.error(f"Task {status.lower()}: {error}")
            raise typer.Exit(code=1)
        case _:
            logging.error(f"Unexpected task status: {status}")
            raise typer.Exit(code=1)


@app.command()
def watch(
    task_id: str = typer.Argument(..., help="Task identifier"),
    interval: float = typer.Option(2.0, help="Polling interval in seconds"),
) -> None:
    """Monitor a task's status by polling until completion."""
    client = FlowMesh()
    last_status: TaskStatus | None = None
    last_update: dict | None = None
    try:
        while True:
            task = client.tasks.retrieve(task_id)
            payload = task.model_dump(mode="json")
            status_value = task.status
            if status_value != last_status:
                logging.log(f"[{task_id}] status: {status_value}")
                last_status = status_value
            latest_update = payload.get("latest_update")
            if isinstance(latest_update, dict) and latest_update != last_update:
                logging.log(json.dumps(latest_update, indent=2))
                _log_ssh_connection_instructions(task_id, latest_update, client)
                last_update = latest_update
            if status_value in TERMINAL_TASK_STATUSES:
                logging.log(json.dumps(payload, indent=2))
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.warning("Cancelled by user.")
        raise typer.Exit(code=1)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)


logs_app = get_typer(help="Query and monitor task logs.")
app.add_typer(logs_app, name="logs")


@logs_app.command("show")
def show_logs(
    task_id: str = typer.Argument(..., help="Task identifier"),
    limit: int = typer.Option(200, help="Maximum number of entries to return"),
    before: str | None = typer.Option(None, help="Return entries before this cursor"),
    after: str | None = typer.Option(None, help="Return entries after this cursor"),
    json_output: bool = typer.Option(
        False, "--json", help="Print raw JSON response instead of formatted lines"
    ),
) -> None:
    """Query recent task logs from the server."""
    client = FlowMesh()
    try:
        result = client.tasks.get_logs(task_id, limit=limit, before=before, after=after)
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
    task_id: str = typer.Argument(..., help="Task identifier"),
    cursor: str | None = typer.Option(None, help="Start streaming after this cursor"),
) -> None:
    """Stream task logs via SSE."""
    client = FlowMesh()
    try:
        for entry in client.tasks.stream_logs(task_id, cursor=cursor):
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
    task_id: str = typer.Argument(..., help="Task identifier"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save logs.jsonl"),
) -> None:
    """Download archived logs.jsonl for a task."""
    client = FlowMesh()
    try:
        client.tasks.download_logs(task_id, output)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    except OSError as exc:
        logging.error(f"Failed to write {output}: {exc}")
        raise typer.Exit(code=1)
    logging.log(f"Wrote logs to {output}")
