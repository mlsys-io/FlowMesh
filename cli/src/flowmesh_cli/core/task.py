"""Task status utilities for the FlowMesh CLI."""

import typer
from flowmesh import FlowMesh
from flowmesh.exceptions import FlowMeshError
from flowmesh.models.common import TaskStatus

from . import logging


def wait_for_task_completion(task_id: str, interval: float) -> tuple[str, str | None]:
    """Poll task until it reaches a terminal state."""
    client = FlowMesh()
    try:
        view = client.tasks.wait_status(task_id, interval=interval)
        status = view.status
        match status:
            case TaskStatus.DONE:
                return TaskStatus.DONE, None
            case TaskStatus.FAILED | TaskStatus.CANCELLED:
                error = view.error or status.lower()
                return status, error
            case _:
                return status, view.error
    except KeyboardInterrupt:
        logging.warning("Cancelled by user.")
        raise typer.Exit(code=1)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
